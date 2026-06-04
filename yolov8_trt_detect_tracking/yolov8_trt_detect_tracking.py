#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Point32
from geometry_msgs.msg import PolygonStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from collections import deque
import copy
import os
import pycuda.driver as cuda
import pycuda.autoinit

# 导入TensorRT版本的模型
from .YOLOv8_TensorRT import YOLOv8TensorRT  # 修正类名
from .BOTSort_rdk import BOTSORT


class SharedCUDAManager:
    """共享CUDA上下文管理器 - 解决上下文冲突"""
    _instance = None
    _context = None
    _ref_count = 0
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def acquire(self):
        """获取CUDA上下文"""
        if self._context is None:
            device = cuda.Device(0)
            self._context = device.retain_primary_context()
            self._context.push()
        self._ref_count += 1
        return self._context
    
    def release(self):
        """释放CUDA上下文"""
        self._ref_count -= 1
        if self._ref_count <= 0 and self._context:
            self._context.pop()
            self._context = None
            self._ref_count = 0


class TrackedTarget:
    """跟踪目标信息类 - 使用__slots__优化内存"""
    __slots__ = ('track_id', 'bbox', 'feature', 'height_pixels', 'first_seen_time', 
                 'last_seen_time', 'last_update_time', 'lost_frames', 'is_recovered', 
                 'is_switched', 'original_track_id', 'recovery_time', 'update_paused')
    
    def __init__(self, track_id: int, bbox: List[float], feature: np.ndarray, 
                 height_pixels: float, timestamp: float):
        self.track_id = track_id
        self.bbox = bbox
        self.feature = feature
        self.height_pixels = height_pixels
        self.first_seen_time = timestamp
        self.last_seen_time = timestamp
        self.last_update_time = timestamp
        self.lost_frames = 0
        self.is_recovered = False
        self.is_switched = False
        self.original_track_id = track_id
        self.recovery_time = None
        self.update_paused = False
    
    def update(self, bbox: List[float], feature: np.ndarray, height_pixels: float, timestamp: float):
        """更新目标信息"""
        self.bbox = bbox
        self.feature = feature
        self.height_pixels = height_pixels
        self.last_seen_time = timestamp
        self.last_update_time = timestamp
        self.lost_frames = 0
        self.is_recovered = False
    
    def mark_lost(self):
        """标记目标丢失"""
        self.lost_frames += 1
        self.update_paused = True
        self.recovery_time = None
    
    def mark_recovered(self, timestamp: float):
        """标记目标找回"""
        self.is_recovered = True
        self.lost_frames = 0
        self.recovery_time = None
        self.update_paused = False
    
    def switch_to_new_id(self, new_track_id: int):
        """切换到新的跟踪ID"""
        self.is_switched = True
        self.track_id = new_track_id


class Yolov8HandTrackNode(Node):
    def __init__(self):
        super().__init__('Yolov8HandTrackNode')
        
        # 共享CUDA管理器
        self.cuda_manager = SharedCUDAManager()
        self.cuda_manager.acquire()

        # 声明参数
        self._declare_parameters()
        
        # 获取参数
        self._get_parameters()
        
        self.min_process_interval = 1.0 / self.max_processing_fps
        self.last_process_time = time.time()

        # 初始化模型和组件
        self._initialize_components()
        
        # 初始化变量
        self._initialize_variables()
        
        self.get_logger().info("YOLOv8 Hand Track Node initialized with TensorRT (Optimized)")
        self.print_parameters()

    def _declare_parameters(self):
        """声明所有参数"""
        self.declare_parameter('model_path', '/home/wheeltec/Ebike_Human_Follower/src/yolov8_trt_detect_tracking/models/yolov8.engine')
        self.declare_parameter('reid_engine_path', '/home/wheeltec/Ebike_Human_Follower/src/yolov8_trt_detect_tracking/models/osnet_x0_25.engine')
        self.declare_parameter('conf_threshold', 0.3)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('max_processing_fps', 15)
        self.declare_parameter('ok_confirm_frames', 10)
        self.declare_parameter('tracking_protection_time', 5.0)
        self.declare_parameter('reid_similarity_threshold', 0.8)
        self.declare_parameter('height_change_threshold', 0.15)
        self.declare_parameter('lost_timeout_threshold', 10.0)
        self.declare_parameter('roi_threshold', 0.5)
        self.declare_parameter('use_cuda', True)
        self.declare_parameter('input_size', 640)
        self.declare_parameter('with_reid', True)  # 是否启用ReID
        
    def _get_parameters(self):
        """获取所有参数值"""
        self.model_path = self.get_parameter('model_path').value
        self.reid_engine_path = self.get_parameter('reid_engine_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.nms_threshold = self.get_parameter('nms_threshold').value
        self.max_processing_fps = self.get_parameter('max_processing_fps').value
        self.ok_confirm_frames = self.get_parameter('ok_confirm_frames').value
        self.tracking_protection_time = self.get_parameter('tracking_protection_time').value
        self.reid_similarity_threshold = self.get_parameter('reid_similarity_threshold').value
        self.height_change_threshold = self.get_parameter('height_change_threshold').value
        self.lost_timeout_threshold = self.get_parameter('lost_timeout_threshold').value
        self.roi_threshold = self.get_parameter('roi_threshold').value
        self.use_cuda = self.get_parameter('use_cuda').value
        self.input_size = self.get_parameter('input_size').value
        self.with_reid = self.get_parameter('with_reid').value

    def print_parameters(self):
        """打印参数信息"""
        self.get_logger().info("===== 参数配置信息 (TensorRT) =====")
        self.get_logger().info(f"YOLO TensorRT引擎: {self.model_path}")
        self.get_logger().info(f"ReID TensorRT引擎: {self.reid_engine_path}")
        self.get_logger().info(f"置信度阈值: {self.conf_threshold}")
        self.get_logger().info(f"NMS阈值: {self.nms_threshold}")
        self.get_logger().info(f"最大处理帧率: {self.max_processing_fps}FPS")
        self.get_logger().info(f"OK手势确认帧数: {self.ok_confirm_frames}")
        self.get_logger().info(f"跟踪保护时间: {self.tracking_protection_time}s")
        self.get_logger().info(f"ReID相似度阈值: {self.reid_similarity_threshold}")
        self.get_logger().info(f"高度变化阈值: {self.height_change_threshold}")
        self.get_logger().info(f"丢失超时阈值: {self.lost_timeout_threshold}s")
        self.get_logger().info(f"ROI重叠阈值: {self.roi_threshold}")
        self.get_logger().info(f"YOLO输入尺寸: {self.input_size}")
        self.get_logger().info(f"启用ReID: {self.with_reid}")
        self.get_logger().info("=========================")

    def _initialize_components(self):
        """初始化模型和跟踪器（TensorRT版本）"""
        # 加载YOLOv8 TensorRT模型
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"YOLO TensorRT引擎文件不存在: {self.model_path}")
            return
        
        try:
            self.model = YOLOv8TensorRT(
                self.model_path, 
                self.conf_threshold, 
                self.nms_threshold,
                shared_context=True  # 使用共享上下文模式
            )
            self.get_logger().info("✅ YOLO TensorRT模型初始化成功")
        except Exception as e:
            self.get_logger().error(f"YOLO TensorRT模型初始化失败: {e}")
            return
        
        # 初始化BOTSORT跟踪器（TensorRT ReID）
        reid_enabled = self.with_reid and os.path.exists(self.reid_engine_path)
        
        # 检查ReID引擎文件
        if self.with_reid and not os.path.exists(self.reid_engine_path):
            self.get_logger().warning(f"⚠️ ReID引擎文件不存在: {self.reid_engine_path}")
            self.get_logger().warning("   请检查路径或复制文件到正确位置")
            reid_enabled = False
        
        # 重要：如果启用ReID，等待一下让YOLO完全初始化
        if reid_enabled:
            time.sleep(0.5)  # 给GPU一些时间完成YOLO初始化
            
        tracker_args = {
            'track_high_thresh': 0.25,
            'track_low_thresh': 0.1,
            'new_track_thresh': 0.25,
            'track_buffer': 30,
            'match_thresh': 0.8,
            'fuse_score': True,
            'proximity_thresh': 0.5,
            'appearance_thresh': 0.8,
            'with_reid': reid_enabled,
            'reid_engine_path': self.reid_engine_path,
            'reid_input_size': (256, 128)
        }
        
        try:
            self.tracker = BOTSORT(tracker_args)
            if reid_enabled:
                self.get_logger().info("✅ TensorRT ReID模型已启用")
            else:
                self.get_logger().warning("⚠️ ReID模型未启用（引擎文件不存在或参数关闭）")
        except Exception as e:
            self.get_logger().error(f"Tracker初始化失败: {e}")
            # 降级模式：不使用ReID
            tracker_args['with_reid'] = False
            self.tracker = BOTSORT(tracker_args)
            self.get_logger().warning("⚠️ 已降级到不使用ReID的跟踪模式")
        
        # 初始化CV bridge
        self.bridge = CvBridge()
        
        # 创建订阅和发布
        self.image_sub = self.create_subscription(Image, '/LxCamera_Rgb', self.image_callback, 10)
        self.detect_pose_pub = self.create_publisher(Image, 'tracks', 10)
        self.keypoint_tracks_pub = self.create_publisher(PolygonStamped, '/keypoint_tracks', 10)

    def _initialize_variables(self):
        """初始化变量"""
        self.tracked_persons: Dict[int, Dict] = {}
        self.ok_gesture_count: Dict[int, int] = {}
        self.stop_gesture_count: Dict[int, int] = {}
        self.current_tracking_id = None
        self.tracked_targets: Dict[int, TrackedTarget] = {}
        self.target_lost_time: Optional[float] = None
        self.class_names = {0: "person", 1: "ok", 2: "stop"}

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        return iou

    def find_person_for_gesture(self, gesture_box, person_boxes):
        """为手势找到对应的人体边界框"""
        gx1, gy1, gx2, gy2 = gesture_box
        gesture_center_x = (gx1 + gx2) / 2
        gesture_center_y = (gy1 + gy2) / 2
        
        best_person_box = None
        best_person_id = None
        min_distance = float('inf')
        
        for person_id, person_box in person_boxes.items():
            px1, py1, px2, py2 = person_box
            
            if (px1 <= gesture_center_x <= px2 and 
                py1 <= gesture_center_y <= py2):
                
                person_center_x = (px1 + px2) / 2
                person_center_y = (py1 + py2) / 2
                distance = ((gesture_center_x - person_center_x) ** 2 + 
                           (gesture_center_y - person_center_y) ** 2) ** 0.5
                
                if distance < min_distance:
                    min_distance = distance
                    best_person_box = person_box
                    best_person_id = person_id
        
        if best_person_id is not None:
            return best_person_id, best_person_box, 1.0
        
        return self.find_person_for_gesture_fallback(gesture_box, person_boxes)

    def find_person_for_gesture_fallback(self, gesture_box, person_boxes):
        """备选方法：使用重叠比例"""
        best_overlap = 0
        best_person_box = None
        best_person_id = None
        
        gx1, gy1, gx2, gy2 = gesture_box
        gesture_area = (gx2 - gx1) * (gy2 - gy1)
        
        for person_id, person_box in person_boxes.items():
            px1, py1, px2, py2 = person_box
            
            inter_x1 = max(gx1, px1)
            inter_y1 = max(gy1, py1)
            inter_x2 = min(gx2, px2)
            inter_y2 = min(gy2, py2)
            
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            overlap = inter_area / gesture_area if gesture_area > 0 else 0
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_person_box = person_box
                best_person_id = person_id
        
        return best_person_id, best_person_box, best_overlap

    def extract_feature_from_bbox(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """从边界框提取特征 - 使用TensorRT ReID"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(512, dtype=np.float32)
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(512, dtype=np.float32)
        
        try:
            if hasattr(self.tracker, 'encoder') and self.tracker.encoder is not None:
                return self.tracker.encoder.extract_feature(crop)
            else:
                return np.zeros(512, dtype=np.float32)
        except Exception as e:
            self.get_logger().debug(f"Feature extraction failed: {e}")
            return np.zeros(512, dtype=np.float32)
        
    def save_tracked_target(self, track_id: int, bbox: List[float], image: np.ndarray, timestamp: float):
        """保存跟踪目标信息"""
        if track_id not in self.tracked_targets:
            feature = self.extract_feature_from_bbox(image, bbox)
            height_pixels = bbox[3] - bbox[1]
            self.tracked_targets[track_id] = TrackedTarget(track_id, bbox, feature, height_pixels, timestamp)
            return
        
        target = self.tracked_targets[track_id]
        
        if target.update_paused:
            target.bbox = bbox
            target.height_pixels = bbox[3] - bbox[1]
            target.last_seen_time = timestamp
            target.lost_frames = 0
            return
        
        feature = self.extract_feature_from_bbox(image, bbox)
        height_pixels = bbox[3] - bbox[1]
        target.update(bbox, feature, height_pixels, timestamp)

    def try_recover_lost_target(self, current_tracks: List[Dict], image: np.ndarray, timestamp: float) -> Optional[int]:
        """立即尝试找回丢失的跟踪目标"""
        if not self.with_reid:
            return None
            
        if self.current_tracking_id is None or self.current_tracking_id not in self.tracked_targets:
            return None
        
        target = self.tracked_targets[self.current_tracking_id]
        
        if self.target_lost_time is None:
            self.target_lost_time = timestamp
            self.get_logger().info(f"目标 {self.current_tracking_id} 丢失，开始立即ReID匹配找回")
        
        candidate_tracks = []
        for track in current_tracks:
            track_id = track['track_id']
            is_currently_tracked = (
                track_id in self.tracked_persons and 
                self.tracked_persons[track_id].get('is_tracking', False) and
                track_id != self.current_tracking_id
            )
            
            if not is_currently_tracked:
                candidate_tracks.append(track)
        
        if not candidate_tracks:
            return None
        
        target_height_pixels = target.height_pixels
        self.get_logger().info(f"目标 {self.current_tracking_id} 丢失时高度: {target_height_pixels:.1f}px")
        
        best_match_id = None
        best_similarity = 0.0
        
        for track in candidate_tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            
            x1, y1, x2, y2 = bbox
            candidate_height_pixels = y2 - y1
            
            height_ratio = candidate_height_pixels / target_height_pixels
            height_change = abs(1.0 - height_ratio)
            
            if height_change > self.height_change_threshold:
                continue
            
            candidate_feature = self.extract_feature_from_bbox(image, bbox)
            
            if candidate_feature is not None and np.any(candidate_feature):
                similarity = np.dot(target.feature, candidate_feature) / (
                    np.linalg.norm(target.feature) * np.linalg.norm(candidate_feature) + 1e-8
                )
                
                self.get_logger().info(f"候选目标 ID:{track_id} ReID相似度: {similarity:.3f}, 高度变化: {height_change:.3f}")
                
                if similarity >= self.reid_similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = track_id
        
        if best_match_id is not None:
            self.get_logger().info(
                f"目标 {self.current_tracking_id} ReID找回成功! 匹配ID: {best_match_id}, 相似度: {best_similarity:.3f}"
            )
            
            target_bbox = next(t['bbox'] for t in candidate_tracks if t['track_id'] == best_match_id)
            target.mark_recovered(timestamp)
            target.update_paused = False
            target.recovery_time = None
            
            self.save_tracked_target(self.current_tracking_id, target_bbox, image, timestamp)
            
            recovered_id = best_match_id
            
            if best_match_id != self.current_tracking_id:
                if best_match_id in self.tracked_targets:
                    self.tracked_targets[best_match_id].switch_to_new_id(best_match_id)
                    self.tracked_targets[best_match_id].original_track_id = self.current_tracking_id
                    self.tracked_targets[best_match_id].mark_recovered(timestamp)
                    self.tracked_targets[best_match_id].update_paused = False
                    self.tracked_targets[best_match_id].recovery_time = None
            
            self.target_lost_time = None
            
            if recovered_id in self.tracked_persons:
                self.tracked_persons[recovered_id]['is_tracking'] = True
                self.tracked_persons[recovered_id]['tracking_start_time'] = timestamp
                self.tracked_persons[recovered_id]['last_seen_time'] = timestamp
            
            return recovered_id
        
        self.get_logger().warning(f"目标 {self.current_tracking_id} ReID找回失败")
        return None

    def _verify_target_with_reid(self, target: TrackedTarget, track: Dict, image: np.ndarray, timestamp: float) -> Optional[int]:
        """使用ReID验证目标身份"""
        if not self.with_reid:
            return track['track_id']
            
        track_id = track['track_id']
        bbox = track['bbox']
        
        x1, y1, x2, y2 = bbox
        candidate_height_pixels = y2 - y1
        target_height_pixels = target.height_pixels
        
        height_ratio = candidate_height_pixels / target_height_pixels
        height_change = abs(1.0 - height_ratio)
        
        if height_change > self.height_change_threshold:
            self.get_logger().warning(f"验证目标 ID:{track_id} 高度变化 {height_change:.3f} 超过阈值, 验证失败")
            return None
        
        candidate_feature = self.extract_feature_from_bbox(image, bbox)
        
        if candidate_feature is not None and np.any(candidate_feature):
            similarity = np.dot(target.feature, candidate_feature) / (
                np.linalg.norm(target.feature) * np.linalg.norm(candidate_feature) + 1e-8
            )
            
            if similarity >= self.reid_similarity_threshold:
                self.get_logger().info(f"ReID验证成功: ID {track_id}, 相似度: {similarity:.3f}, 高度变化: {height_change:.3f}")
                
                target.mark_recovered(timestamp)
                target.update_paused = False
                target.recovery_time = None
                
                self.save_tracked_target(target.track_id, bbox, image, timestamp)
                self.target_lost_time = None
                
                if track_id in self.tracked_persons:
                    self.tracked_persons[track_id]['is_tracking'] = True
                    self.tracked_persons[track_id]['last_seen_time'] = timestamp
                    if track_id == target.track_id:
                        pass
                    else:
                        self.tracked_persons[track_id]['tracking_start_time'] = timestamp
                
                return track_id
            else:
                self.get_logger().warning(f"ReID验证失败: ID {track_id}, 相似度: {similarity:.3f}")

        return None

    def image_callback(self, msg):
        """图像回调 - 使用TensorRT推理"""
        current_time = time.time()
        if current_time - self.last_process_time < self.min_process_interval:
            return
        
        self.last_process_time = current_time

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 使用TensorRT YOLO模型检测
            results, inference_time = self.model.detect(cv_image)
            
            # 记录检测耗时
            self.get_logger().debug(f"YOLO推理时间: {inference_time:.2f}ms")

            # 分离检测结果
            person_detections = []
            ok_gestures = []
            stop_gestures = []
            
            for class_id, score, x1, y1, x2, y2 in results:
                if class_id == 0:  # person
                    person_detections.append([x1, y1, x2-x1, y2-y1, score, 0])
                elif class_id == 1:  # ok手势
                    ok_gestures.append((x1, y1, x2, y2, score))
                elif class_id == 2:  # stop手势
                    stop_gestures.append((x1, y1, x2, y2, score))

            
            # 跟踪person检测结果
            tracking_results = self.tracker.update(person_detections, cv_image)

            # 处理跟踪结果
            tracks = []
            person_boxes = {}
            
            for result in tracking_results:
                x, y, w, h, track_id, score, cls, _, _ = result
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                
                track_data = {
                    'track_id': int(track_id),
                    'bbox': [x1, y1, x2, y2],
                    'conf': float(score),
                }
                tracks.append(track_data)
                person_boxes[track_id] = (x1, y1, x2, y2)

            # 更新跟踪状态
            self._update_tracking_state(tracks, cv_image, current_time, ok_gestures, stop_gestures, person_boxes)
            
            # 清理长时间未出现的跟踪目标
            self._cleanup_old_tracks(current_time, set(track['track_id'] for track in tracks))

            # 可视化并发布结果
            self._publish_results(cv_image, tracks, msg.header, ok_gestures, stop_gestures)
                   
        except Exception as e:
            self.get_logger().error(f"Image processing error: {str(e)}")
            import traceback
            traceback.print_exc()

    def _update_tracking_state(self, tracks: List[Dict], cv_image: np.ndarray, current_time: float, 
                             ok_gestures: List, stop_gestures: List, person_boxes: Dict):
        """更新跟踪状态"""
        current_track_ids = set()
        
        for track in tracks:
            track_id = track['track_id']
            current_track_ids.add(track_id)
            
            if track_id not in self.tracked_persons:
                self._initialize_new_track(track_id, current_time)
            else:
                self.tracked_persons[track_id]['last_seen_time'] = current_time

            if self.current_tracking_id == track_id:
                self.save_tracked_target(track_id, track['bbox'], cv_image, current_time)

        self._process_gesture_control(ok_gestures, stop_gestures, person_boxes, cv_image, current_time)
        self._handle_lost_targets(current_track_ids, tracks, cv_image, current_time)

    def _initialize_new_track(self, track_id: int, current_time: float):
        """初始化新跟踪目标"""
        self.tracked_persons[track_id] = {
            'is_tracking': False,
            'tracking_start_time': 0.0,
            'last_ok_time': 0.0,
            'first_seen_time': current_time,
            'last_seen_time': current_time
        }
        if track_id not in self.ok_gesture_count:
            self.ok_gesture_count[track_id] = 0
        if track_id not in self.stop_gesture_count:
            self.stop_gesture_count[track_id] = 0

    def _process_gesture_control(self, ok_gestures: List, stop_gestures: List, 
                               person_boxes: Dict, cv_image: np.ndarray, current_time: float):
        """处理手势控制逻辑"""
        for track_id in list(self.tracked_persons.keys()):
            target_has_ok = False
            target_has_stop = False
            
            for ok_gesture in ok_gestures:
                x1, y1, x2, y2, score = ok_gesture
                ok_box = (x1, y1, x2, y2)
                person_id, _, iou = self.find_person_for_gesture(ok_box, person_boxes)
                if person_id == track_id and iou >= self.roi_threshold:
                    target_has_ok = True
                    break
            
            for stop_gesture in stop_gestures:
                x1, y1, x2, y2, score = stop_gesture
                stop_box = (x1, y1, x2, y2)
                person_id, _, iou = self.find_person_for_gesture(stop_box, person_boxes)
                if person_id == track_id and iou >= self.roi_threshold:
                    target_has_stop = True
                    break
            
            if target_has_ok:
                self.ok_gesture_count[track_id] += 1
                self.stop_gesture_count[track_id] = 0
                
                if self.ok_gesture_count[track_id] >= self.ok_confirm_frames:
                    if track_id in person_boxes:
                        person_box = person_boxes[track_id]
                        self._handle_ok_gesture(track_id, person_box, cv_image, current_time)
            else:
                self.ok_gesture_count[track_id] = 0
            
            if target_has_stop:
                self.stop_gesture_count[track_id] += 1
                self.ok_gesture_count[track_id] = 0
                
                if self.stop_gesture_count[track_id] >= self.ok_confirm_frames:
                    self._handle_stop_gesture(track_id, current_time)
            else:
                self.stop_gesture_count[track_id] = 0

    def _handle_ok_gesture(self, person_id: int, person_box: Tuple, cv_image: np.ndarray, current_time: float):
        """处理ok手势确认"""
        if self.current_tracking_id is not None and self.current_tracking_id != person_id:
            self.get_logger().info(f"已在跟踪ID {self.current_tracking_id}，忽略ID {person_id}的OK手势")
            self.ok_gesture_count[person_id] = 0
            return
            
        person = self.tracked_persons[person_id]
        in_cooldown_period = (current_time - person['last_ok_time'] < self.tracking_protection_time)
        
        if not in_cooldown_period:
            self.current_tracking_id = person_id
            person['is_tracking'] = True
            person['tracking_start_time'] = current_time
            person['last_ok_time'] = current_time
            
            self.ok_gesture_count[person_id] = 0
            self.stop_gesture_count[person_id] = 0
            
            self.save_tracked_target(person_id, list(person_box), cv_image, current_time)
            self.target_lost_time = None
            self.get_logger().info(f"🎯 开始跟踪 ID: {person_id} (连续OK手势确认)")
        else:
            self.ok_gesture_count[person_id] = 0
            self.get_logger().info(f"ID {person_id} 在冷却期内，重置OK手势计数器")

    def _handle_stop_gesture(self, person_id: int, current_time: float):
        """处理stop手势确认"""
        if self.current_tracking_id != person_id:
            self.get_logger().info(f"当前未跟踪ID {person_id}，忽略STOP手势")
            self.stop_gesture_count[person_id] = 0
            return
            
        person = self.tracked_persons[person_id]
        in_protection_period = (current_time - person['tracking_start_time'] < self.tracking_protection_time)
        
        if not in_protection_period:
            person['is_tracking'] = False
            person['last_ok_time'] = current_time
            
            self.ok_gesture_count[person_id] = 0
            self.stop_gesture_count[person_id] = 0
            
            self.current_tracking_id = None
            self.target_lost_time = None
            self.get_logger().info(f"🛑 停止跟踪 ID: {person_id} (连续STOP手势确认)")
        else:
            self.stop_gesture_count[person_id] = 0
            self.get_logger().info(f"ID {person_id} 在保护期内，重置STOP手势计数器")

    def _handle_lost_targets(self, current_track_ids: set, tracks: List[Dict], 
                            cv_image: np.ndarray, current_time: float):
        """处理丢失目标"""
        if self.current_tracking_id is not None and self.current_tracking_id not in current_track_ids:
            if self.current_tracking_id in self.tracked_targets:
                self.tracked_targets[self.current_tracking_id].mark_lost()
                
                if self.current_tracking_id in self.tracked_persons:
                    self.tracked_persons[self.current_tracking_id]['is_tracking'] = False
                
                if self.target_lost_time is None:
                    self.target_lost_time = current_time
                    self.get_logger().warning(f"目标 {self.current_tracking_id} 丢失，立即启动ReID匹配找回")
                
                time_since_lost = current_time - self.target_lost_time
                
                if time_since_lost > self.lost_timeout_threshold:
                    self.get_logger().warning(
                        f"目标 {self.current_tracking_id} 丢失超过 {self.lost_timeout_threshold} 秒，停止跟踪"
                    )
                    self._clear_tracking_target()
                    return
                
                if self.with_reid:
                    recovered_id = self.try_recover_lost_target(tracks, cv_image, current_time)
                    if recovered_id is not None:
                        old_tracking_id = self.current_tracking_id
                        self.current_tracking_id = recovered_id
                        
                        if recovered_id in self.tracked_persons:
                            self.tracked_persons[recovered_id]['is_tracking'] = True
                            self.tracked_persons[recovered_id]['tracking_start_time'] = current_time
                            self.tracked_persons[recovered_id]['last_seen_time'] = current_time
                        
                        self.target_lost_time = None
                        
                        if recovered_id in [t['track_id'] for t in tracks]:
                            track = next(t for t in tracks if t['track_id'] == recovered_id)
                            self.save_tracked_target(recovered_id, track['bbox'], cv_image, current_time)
                        
                        self.get_logger().info(f"ReID找回成功，从ID {old_tracking_id} 切换到新ID: {recovered_id}")
                    else:
                        self.get_logger().warning(f"目标 {self.current_tracking_id} ReID找回失败，保持丢失状态")
        
        elif self.current_tracking_id is not None and self.current_tracking_id in current_track_ids:
            if self.target_lost_time is not None:
                track = next(t for t in tracks if t['track_id'] == self.current_tracking_id)
                verified_id = self._verify_target_with_reid(
                    self.tracked_targets[self.current_tracking_id], track, cv_image, current_time
                )
                
                if verified_id is not None:
                    self.target_lost_time = None
                    if self.current_tracking_id in self.tracked_targets:
                        self.tracked_targets[self.current_tracking_id].lost_frames = 0
                    self.get_logger().info(f"目标 {self.current_tracking_id}重新出现，ReID验证成功，继续跟踪")
                    
                    if verified_id in self.tracked_persons:
                        self.tracked_persons[verified_id]['is_tracking'] = True
                        self.tracked_persons[verified_id]['last_seen_time'] = current_time
                else:
                    if self.current_tracking_id in self.tracked_targets:
                        self.tracked_targets[self.current_tracking_id].mark_lost()
                    
                    time_since_lost = current_time - self.target_lost_time
                    if time_since_lost > self.lost_timeout_threshold:
                        self.get_logger().warning(
                            f"目标 {self.current_tracking_id} ReID验证失败超过 {self.lost_timeout_threshold} 秒，停止跟踪"
                        )
                        self._clear_tracking_target()
                        return

    def _clear_tracking_target(self):
        """清除当前跟踪目标的所有信息"""
        if self.current_tracking_id is not None:
            target_id = self.current_tracking_id          
            if target_id in self.tracked_persons:
                del self.tracked_persons[target_id]
            if target_id in self.ok_gesture_count:
                del self.ok_gesture_count[target_id]
            if target_id in self.stop_gesture_count:
                del self.stop_gesture_count[target_id]
            if target_id in self.tracked_targets:
                del self.tracked_targets[target_id]
      
        self.current_tracking_id = None
        self.target_lost_time = None
        
    def _cleanup_old_tracks(self, current_time: float, current_track_ids: set):
        """清理长时间未出现的跟踪目标"""
        max_track_age = 5.0
        
        for track_id in list(self.tracked_persons.keys()):
            if track_id == self.current_tracking_id:
                continue
                
            if track_id not in current_track_ids:
                last_seen = self.tracked_persons[track_id]['last_seen_time']
                if current_time - last_seen > max_track_age:
                    self._remove_track(track_id)

    def _remove_track(self, track_id: int):
        """移除跟踪目标"""
        if track_id in self.tracked_persons:
            del self.tracked_persons[track_id]
        if track_id in self.ok_gesture_count:
            del self.ok_gesture_count[track_id]
        if track_id in self.stop_gesture_count:
            del self.stop_gesture_count[track_id]
        if track_id in self.tracked_targets:
            target = self.tracked_targets[track_id]
            if target.is_switched and target.original_track_id not in self.tracked_targets:
                original_id = target.original_track_id
                self.tracked_targets[original_id] = copy.deepcopy(target)
                self.tracked_targets[original_id].track_id = original_id
                self.tracked_targets[original_id].is_switched = False
            
            del self.tracked_targets[track_id]
        
    def _publish_results(self, image: np.ndarray, tracks: List[Dict], header, ok_gestures, stop_gestures):
        """发布结果"""
        self.visualize_results(image, tracks, ok_gestures, stop_gestures)
        self.publish_tracked_keypoints(tracks, header)

        if self.detect_pose_pub.get_subscription_count() > 0:
            detect_pose_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            detect_pose_msg.header = header
            self.detect_pose_pub.publish(detect_pose_msg)

    def visualize_results(self, image: np.ndarray, tracks: List[Dict], ok_gestures, stop_gestures):
        """可视化跟踪结果"""
        display_image = image.copy()
        
        for ok_gesture in ok_gestures:
            x1, y1, x2, y2, score = ok_gesture
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"OK: {score:.2f}"
            cv2.putText(display_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for stop_gesture in stop_gestures:
            x1, y1, x2, y2, score = stop_gesture
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"STOP: {score:.2f}"
            cv2.putText(display_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']
            confidence = track['conf']

            is_tracking = (track_id == self.current_tracking_id and 
                        track_id in self.tracked_persons and 
                        self.tracked_persons[track_id].get('is_tracking', False))
            
            color = (255, 0, 0) if is_tracking else (0, 255, 0)
            thickness = 3 if is_tracking else 2

            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)
            
            label = f"ID:{track_id} {confidence:.2f}"
            if is_tracking:
                label = f"TRACKING {label}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(display_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        image[:] = display_image

    def publish_tracked_keypoints(self, tracks: List[Dict], header):
        """发布边界框坐标"""
        current_tracking_id = self.current_tracking_id
        
        tracking_target_in_frame = None
        if current_tracking_id is not None:
            for track in tracks:
                if track['track_id'] == current_tracking_id:
                    tracking_target_in_frame = track
                    break
        
        polygon_msg = PolygonStamped()
        polygon_msg.header = header
        polygon_msg.header.frame_id = "camera_link"
        
        if tracking_target_in_frame is not None:
            track_id = tracking_target_in_frame['track_id']
            x1, y1, x2, y2 = tracking_target_in_frame['bbox']
            
            # 修复：确保列表正确闭合
            points = [
                Point32(x=float(track_id), y=1.0, z=2.0),
                Point32(x=float(x1), y=float(y1), z=0.0),
                Point32(x=float(x2), y=float(y2), z=0.0)
            ]  # 这里必须正确闭合
            polygon_msg.polygon.points = points
            self.get_logger().info(f"📤 发布跟踪信息: ID {track_id}, bbox: ({x1},{y1})-({x2},{y2})")
            
        elif current_tracking_id is not None:
            points = [
                Point32(x=float(current_tracking_id), y=0.0, z=0.0),
                Point32(x=0.0, y=0.0, z=0.0),
                Point32(x=0.0, y=0.0, z=0.0)
            ]  # 这里必须正确闭合
            polygon_msg.polygon.points = points
            self.get_logger().info(f"📤 发布目标丢失状态: ID {current_tracking_id}")
            
        else:
            points = [
                Point32(x=0.0, y=0.0, z=0.0),
                Point32(x=0.0, y=0.0, z=0.0),
                Point32(x=0.0, y=0.0, z=0.0)
            ]  # 这里必须正确闭合
            polygon_msg.polygon.points = points
            self.get_logger().debug("无跟踪目标，发布空状态")
        
        self.keypoint_tracks_pub.publish(polygon_msg)
    
    def __del__(self):
        """清理资源"""
        try:
            # 释放CUDA上下文
            self.cuda_manager.release()
        except:
            pass


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = Yolov8HandTrackNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
            
    except Exception as e:
        print(f"Node initialization failed: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()