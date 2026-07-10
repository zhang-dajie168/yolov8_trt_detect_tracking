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
        
        # 自动锁定参数
        self.auto_lock_cooldown = self.auto_lock_cooldown  # 自动锁定冷却时间（秒），防止频繁切换
        self.last_lock_time = 0.0
        
        # 重新锁定阈值参数
        self.relock_score_threshold = self.relock_score_threshold  # 重新锁定所需的最低综合得分
        self.was_locked_before = False  # 记录之前是否有锁定目标（用于判断是否需要高阈值）
        self.previous_tracking_id = None  # 记录之前的跟踪ID

        # 初始化模型和组件
        self._initialize_components()
        
        # 初始化变量
        self._initialize_variables()
        
        self.get_logger().info("YOLOv8 Hand Track Node initialized with TensorRT (Auto-lock Mode)")
        self.get_logger().info(f"重新锁定阈值: {self.relock_score_threshold}")
        self.print_parameters()

    def _declare_parameters(self):
        """声明所有参数"""
        self.declare_parameter('model_path', '/home/wheeltec/Ebike_Human_Follower/src/yolov8_trt_detect_tracking/models/yolov8.engine')
        self.declare_parameter('reid_engine_path', '/home/wheeltec/Ebike_Human_Follower/src/yolov8_trt_detect_tracking/models/osnet_x0_25.engine')
        self.declare_parameter('conf_threshold', 0.3)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('max_processing_fps', 15)
        self.declare_parameter('tracking_protection_time', 5.0)
        # 优化后的参数
        self.declare_parameter('reid_similarity_threshold', 0.65)  # 降低阈值，适应小目标
        self.declare_parameter('height_change_threshold', 0.20)   # 放宽高度变化容忍度
        self.declare_parameter('lost_timeout_threshold', 1.0)     # 增加丢失超时时间
        self.declare_parameter('roi_threshold', 0.5)
        self.declare_parameter('use_cuda', True)
        self.declare_parameter('input_size', 640)
        self.declare_parameter('with_reid', True)  # 是否启用ReID
        # 新增自动锁定参数
        self.declare_parameter('auto_lock_enabled', True)  # 是否启用自动锁定
        self.declare_parameter('auto_lock_cooldown', 5.0)  # 自动锁定冷却时间，增加防止频繁切换
        self.declare_parameter('relock_score_threshold', 0.85)  # 降低重新锁定阈值
        
    def _get_parameters(self):
        """获取所有参数值"""
        self.model_path = self.get_parameter('model_path').value
        self.reid_engine_path = self.get_parameter('reid_engine_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.nms_threshold = self.get_parameter('nms_threshold').value
        self.max_processing_fps = self.get_parameter('max_processing_fps').value
        self.tracking_protection_time = self.get_parameter('tracking_protection_time').value
        self.reid_similarity_threshold = self.get_parameter('reid_similarity_threshold').value
        self.height_change_threshold = self.get_parameter('height_change_threshold').value
        self.lost_timeout_threshold = self.get_parameter('lost_timeout_threshold').value
        self.roi_threshold = self.get_parameter('roi_threshold').value
        self.use_cuda = self.get_parameter('use_cuda').value
        self.input_size = self.get_parameter('input_size').value
        self.with_reid = self.get_parameter('with_reid').value
        # 新增参数
        self.auto_lock_enabled = self.get_parameter('auto_lock_enabled').value
        self.auto_lock_cooldown = self.get_parameter('auto_lock_cooldown').value
        self.relock_score_threshold = self.get_parameter('relock_score_threshold').value

    def print_parameters(self):
        """打印参数信息"""
        self.get_logger().info("===== 参数配置信息 (TensorRT - Auto Lock Mode) =====")
        self.get_logger().info(f"YOLO TensorRT引擎: {self.model_path}")
        self.get_logger().info(f"ReID TensorRT引擎: {self.reid_engine_path}")
        self.get_logger().info(f"置信度阈值: {self.conf_threshold}")
        self.get_logger().info(f"NMS阈值: {self.nms_threshold}")
        self.get_logger().info(f"最大处理帧率: {self.max_processing_fps}FPS")
        self.get_logger().info(f"跟踪保护时间: {self.tracking_protection_time}s")
        self.get_logger().info(f"ReID相似度阈值: {self.reid_similarity_threshold}")
        self.get_logger().info(f"高度变化阈值: {self.height_change_threshold}")
        self.get_logger().info(f"丢失超时阈值: {self.lost_timeout_threshold}s")
        self.get_logger().info(f"YOLO输入尺寸: {self.input_size}")
        self.get_logger().info(f"启用ReID: {self.with_reid}")
        self.get_logger().info(f"自动锁定模式: {'启用' if self.auto_lock_enabled else '禁用'}")
        self.get_logger().info(f"自动锁定冷却时间: {self.auto_lock_cooldown}s")
        self.get_logger().info(f"重新锁定阈值: {self.relock_score_threshold}")
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
        self.current_tracking_id = None
        self.tracked_targets: Dict[int, TrackedTarget] = {}
        self.target_lost_time: Optional[float] = None
        self.class_names = {0: "person"}

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
        area2 = (x2_1 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        iou = inter_area / union_area if union_area > 0 else 0
        return iou

    def calculate_target_score(self, track: Dict, image_width: int, image_height: int, max_area: float) -> float:
        """
        计算单个目标的综合得分
        针对中心区域目标优化：面积权重更高，使用指数衰减距离评分
        图像尺寸: 1280x1080，目标尺寸约130x400
        """
        x1, y1, x2, y2 = track['bbox']
        
        # 计算中心点坐标
        image_center_x = image_width / 2
        image_center_y = image_height / 2
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 计算到图像中心的距离归一化（使用指数衰减，让中心附近的目标得分更高）
        max_distance = np.sqrt(image_center_x**2 + image_center_y**2)
        distance = np.sqrt((center_x - image_center_x)**2 + (center_y - image_center_y)**2)
        distance_normalized = np.exp(-distance / (max_distance * 0.3))  # 指数衰减，中心区域得分更高
        
        # 计算面积归一化
        area = (x2 - x1) * (y2 - y1)
        area_normalized = area / max_area if max_area > 0 else 1.0
        
        # 针对目标尺寸约130x400 (面积约52000) 优化权重
        # 图像尺寸1280x1080，目标占图像比例适中
        # 面积权重0.7，距离权重0.3（更看重面积）
        area_weight = 0.7
        distance_weight = 0.3
        score = (area_normalized * area_weight) + (distance_normalized * distance_weight)
        
        # 如果目标在中心区域（距离中心<200像素），额外加分
        if distance < 200:
            score = min(1.0, score * 1.1)
        
        return score

    def select_best_target(self, tracks: List[Dict], image_width: int, image_height: int, 
                           require_high_score: bool = False) -> Optional[Tuple[int, float]]:
        """
        自动选择最佳跟踪目标
        针对中心区域目标优化，过滤边界附近的目标
        """
        if not tracks:
            return None
        
        # 过滤掉边界附近的目标（减少误选）
        margin = 50  # 边界余量
        filtered_tracks = []
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            # 如果目标过于靠近边界，降低优先级
            if x1 < margin or x2 > image_width - margin or y1 < margin or y2 > image_height - margin:
                # 但目标仍然保留，只是不优先选择
                filtered_tracks.append(track)
            else:
                filtered_tracks.append(track)
        
        # 计算面积范围用于归一化
        areas = []
        for track in filtered_tracks:
            x1, y1, x2, y2 = track['bbox']
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)
        
        max_area = max(areas) if areas else 1
        
        best_track_id = None
        best_score = -1
        
        for track in filtered_tracks:
            track_id = track['track_id']
            score = self.calculate_target_score(track, image_width, image_height, max_area)
            
            self.get_logger().debug(
                f"目标 ID:{track_id} - 综合得分: {score:.3f}"
            )
            
            if score > best_score:
                best_score = score
                best_track_id = track_id
        
        # 如果需要高分阈值，检查是否达到要求
        if require_high_score and best_score < self.relock_score_threshold:
            self.get_logger().info(
                f"重新锁定条件未满足: 最佳得分 {best_score:.3f} < 阈值 {self.relock_score_threshold:.3f}"
            )
            return None
        
        if best_track_id is not None:
            self.get_logger().info(f"🎯 最佳目标 ID: {best_track_id}, 综合得分: {best_score:.3f}")
        
        return (best_track_id, best_score)

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
        """立即尝试找回丢失的跟踪目标（优化版）"""
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
        best_height_match = float('inf')
        
        for track in candidate_tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            
            x1, y1, x2, y2 = bbox
            candidate_height_pixels = y2 - y1
            
            height_ratio = candidate_height_pixels / target_height_pixels
            height_change = abs(1.0 - height_ratio)
            
            # 放宽高度变化阈值
            if height_change > self.height_change_threshold * 1.5:  # 放宽50%
                self.get_logger().debug(f"候选 ID:{track_id} 高度变化 {height_change:.3f} 过大,跳过")
                continue
            
            candidate_feature = self.extract_feature_from_bbox(image, bbox)
            
            if candidate_feature is not None and np.any(candidate_feature):
                similarity = np.dot(target.feature, candidate_feature) / (
                    np.linalg.norm(target.feature) * np.linalg.norm(candidate_feature) + 1e-8
                )
                
                self.get_logger().info(f"候选目标 ID:{track_id} ReID相似度: {similarity:.3f}, 高度变化: {height_change:.3f}")
                
                # 综合考虑相似度和高度匹配度
                combined_score = similarity * 0.7 + (1.0 - height_change) * 0.3
                
                if similarity >= self.reid_similarity_threshold and combined_score > best_similarity:
                    best_similarity = combined_score
                    best_match_id = track_id
                    best_height_match = height_change
        
        if best_match_id is not None:
            self.get_logger().info(
                f"目标 {self.current_tracking_id} ReID找回成功! 匹配ID: {best_match_id}, "
                f"综合得分: {best_similarity:.3f}, 高度变化: {best_height_match:.3f}"
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
            self.was_locked_before = True
            self.previous_tracking_id = self.current_tracking_id
            
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

            # 只检测person类别
            person_detections = []
            
            for class_id, score, x1, y1, x2, y2 in results:
                if class_id == 0:  # person
                    person_detections.append([x1, y1, x2-x1, y2-y1, score, 0])
            
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

            # 更新跟踪状态（自动锁定模式）
            self._update_tracking_state_auto(tracks, cv_image, current_time)
            
            # 清理长时间未出现的跟踪目标
            self._cleanup_old_tracks(current_time, set(track['track_id'] for track in tracks))

            # 可视化并发布结果
            self._publish_results(cv_image, tracks, msg.header)
                   
        except Exception as e:
            self.get_logger().error(f"Image processing error: {str(e)}")
            import traceback
            traceback.print_exc()

    def _update_tracking_state_auto(self, tracks: List[Dict], cv_image: np.ndarray, current_time: float):
        """自动锁定模式：更新跟踪状态（优化版）"""
        current_track_ids = set(track['track_id'] for track in tracks)
        h, w = cv_image.shape[:2]
        
        # 更新所有出现的目标的最近出现时间
        for track in tracks:
            track_id = track['track_id']
            if track_id not in self.tracked_persons:
                self._initialize_new_track(track_id, current_time)
            else:
                self.tracked_persons[track_id]['last_seen_time'] = current_time
        
        # 如果没有当前跟踪目标，尝试自动锁定一个
        if self.current_tracking_id is None:
            if self.auto_lock_enabled and tracks:
                # 判断是否需要高分阈值（之前有过锁定目标且已解锁）
                require_high_score = self.was_locked_before
                
                result = self.select_best_target(tracks, w, h, require_high_score)
                if result is not None:
                    best_target_id, best_score = result
                    self._lock_target(best_target_id, tracks, cv_image, current_time, best_score)
            return
        
        # 检查当前跟踪目标是否还在视野中
        if self.current_tracking_id not in current_track_ids:
            # 目标丢失，尝试找回或清理
            self._handle_lost_target_auto(tracks, cv_image, current_time)
        else:
            # 目标还在视野中，更新信息
            self.target_lost_time = None
            track = next(t for t in tracks if t['track_id'] == self.current_tracking_id)
            
            # 检查当前目标是否仍然有效（面积变化检查）
            x1, y1, x2, y2 = track['bbox']
            current_area = (x2 - x1) * (y2 - y1)
            if self.current_tracking_id in self.tracked_targets:
                target = self.tracked_targets[self.current_tracking_id]
                target_bbox = target.bbox
                tx1, ty1, tx2, ty2 = target_bbox
                target_area = (tx2 - tx1) * (ty2 - ty1)
                area_ratio = current_area / target_area if target_area > 0 else 1.0
                # 如果面积变化超过50%，可能目标切换了，需要重新评估
                if area_ratio < 0.5 or area_ratio > 2.0:
                    self.get_logger().warning(
                        f"目标 {self.current_tracking_id} 面积变化过大 ({area_ratio:.2f})，可能切换目标"
                    )
                    # 不立即解锁，但增加验证
                    if (current_time - self.last_lock_time) > self.auto_lock_cooldown:
                        self._unlock_target()
                        return
            
            self.save_tracked_target(self.current_tracking_id, track['bbox'], cv_image, current_time)
            
            # 在冷却期内不切换目标，保持稳定
            if (current_time - self.last_lock_time) > self.auto_lock_cooldown:
                # 只有在目标评分明显下降时才考虑切换
                current_track = next(t for t in tracks if t['track_id'] == self.current_tracking_id)
                current_score = self.calculate_target_score(current_track, w, h, 1.0)
                
                # 寻找最佳目标
                result = self.select_best_target(tracks, w, h, require_high_score=False)
                if result is not None:
                    best_target_id, best_score = result
                    # 只有当新目标得分显著高于当前目标（超过0.15）时才切换
                    if best_target_id != self.current_tracking_id and best_score > current_score + 0.15:
                        self.get_logger().info(
                            f"🔄 自动切换目标: ID {self.current_tracking_id} (得分:{current_score:.3f}) "
                            f"-> ID {best_target_id} (得分:{best_score:.3f})"
                        )
                        self._unlock_target()
                        self._lock_target(best_target_id, tracks, cv_image, current_time, best_score)

    def _lock_target(self, track_id: int, tracks: List[Dict], cv_image: np.ndarray, 
                     current_time: float, score: float = None):
        """锁定目标"""
        if track_id not in self.tracked_persons:
            self._initialize_new_track(track_id, current_time)
        
        self.current_tracking_id = track_id
        self.tracked_persons[track_id]['is_tracking'] = True
        self.tracked_persons[track_id]['tracking_start_time'] = current_time
        self.target_lost_time = None
        self.last_lock_time = current_time
        self.was_locked_before = True  # 标记曾经有过锁定目标
        self.previous_tracking_id = track_id
        
        # 保存目标特征
        track = next(t for t in tracks if t['track_id'] == track_id)
        self.save_tracked_target(track_id, track['bbox'], cv_image, current_time)
        
        score_str = f", 得分: {score:.3f}" if score is not None else ""
        self.get_logger().info(f"🎯 锁定目标 ID: {track_id}{score_str}")

    def _unlock_target(self):
        """解锁当前目标"""
        if self.current_tracking_id is not None:
            target_id = self.current_tracking_id
            if target_id in self.tracked_persons:
                self.tracked_persons[target_id]['is_tracking'] = False
            self.get_logger().info(f"🔓 解锁目标 ID: {target_id}")
            
            # 注意：这里不清除 was_locked_before，保持记录
            # 这样下次重新锁定时需要高分阈值
        
        self.current_tracking_id = None
        self.target_lost_time = None

    def _reset_lock_state(self):
        """完全重置锁定状态（清除历史记录）"""
        self.was_locked_before = False
        self.previous_tracking_id = None
        self.get_logger().info("🔄 重置锁定历史状态")

    def _initialize_new_track(self, track_id: int, current_time: float):
        """初始化新跟踪目标"""
        self.tracked_persons[track_id] = {
            'is_tracking': False,
            'tracking_start_time': 0.0,
            'first_seen_time': current_time,
            'last_seen_time': current_time
        }

    def _handle_lost_target_auto(self, tracks: List[Dict], cv_image: np.ndarray, current_time: float):
        """处理丢失目标（自动锁定模式）"""
        if self.current_tracking_id is None:
            return
        
        if self.current_tracking_id in self.tracked_targets:
            self.tracked_targets[self.current_tracking_id].mark_lost()
        
        if self.current_tracking_id in self.tracked_persons:
            self.tracked_persons[self.current_tracking_id]['is_tracking'] = False
        
        if self.target_lost_time is None:
            self.target_lost_time = current_time
            self.get_logger().warning(f"目标 {self.current_tracking_id} 丢失，启动ReID匹配找回")
        
        time_since_lost = current_time - self.target_lost_time
        
        if time_since_lost > self.lost_timeout_threshold:
            self.get_logger().warning(
                f"目标 {self.current_tracking_id} 丢失超过 {self.lost_timeout_threshold} 秒，停止跟踪"
            )
            # 注意：这里完全重置状态，下次锁定需要高分阈值
            self._unlock_target()
            # 重置历史状态，这样下次重新锁定时需要高分阈值
            self._reset_lock_state()
            return
        
        # 尝试ReID找回
        if self.with_reid and tracks:
            recovered_id = self.try_recover_lost_target(tracks, cv_image, current_time)
            if recovered_id is not None:
                old_tracking_id = self.current_tracking_id
                self.current_tracking_id = recovered_id
                
                if recovered_id in self.tracked_persons:
                    self.tracked_persons[recovered_id]['is_tracking'] = True
                    self.tracked_persons[recovered_id]['tracking_start_time'] = current_time
                    self.tracked_persons[recovered_id]['last_seen_time'] = current_time
                
                self.target_lost_time = None
                self.last_lock_time = current_time
                self.was_locked_before = True
                
                if recovered_id in [t['track_id'] for t in tracks]:
                    track = next(t for t in tracks if t['track_id'] == recovered_id)
                    self.save_tracked_target(recovered_id, track['bbox'], cv_image, current_time)
                
                self.get_logger().info(f"ReID找回成功，从ID {old_tracking_id} 切换到新ID: {recovered_id}")
            else:
                self.get_logger().warning(f"目标 {self.current_tracking_id} ReID找回失败，保持丢失状态")

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
        if track_id in self.tracked_targets:
            target = self.tracked_targets[track_id]
            if target.is_switched and target.original_track_id not in self.tracked_targets:
                original_id = target.original_track_id
                self.tracked_targets[original_id] = copy.deepcopy(target)
                self.tracked_targets[original_id].track_id = original_id
                self.tracked_targets[original_id].is_switched = False
            
            del self.tracked_targets[track_id]
        
    def _publish_results(self, image: np.ndarray, tracks: List[Dict], header):
        """发布结果"""
        self.visualize_results(image, tracks)
        self.publish_tracked_keypoints(tracks, header)

        if self.detect_pose_pub.get_subscription_count() > 0:
            detect_pose_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            detect_pose_msg.header = header
            self.detect_pose_pub.publish(detect_pose_msg)

    def visualize_results(self, image: np.ndarray, tracks: List[Dict]):
        """可视化跟踪结果"""
        display_image = image.copy()
        
        # 在图像顶部显示重新锁定阈值信息（当无锁定时）
        if self.current_tracking_id is None and self.was_locked_before:
            h, w = display_image.shape[:2]
            relock_text = f"Need score >= {self.relock_score_threshold} to relock"
            cv2.putText(display_image, relock_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']
            confidence = track['conf']

            is_tracking = (track_id == self.current_tracking_id and 
                        track_id in self.tracked_persons and 
                        self.tracked_persons[track_id].get('is_tracking', False))
            
            color = (0, 0, 255) if is_tracking else (0, 255, 0)  # 锁定时红色
            thickness = 3 if is_tracking else 2

            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)
            
            label = f"ID:{track_id} {confidence:.2f}"
            if is_tracking:
                label = f"LOCKED {label}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(display_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 在图像中心绘制十字标记
        h, w = display_image.shape[:2]
        center_x, center_y = w // 2, h // 2
        cv2.drawMarker(display_image, (center_x, center_y), (255, 255, 0), 
                       cv2.MARKER_CROSS, 30, 2)

        image[:] = display_image

    def publish_tracked_keypoints(self, tracks: List[Dict], header):
        """
        发布边界框坐标 - 严格区分锁定状态
        只有当前目标处于锁定状态（红框）时才发布有效的bbox信息
        """
        current_tracking_id = self.current_tracking_id
        
        polygon_msg = PolygonStamped()
        polygon_msg.header = header
        polygon_msg.header.frame_id = "camera_link"
        
        # 检查是否有锁定目标
        is_target_locked = False
        target_track = None
        
        if current_tracking_id is not None:
            # 检查目标是否在 tracked_persons 中被标记为 is_tracking
            if (current_tracking_id in self.tracked_persons and 
                self.tracked_persons[current_tracking_id].get('is_tracking', False)):
                is_target_locked = True
                
                # 查找对应的track数据
                for track in tracks:
                    if track['track_id'] == current_tracking_id:
                        target_track = track
                        break
        
        if is_target_locked and target_track is not None:
            # 只有锁定状态（红框）才发布有效bbox信息
            track_id = target_track['track_id']
            x1, y1, x2, y2 = target_track['bbox']
            
            # 构建消息点：状态信息 + 边界框
            points = [
                Point32(x=float(track_id), y=1.0, z=2.0),  # 状态点: y=1.0表示锁定跟踪
                Point32(x=float(x1), y=float(y1), z=0.0),   # 边界框左上
                Point32(x=float(x2), y=float(y2), z=0.0),   # 边界框右下
            ]
            
            polygon_msg.polygon.points = points
            self.get_logger().info(f"📤 发布锁定跟踪信息: ID {track_id}, bbox: ({x1},{y1})-({x2},{y2})")
            
        else:
            # 非锁定状态（绿框或无目标）发布空消息
            points = [
                Point32(x=0.0, y=0.0, z=0.0),
                Point32(x=0.0, y=0.0, z=0.0),
                Point32(x=0.0, y=0.0, z=0.0),
            ]
            polygon_msg.polygon.points = points
            
            if current_tracking_id is not None:
                self.get_logger().debug(f"目标 ID {current_tracking_id} 未锁定，发布空状态")
            else:
                self.get_logger().debug("无跟踪目标，发布空状态")
        
        # 发布消息
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