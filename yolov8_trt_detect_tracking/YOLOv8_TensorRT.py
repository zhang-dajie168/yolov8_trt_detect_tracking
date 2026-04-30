#!/usr/bin/env python3
"""
TensorRT YOLOv8 推理脚本 - 修正版（坐标已是像素值）
"""

import os
import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class YOLOv8TensorRT:
    def __init__(self, engine_path, conf_thres=0.3, nms_thres=0.5):
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.engine_path = engine_path
        
        # 创建独立的CUDA上下文
        self.cuda_ctx = None
        self._init_cuda_context()
        
        # 加载引擎
        print(f"Loading engine: {engine_path}")
        try:
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            
            # 获取输入输出信息
            self.input_names = []
            self.output_names = []
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.input_names.append(name)
                    print(f"Input: {name}, shape: {self.engine.get_tensor_shape(name)}")
                else:
                    self.output_names.append(name)
                    print(f"Output: {name}, shape: {self.engine.get_tensor_shape(name)}")
            
            # 分配内存
            self.inputs = []
            self.outputs = []
            self.stream = cuda.Stream()
            
            for name in self.input_names:
                shape = self.engine.get_tensor_shape(name)
                host = np.empty(shape, dtype=np.float32)
                device = cuda.mem_alloc(host.nbytes)
                self.inputs.append({'name': name, 'host': host, 'device': device})
                self.context.set_tensor_address(name, int(device))
            
            for name in self.output_names:
                shape = self.engine.get_tensor_shape(name)
                host = np.empty(shape, dtype=np.float32)
                device = cuda.mem_alloc(host.nbytes)
                self.outputs.append({'name': name, 'host': host, 'device': device})
                self.context.set_tensor_address(name, int(device))
            
            # 同步CUDA上下文
            cuda.Context.synchronize()
            
            self.class_names = {0: "person", 1: "ok", 2: "stop"}
            print("✅ YOLO TensorRT Engine loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load engine: {e}")
            raise
    
    def _init_cuda_context(self):
        """初始化独立的CUDA上下文"""
        try:
            # 获取当前设备
            device = cuda.Device(0)
            # 创建新的上下文
            self.cuda_ctx = device.retain_primary_context()
            self.cuda_ctx.push()
        except Exception as e:
            print(f"CUDA context init warning: {e}")
            self.cuda_ctx = None
    
    def _ensure_context(self):
        """确保正确的CUDA上下文"""
        if self.cuda_ctx:
            self.cuda_ctx.push()
    
    def preprocess(self, img):
        """预处理 - 保持宽高比"""
        self.orig_h, self.orig_w = img.shape[:2]
        
        # 计算缩放比例
        self.scale = min(640/self.orig_w, 640/self.orig_h)
        new_w = int(self.orig_w * self.scale)
        new_h = int(self.orig_h * self.scale)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h))
        
        # 填充
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        self.pad_w = (640 - new_w) // 2
        self.pad_h = (640 - new_h) // 2
        canvas[self.pad_h:self.pad_h+new_h, self.pad_w:self.pad_w+new_w] = resized
        
        # 归一化
        img_norm = canvas.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_norm - mean) / std
        
        # CHW格式
        img_norm = img_norm.transpose(2, 0, 1)
        img_norm = np.expand_dims(img_norm, axis=0)
        
        return np.ascontiguousarray(img_norm.astype(np.float32))
    
    def detect(self, img):
        """检测 - 添加错误恢复"""
        self._ensure_context()
        
        try:
            input_tensor = self.preprocess(img)
            
            # 拷贝到GPU
            cuda.memcpy_htod_async(self.inputs[0]['device'], input_tensor, self.stream)
            
            # 推理
            start = time.time()
            success = self.context.execute_async_v3(self.stream.handle)
            if not success:
                raise RuntimeError("TensorRT execution failed")
            
            # 拷贝回CPU
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            
            self.stream.synchronize()
            inference_time = (time.time() - start) * 1000
            
            # 后处理
            detections = self.postprocess(self.outputs[0]['host'])
            
            return detections, inference_time
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            # 重置上下文
            if self.cuda_ctx:
                self.cuda_ctx.pop()
                self.cuda_ctx.push()
            return [], 0
    
    def postprocess(self, output):
        """后处理 - 坐标已经是像素值"""
        detections = []
        
        # output shape: (1, 7, 8400)
        pred = output[0]  # (7, 8400)
        
        # 解析边界框 - 这些已经是像素坐标
        cx = pred[0, :]  # 中心x (像素)
        cy = pred[1, :]  # 中心y (像素)
        w = pred[2, :]   # 宽度 (像素)
        h = pred[3, :]   # 高度 (像素)
        
        # 解析类别分数
        person_scores = pred[4, :]
        ok_scores = pred[5, :]
        stop_scores = pred[6, :]
        
        # 获取每个检测的最高分和对应类别
        max_scores = np.maximum(np.maximum(person_scores, ok_scores), stop_scores)
        class_ids = np.zeros_like(max_scores, dtype=np.int32)
        class_ids[stop_scores > person_scores] = 2
        class_ids[ok_scores > person_scores] = 1
        
        # 过滤低置信度
        mask = max_scores > self.conf_thres
        if not np.any(mask):
            return detections
        
        cx = cx[mask]
        cy = cy[mask]
        w = w[mask]
        h = h[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]
        
        # 转换为边界框坐标 [x1, y1, x2, y2]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # 去除填充并缩放到原图尺寸
        x1 = (x1 - self.pad_w) / self.scale
        y1 = (y1 - self.pad_h) / self.scale
        x2 = (x2 - self.pad_w) / self.scale
        y2 = (y2 - self.pad_h) / self.scale
        
        # 裁剪到图像范围
        x1 = np.clip(x1, 0, self.orig_w)
        y1 = np.clip(y1, 0, self.orig_h)
        x2 = np.clip(x2, 0, self.orig_w)
        y2 = np.clip(y2, 0, self.orig_h)
        
        # 过滤无效框
        valid = (x2 > x1) & (y2 > y1)
        if not np.any(valid):
            return detections
        
        boxes = np.stack([x1[valid], y1[valid], x2[valid], y2[valid]], axis=1)
        scores_nms = max_scores[valid]
        class_ids_nms = class_ids[valid]
        
        # NMS
        keep = self.nms(boxes, scores_nms)
        
        for idx in keep:
            detections.append((
                int(class_ids_nms[idx]),
                float(scores_nms[idx]),
                int(boxes[idx][0]),
                int(boxes[idx][1]),
                int(boxes[idx][2]),
                int(boxes[idx][3])
            ))
        
        return detections
    
    def nms(self, boxes, scores):
        """非极大值抑制"""
        if len(boxes) == 0:
            return []
        
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            
            if len(indices) == 1:
                break
            
            # 计算IOU
            iou_list = []
            for j in indices[1:]:
                inter_x1 = max(boxes[i][0], boxes[j][0])
                inter_y1 = max(boxes[i][1], boxes[j][1])
                inter_x2 = min(boxes[i][2], boxes[j][2])
                inter_y2 = min(boxes[i][3], boxes[j][3])
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                    area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                    iou_val = inter_area / (area_i + area_j - inter_area)
                else:
                    iou_val = 0
                iou_list.append(iou_val)
            
            indices = indices[1:][np.array(iou_list) <= self.nms_thres]
        
        return keep
    
    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, 'stream'):
                self.stream.synchronize()
            for inp in self.inputs:
                if 'device' in inp:
                    inp['device'].free()
            for out in self.outputs:
                if 'device' in out:
                    out['device'].free()
            if self.cuda_ctx:
                self.cuda_ctx.pop()
                del self.cuda_ctx
        except:
            pass


def draw_detections(image, detections, class_names):
    """绘制检测结果"""
    colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}
    
    for cls, score, x1, y1, x2, y2 in detections:
        color = colors.get(cls, (255, 255, 255))
        class_name = class_names.get(cls, "unknown")
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景
        label = f"{class_name}: {score:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        
        # 绘制标签文字
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return image


def main():
    # 路径配置
    engine_path = "/home/peng/Ebike_Human_Follower/src/yolov8_pytorch_detect_tracking/models/yolov8.engine"
    test_image_dir = "/home/peng/Ebike_Human_Follower/src/yolov8_pytorch_detect_tracking/test_img"
    output_dir = "/home/peng/Ebike_Human_Follower/src/yolov8_pytorch_detect_tracking/test_img/output_final"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查模型
    if not os.path.exists(engine_path):
        print(f"❌ Engine not found: {engine_path}")
        return
    
    # 初始化模型
    print("="*60)
    print("Initializing YOLOv8 TensorRT Model")
    print("="*60)
    model = YOLOv8TensorRT(engine_path, conf_thres=0.25, nms_thres=0.45)
    print("="*60)
    
    # 获取测试图片
    test_images = [f for f in os.listdir(test_image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not test_images:
        print(f"❌ No test images found")
        return
    
    print(f"\nFound {len(test_images)} test images\n")
    
    total_time = 0
    total_detections = 0
    
    # 处理每张图片
    for img_name in test_images:
        print(f"📷 Processing: {img_name}")
        img_path = os.path.join(test_image_dir, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"  Failed to load image")
            continue
        
        original = image.copy()
        
        try:
            # 检测
            detections, inference_time = model.detect(image)
            
            total_time += inference_time
            total_detections += len(detections)
            
            print(f"  Inference time: {inference_time:.2f} ms ({1000/inference_time:.1f} FPS)")
            print(f"  Detections: {len(detections)}")
            
            for i, (cls, score, x1, y1, x2, y2) in enumerate(detections):
                print(f"    {i+1}. {model.class_names[cls]}: {score:.3f} - ({x1},{y1},{x2},{y2})")
            
            # 绘制结果
            result = draw_detections(original, detections, model.class_names)
            
            # 添加性能信息
            cv2.putText(result, f"TensorRT: {1000/inference_time:.1f}FPS", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 保存结果
            output_path = os.path.join(output_dir, f"result_{img_name}")
            cv2.imwrite(output_path, result)
            print(f"  💾 Saved: {output_path}\n")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    if len(test_images) > 0:
        print("="*60)
        print("Testing Summary")
        print("="*60)
        print(f"Total images: {len(test_images)}")
        print(f"Total detections: {total_detections}")
        avg_time = total_time / len(test_images)
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"Average FPS: {1000/avg_time:.1f}")
        print(f"\n✅ Results saved to: {output_dir}")
        print("="*60)


if __name__ == '__main__':
    main()