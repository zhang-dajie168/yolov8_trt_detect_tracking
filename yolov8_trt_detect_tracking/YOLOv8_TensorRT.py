#!/usr/bin/env python3
"""
TensorRT YOLOv8 推理脚本 - 修复版
"""

import os
import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
# 注意：不再导入 pycuda.autoinit


class SharedCUDAManager:
    """共享CUDA上下文管理器 - 单例，统一管理主上下文"""
    _instance = None
    _context = None
    _ref_count = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def acquire(self):
        """获取CUDA上下文（push）"""
        if self._context is None:
            device = cuda.Device(0)
            self._context = device.retain_primary_context()
            self._context.push()
        self._ref_count += 1
        return self._context

    def release(self):
        """释放CUDA上下文（pop） - 改进引用计数，防止负数"""
        if self._ref_count > 0:
            self._ref_count -= 1
            if self._ref_count == 0 and self._context:
                self._context.pop()
                self._context = None


# 全局共享实例
_shared_cuda_manager = SharedCUDAManager()


class YOLOv8TensorRT:
    def __init__(self, engine_path, conf_thres=0.3, nms_thres=0.45, shared_context=True):
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.engine_path = engine_path
        self.shared_context = shared_context
        self.input_size = 640  # 固定输入尺寸

        # 使用共享CUDA上下文
        if shared_context:
            self.cuda_ctx = _shared_cuda_manager.acquire()
        else:
            self.cuda_ctx = None

        # 加载引擎
        self._load_engine()

    def _load_engine(self):
        """加载TensorRT引擎"""
        print(f"Loading engine: {self.engine_path}")

        with open(self.engine_path, 'rb') as f:
            # 降低日志级别，避免无关警告
            runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # 获取输入输出信息
        self.input_name = None
        self.output_name = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
                shape = self.engine.get_tensor_shape(name)
                print(f"Input: {name}, shape: {shape}")
            else:
                self.output_name = name
                shape = self.engine.get_tensor_shape(name)
                print(f"Output: {name}, shape: {shape}")

        # 分配内存
        input_shape = self.engine.get_tensor_shape(self.input_name)
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        self.input_host = np.empty(input_shape, dtype=np.float32)
        self.input_device = cuda.mem_alloc(self.input_host.nbytes)

        output_shape = self.engine.get_tensor_shape(self.output_name)
        self.output_host = np.empty(output_shape, dtype=np.float32)
        self.output_device = cuda.mem_alloc(self.output_host.nbytes)

        # 设置张量地址
        self.context.set_tensor_address(self.input_name, int(self.input_device))
        self.context.set_tensor_address(self.output_name, int(self.output_device))

        self.stream = cuda.Stream()

        print("✅ TensorRT engine loaded")

    def preprocess(self, img):
        """预处理图像"""
        self.orig_h, self.orig_w = img.shape[:2]

        # 计算缩放比例（保持宽高比）
        scale_w = self.input_width / self.orig_w
        scale_h = self.input_height / self.orig_h
        self.scale = min(scale_w, scale_h)

        new_w = int(self.orig_w * self.scale)
        new_h = int(self.orig_h * self.scale)

        # Resize
        resized = cv2.resize(img, (new_w, new_h))

        # Letterbox 填充
        self.pad_w = (self.input_width - new_w) // 2
        self.pad_h = (self.input_height - new_h) // 2

        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        canvas[self.pad_h:self.pad_h + new_h, self.pad_w:self.pad_w + new_w] = resized

        # 归一化 (YOLOv8 标准)
        img_norm = canvas.astype(np.float32) / 255.0

        # BGR to RGB, HWC to CHW
        img_norm = img_norm[:, :, ::-1].transpose(2, 0, 1)

        # 添加 batch 维度
        img_norm = np.expand_dims(img_norm, axis=0)

        return np.ascontiguousarray(img_norm)

    def detect(self, img):
        """执行检测"""
        try:
            # 预处理
            input_tensor = self.preprocess(img)

            # 拷贝到GPU
            cuda.memcpy_htod_async(self.input_device, input_tensor, self.stream)

            # 推理
            start = time.time()
            self.context.execute_async_v3(self.stream.handle)

            # 拷贝结果回CPU
            cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
            self.stream.synchronize()

            inference_time = (time.time() - start) * 1000

            # 后处理
            detections = self.postprocess(self.output_host)

            return detections, inference_time

        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            return [], 0

    def postprocess(self, output):
        """后处理"""
        # output shape: (1, 84, 8400) 对于 YOLOv8
        # 或 (1, 7, 8400) 对于自定义模型

        pred = output[0]  # (84, 8400) 或 (7, 8400)

        # 判断输出格式
        if pred.shape[0] == 84:  # 标准 YOLOv8 (80类)
            # 提取边界框
            cx = pred[0, :]
            cy = pred[1, :]
            w = pred[2, :]
            h = pred[3, :]

            # 类别分数 (4:84)
            class_scores = pred[4:, :]
            scores = np.max(class_scores, axis=0)
            class_ids = np.argmax(class_scores, axis=0)

        else:  # 自定义模型 (person, ok, stop)
            cx = pred[0, :]
            cy = pred[1, :]
            w = pred[2, :]
            h = pred[3, :]

            # 假设最后3个是 person, ok, stop
            person_scores = pred[4, :]
            ok_scores = pred[5, :] if pred.shape[0] > 5 else np.zeros_like(person_scores)
            stop_scores = pred[6, :] if pred.shape[0] > 6 else np.zeros_like(person_scores)

            # 取最高分
            scores = np.maximum(person_scores, ok_scores)
            scores = np.maximum(scores, stop_scores)

            class_ids = np.zeros_like(scores, dtype=np.int32)
            class_ids[stop_scores > person_scores] = 2
            class_ids[ok_scores > person_scores] = 1

        # 过滤低置信度
        mask = scores > self.conf_thres
        if not np.any(mask):
            return []

        cx = cx[mask]
        cy = cy[mask]
        w = w[mask]
        h = h[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # 转换到原图坐标
        x1 = (cx - w / 2 - self.pad_w) / self.scale
        y1 = (cy - h / 2 - self.pad_h) / self.scale
        x2 = (cx + w / 2 - self.pad_w) / self.scale
        y2 = (cy + h / 2 - self.pad_h) / self.scale

        # 裁剪到图像范围
        x1 = np.clip(x1, 0, self.orig_w)
        y1 = np.clip(y1, 0, self.orig_h)
        x2 = np.clip(x2, 0, self.orig_w)
        y2 = np.clip(y2, 0, self.orig_h)

        # 过滤无效框
        valid = (x2 > x1) & (y2 > y1)
        if not np.any(valid):
            return []

        # NMS (简单版本)
        boxes = np.stack([x1[valid], y1[valid], x2[valid], y2[valid]], axis=1)
        scores_nms = scores[valid]
        class_ids_nms = class_ids[valid]

        keep = self.nms(boxes, scores_nms)

        detections = []
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

        # 按分数排序
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
                    iou = inter_area / (area_i + area_j - inter_area + 1e-8)
                else:
                    iou = 0
                iou_list.append(iou)

            indices = indices[1:][np.array(iou_list) <= self.nms_thres]

        return keep

    def __del__(self):
        """清理资源 - 仅释放设备内存，不操作共享上下文（由管理器统一释放）"""
        try:
            if hasattr(self, 'stream'):
                self.stream.synchronize()
            if hasattr(self, 'input_device'):
                self.input_device.free()
            if hasattr(self, 'output_device'):
                self.output_device.free()
            # 如果使用非共享上下文（极少情况），才在这里 pop
            if not self.shared_context and hasattr(self, 'cuda_ctx') and self.cuda_ctx:
                self.cuda_ctx.pop()
        except:
            pass
