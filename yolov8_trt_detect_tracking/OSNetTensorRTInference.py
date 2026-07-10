#!/usr/bin/env python3
"""
OSNet TensorRT推理类 - 线程局部上下文优化版
"""

import os
import cv2
import numpy as np
import time
import threading
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class ThreadLocalCUDAManager:
    """线程局部CUDA上下文管理器 - 避免频繁push/pop"""
    _thread_local = threading.local()
    _global_context = None
    _lock = threading.Lock()
    _ref_count = 0
    
    @classmethod
    def get_context(cls):
        """获取当前线程的CUDA上下文（每个线程只push一次）"""
        if not hasattr(cls._thread_local, 'context'):
            with cls._lock:
                if cls._global_context is None:
                    device = cuda.Device(0)
                    cls._global_context = device.retain_primary_context()
                    print("✅ Created global CUDA primary context")
                # 每个线程独立push一次
                cls._global_context.push()
                cls._thread_local.context = cls._global_context
                cls._ref_count += 1
                print(f"Thread {threading.current_thread().name}: CUDA context pushed (ref: {cls._ref_count})")
        return cls._thread_local.context
    
    @classmethod
    def release_thread(cls):
        """释放当前线程的CUDA上下文"""
        if hasattr(cls._thread_local, 'context'):
            cls._thread_local.context.pop()
            delattr(cls._thread_local, 'context')
            cls._ref_count -= 1
            print(f"Thread {threading.current_thread().name}: CUDA context popped (ref: {cls._ref_count})")


class OSNetTensorRTInference:
    """OSNet TensorRT推理类 - 线程局部上下文优化版"""
    
    def __init__(self, engine_path, input_size=(256, 128), batch_size=1):
        self.input_size = input_size
        self.height, self.width = input_size
        self.feature_dim = 512
        self.batch_size = batch_size
        self.engine_path = engine_path
        
        # 获取线程局部上下文（只push一次）
        self.cuda_ctx = ThreadLocalCUDAManager.get_context()
        
        # 加载引擎
        print(f"Loading TensorRT engine: {engine_path}")
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
                    print(f"  Input: {name}")
                else:
                    self.output_names.append(name)
                    print(f"  Output: {name}")
            
            # 设置具体的输入shape
            input_shape = (self.batch_size, 3, self.height, self.width)
            self.context.set_input_shape(self.input_names[0], input_shape)
            
            # 分配内存（每个线程独立）
            self.inputs = []
            self.outputs = []
            self.stream = cuda.Stream()
            
            for name in self.input_names:
                shape = self.context.get_tensor_shape(name)
                host = np.empty(shape, dtype=np.float32)
                device = cuda.mem_alloc(host.nbytes)
                self.inputs.append({'name': name, 'host': host, 'device': device})
                self.context.set_tensor_address(name, int(device))
            
            for name in self.output_names:
                shape = self.context.get_tensor_shape(name)
                host = np.empty(shape, dtype=np.float32)
                device = cuda.mem_alloc(host.nbytes)
                self.outputs.append({'name': name, 'host': host, 'device': device})
                self.context.set_tensor_address(name, int(device))
            
            print(f"✅ OSNet TensorRT loaded successfully (thread: {threading.current_thread().name})")
            
        except Exception as e:
            print(f"❌ Failed to load OSNet engine: {e}")
            raise
    
    def preprocess(self, image):
        """预处理图像 - 快速版本"""
        # 直接resize
        if len(image.shape) == 3:
            img = cv2.resize(image, (self.width, self.height))
        else:
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (self.width, self.height))
        
        # BGR转RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 归一化（使用更快的in-place操作）
        img = img.astype(np.float32, copy=False) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # 转换维度 CHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        
        return np.ascontiguousarray(img)
    
    def extract_feature(self, image):
        """提取单个图像的特征向量 - 无上下文切换开销"""
        if image is None or image.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        try:
            input_tensor = self.preprocess(image)
            
            # 拷贝到GPU
            cuda.memcpy_htod_async(self.inputs[0]['device'], input_tensor, self.stream)
            
            # 推理
            success = self.context.execute_async_v3(self.stream.handle)
            if not success:
                raise RuntimeError("TensorRT execution failed")
            
            # 拷贝回CPU
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            
            self.stream.synchronize()
            
            # 获取特征并归一化
            feature = self.outputs[0]['host'][0].flatten()
            norm = np.linalg.norm(feature)
            if norm > 1e-6:
                feature = feature / norm
            
            return feature
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def extract_features_batch(self, images):
        """批量提取特征向量"""
        if not images:
            return []
        
        try:
            batch_size = len(images)
            
            # 检查是否需要重新分配缓冲区
            current_shape = self.context.get_tensor_shape(self.input_names[0])
            if current_shape[0] != batch_size:
                self._reallocate_buffers(batch_size)
            
            # 批量预处理
            batch_tensor = []
            for img in images:
                processed = self.preprocess(img)
                batch_tensor.append(processed)
            
            batch_input = np.concatenate(batch_tensor, axis=0)
            
            # 拷贝到GPU
            cuda.memcpy_htod_async(self.inputs[0]['device'], batch_input, self.stream)
            
            # 推理
            success = self.context.execute_async_v3(self.stream.handle)
            if not success:
                raise RuntimeError("TensorRT batch execution failed")
            
            # 拷贝回CPU
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            
            self.stream.synchronize()
            
            # 提取并归一化特征
            features = []
            output_data = self.outputs[0]['host']
            for i in range(batch_size):
                feature = output_data[i].flatten()
                norm = np.linalg.norm(feature)
                if norm > 1e-6:
                    feature = feature / norm
                features.append(feature)
            
            return features
            
        except Exception as e:
            print(f"Batch feature extraction error: {e}")
            return [np.zeros(self.feature_dim, dtype=np.float32) for _ in range(len(images))]
    
    def _reallocate_buffers(self, batch_size):
        """重新分配缓冲区（当batch size改变时）"""
        # 释放旧内存
        for inp in self.inputs:
            if 'device' in inp:
                inp['device'].free()
        for out in self.outputs:
            if 'device' in out:
                out['device'].free()
        
        self.inputs = []
        self.outputs = []
        
        # 设置新的输入shape
        new_shape = (batch_size, 3, self.height, self.width)
        self.context.set_input_shape(self.input_names[0], new_shape)
        
        # 重新分配
        for name in self.input_names:
            shape = self.context.get_tensor_shape(name)
            host = np.empty(shape, dtype=np.float32)
            device = cuda.mem_alloc(host.nbytes)
            self.inputs.append({'name': name, 'host': host, 'device': device})
            self.context.set_tensor_address(name, int(device))
        
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            host = np.empty(shape, dtype=np.float32)
            device = cuda.mem_alloc(host.nbytes)
            self.outputs.append({'name': name, 'host': host, 'device': device})
            self.context.set_tensor_address(name, int(device))
    
    def __del__(self):
        """清理GPU内存"""
        try:
            if hasattr(self, 'stream'):
                self.stream.synchronize()
            for inp in self.inputs:
                if 'device' in inp:
                    inp['device'].free()
            for out in self.outputs:
                if 'device' in out:
                    out['device'].free()
        except:
            pass


class OSNetTensorRTReID:
    """OSNet TensorRT ReID模型，用于BOTSort跟踪器"""
    
    def __init__(self, engine_path, input_size=(256, 128)):
        self.inference = OSNetTensorRTInference(engine_path, input_size, batch_size=1)
        self.feature_dim = 512
    
    def extract_feature(self, image):
        return self.inference.extract_feature(image)
    
    def extract_features_batch(self, images):
        return self.inference.extract_features_batch(images)
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'inference'):
            del self.inference


def main():
    engine_path = "/home/wheeltec/Ebike_Human_Follower/src/yolov8_pytorch_detect_tracking/models/osnet_x0_25.engine"
    
    if not os.path.exists(engine_path):
        print(f"❌ Engine not found: {engine_path}")
        return
    
    print("="*60)
    print("Initializing OSNet TensorRT Model (Optimized)")
    print("="*60)
    
    # 创建推理器
    inferencer = OSNetTensorRTInference(engine_path, input_size=(256, 128), batch_size=1)
    
    # 测试性能
    test_img = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
    
    print("\nTesting performance...")
    
    # 预热
    for _ in range(5):
        _ = inferencer.extract_feature(test_img)
    
    # 测试
    times = []
    for _ in range(20):
        start = time.perf_counter()
        feature = inferencer.extract_feature(test_img)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    avg_time = np.mean(times)
    print(f"Feature shape: {feature.shape}")
    print(f"Feature norm: {np.linalg.norm(feature):.4f}")
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"FPS: {1000/avg_time:.1f}")
    
    print("\n✅ OSNet TensorRT model ready (optimized for multi-threading)!")
    print("="*60)
    
    # 清理
    del inferencer


if __name__ == '__main__':
    main()