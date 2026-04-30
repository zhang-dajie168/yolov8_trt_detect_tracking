#!/usr/bin/env python3
"""
OSNet TensorRT推理类 - 优化版（修复上下文冲突）
"""

import os
import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 全局CUDA上下文管理器
class CUDAManager:
    """全局CUDA上下文管理器"""
    _instance = None
    _context = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_context(self):
        if self._context is None:
            device = cuda.Device(0)
            self._context = device.retain_primary_context()
        return self._context
    
    def push(self):
        ctx = self.get_context()
        ctx.push()
    
    def pop(self):
        if self._context:
            self._context.pop()


class OSNetTensorRTInference:
    """OSNet TensorRT推理类 - 使用全局上下文管理"""
    
    def __init__(self, engine_path, input_size=(256, 128), batch_size=1):
        self.input_size = input_size
        self.height, self.width = input_size
        self.feature_dim = 512
        self.batch_size = batch_size
        self.cuda_manager = CUDAManager()
        
        # 确保CUDA上下文
        self.cuda_manager.push()
        
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
                    print(f"Input: {name}, shape: {self.engine.get_tensor_shape(name)}")
                else:
                    self.output_names.append(name)
                    print(f"Output: {name}, shape: {self.engine.get_tensor_shape(name)}")
            
            # 设置具体的输入shape
            input_shape = (self.batch_size, 3, self.height, self.width)
            self.context.set_input_shape(self.input_names[0], input_shape)
            print(f"Set input shape to: {input_shape}")
            
            # 分配内存
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
            
            print("✅ OSNet TensorRT engine loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load OSNet engine: {e}")
            raise
        finally:
            self.cuda_manager.pop()
    
    def preprocess(self, image):
        """预处理图像"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img = image
        
        img = cv2.resize(img, (self.width, self.height))
        
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        
        return np.ascontiguousarray(img.astype(np.float32))
    
    def _ensure_context(self):
        """确保CUDA上下文"""
        self.cuda_manager.push()
    
    def extract_feature(self, image):
        """提取单个图像的特征向量"""
        self._ensure_context()
        
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
            if norm > 0:
                feature = feature / norm
            
            return feature
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
        finally:
            self.cuda_manager.pop()
    
    def extract_features_batch(self, images):
        """批量提取特征向量"""
        if len(images) == 0:
            return []
        
        self._ensure_context()
        
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
                if norm > 0:
                    feature = feature / norm
                features.append(feature)
            
            return features
            
        except Exception as e:
            print(f"Batch feature extraction error: {e}")
            return [np.zeros(self.feature_dim, dtype=np.float32) for _ in range(len(images))]
        finally:
            self.cuda_manager.pop()
    
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
    print("Initializing OSNet TensorRT Model")
    print("="*60)
    
    # 创建推理器
    inferencer = OSNetTensorRTInference(engine_path, input_size=(256, 128), batch_size=1)
    
    # 测试真实图片（如果存在）
    test_img_path = "/home/wheeltec/Ebike_Human_Follower/src/yolov8_pytorch_detect_tracking/test_img"
    if os.path.exists(test_img_path):
        test_images = [f for f in os.listdir(test_img_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if test_images:
            img_path = os.path.join(test_img_path, test_images[0])
            test_img = cv2.imread(img_path)
            if test_img is not None:
                print(f"\nTesting with real image: {test_images[0]}")
                print(f"Image shape: {test_img.shape}")
                
                # 测试单张
                start = time.time()
                feature = inferencer.extract_feature(test_img)
                elapsed = (time.time() - start) * 1000
                
                print(f"Feature shape: {feature.shape}")
                print(f"Feature norm: {np.linalg.norm(feature):.4f}")
                print(f"Inference time: {elapsed:.2f} ms")
                
                # 测试批处理
                batch_images = [test_img, test_img]
                start = time.time()
                features = inferencer.extract_features_batch(batch_images)
                elapsed = (time.time() - start) * 1000
                print(f"Batch inference (2 images): {elapsed:.2f} ms")
                print(f"Batch features count: {len(features)}")
            else:
                print("❌ Failed to load test image")
        else:
            print("⚠️  No test images found, using random test")
            # 使用随机测试
            test_img = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
            print("\nTesting with random image...")
            start = time.time()
            feature = inferencer.extract_feature(test_img)
            elapsed = (time.time() - start) * 1000
            print(f"Feature shape: {feature.shape}")
            print(f"Feature norm: {np.linalg.norm(feature):.4f}")
            print(f"Inference time: {elapsed:.2f} ms")
    else:
        # 使用随机测试
        test_img = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
        print("\nTesting with random image...")
        start = time.time()
        feature = inferencer.extract_feature(test_img)
        elapsed = (time.time() - start) * 1000
        print(f"Feature shape: {feature.shape}")
        print(f"Feature norm: {np.linalg.norm(feature):.4f}")
        print(f"Inference time: {elapsed:.2f} ms")
    
    print("\n✅ OSNet TensorRT model ready!")
    print("="*60)
    
    # 手动清理（确保在程序结束前释放资源）
    del inferencer


if __name__ == '__main__':
    main()