#!/usr/bin/env python3
"""
OSNet x0_25 模型转TensorRT脚本
"""

import os
import sys
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 导入OSNet模型
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from OSNet import osnet_x0_25

class OSNetTensorRTConverter:
    def __init__(self, model_path, input_size=(256, 128), precision='fp16'):
        """
        初始化OSNet转换器
        
        Args:
            model_path: osnet_x0_25.pth 模型文件路径
            input_size: 输入图像尺寸 (height, width)
            precision: 精度 ('fp32' or 'fp16')
        """
        self.model_path = model_path
        self.input_size = input_size  # (height, width) = (256, 128)
        self.precision = precision
        self.logger = trt.Logger(trt.Logger.INFO)
        self.feature_dim = 512  # OSNet x0_25 输出特征维度
        
        # 加载PyTorch模型
        self.load_pytorch_model()
        
    def load_pytorch_model(self):
        """加载PyTorch OSNet模型"""
        print("="*60)
        print("Loading PyTorch OSNet x0_25 model")
        print("="*60)
        
        # 创建模型
        self.model = osnet_x0_25(num_classes=1000, pretrained=False)
        
        # 加载权重
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 移除'module.'前缀和分类层
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            # 跳过分类层（classifier），因为我们只需要特征提取
            if not k.startswith('classifier'):
                new_state_dict[k] = v
        
        # 加载权重
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
        print(f"模型加载完成")
        print(f"  缺失的键: {len(missing_keys)}")
        print(f"  意外的键: {len(unexpected_keys)}")
        
        # 设置为评估模式
        self.model.eval()
        
        # 导出ONNX
        self.export_to_onnx()
    
    def export_to_onnx(self):
        """导出ONNX模型"""
        print("\n" + "="*60)
        print("Exporting to ONNX")
        print("="*60)
        
        onnx_path = self.model_path.replace('.pth', '.onnx')
        
        # 创建示例输入
        dummy_input = torch.randn(1, 3, self.input_size[0], self.input_size[1])
        
        # 导出ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX exported: {onnx_path}")
        self.onnx_path = onnx_path
        
        # 转换为TensorRT
        self.convert_to_tensorrt()
    
    def convert_to_tensorrt(self):
        """转换ONNX到TensorRT"""
        print("\n" + "="*60)
        print("Converting ONNX to TensorRT")
        print("="*60)
        
        engine_path = self.model_path.replace('.pth', '.engine')
        
        # 创建builder
        builder = trt.Builder(self.logger)
        
        # 创建网络
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        
        # 创建ONNX解析器
        parser = trt.OnnxParser(network, self.logger)
        
        # 解析ONNX
        print(f"Parsing ONNX: {self.onnx_path}")
        with open(self.onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                print("❌ ONNX parsing failed:")
                for i in range(parser.num_errors):
                    print(f"  {parser.get_error(i)}")
                return False
        
        print("✅ ONNX parsed successfully")
        
        # 打印网络信息
        print(f"\nNetwork inputs: {network.num_inputs}")
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            print(f"  Input {i}: {inp.name}, shape: {inp.shape}")
        
        print(f"Network outputs: {network.num_outputs}")
        for i in range(network.num_outputs):
            out = network.get_output(i)
            print(f"  Output {i}: {out.name}, shape: {out.shape}")
        
        # 创建配置
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        
        # 设置精度
        if self.precision == 'fp16' and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✅ FP16 enabled")
        else:
            print("Using FP32 precision")
        
        # 创建优化配置文件
        profile = builder.create_optimization_profile()
        
        # 设置输入形状（支持动态batch）
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            shape = inp.shape
            # 设置动态batch: batch从1到8
            min_shape = tuple([1 if dim == -1 else dim for dim in shape])
            opt_shape = tuple([4 if dim == -1 else dim for dim in shape])
            max_shape = tuple([8 if dim == -1 else dim for dim in shape])
            
            profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
            print(f"Set shape for {inp.name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
        
        config.add_optimization_profile(profile)
        
        # 构建引擎
        print("\nBuilding TensorRT engine (this may take 1-2 minutes)...")
        serialized_network = builder.build_serialized_network(network, config)
        
        if serialized_network is None:
            print("❌ Engine building failed")
            return False
        
        # 保存引擎
        print(f"Saving engine to: {engine_path}")
        with open(engine_path, 'wb') as f:
            f.write(serialized_network)
        
        # 显示结果
        size_mb = os.path.getsize(engine_path) / 1024 / 1024
        print(f"\n✅ TensorRT engine saved successfully!")
        print(f"   Path: {engine_path}")
        print(f"   Size: {size_mb:.2f} MB")
        
        self.engine_path = engine_path
        return True




def main():
    # 路径配置
    model_path = "/home/wheeltec/Ebike_Human_Follower/src/yolov8_pytorch_detect_tracking/models/osnet_x0_25.pth"
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # 转换模型
    print("="*60)
    print("OSNet x0_25 TensorRT Conversion")
    print("="*60)
    
    converter = OSNetTensorRTConverter(
        model_path=model_path,
        input_size=(256, 128),  # OSNet标准输入尺寸
        precision='fp16'
    )
    

if __name__ == '__main__':
    main()