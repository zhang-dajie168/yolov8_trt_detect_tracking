#!/usr/bin/env python3
"""
TensorRT模型转换脚本 - 适配TensorRT 10.3 API
"""

import os
import sys

def main():
    # 先导出ONNX
    print("Exporting ONNX...")
    os.system("""
python -c "
from ultralytics import YOLO
model = YOLO('/home/peng/Ebike_Human_Follower/src/yolov8_pytorch_detect_tracking/models/yolov8n-8.0.pt')
model.export(format='onnx', imgsz=640, half=False, opset=12, simplify=True, device='cpu')
print('ONNX export completed')
"
    """)
    
    # 重命名
    src = "/home/peng/Ebike_Human_Follower/src/yolov8_pytorch_detect_tracking/models/yolov8n-8.0.onnx"
    dst = "/home/peng/Ebike_Human_Follower/src/yolov8_pytorch_detect_tracking/models/yolov8.onnx"
    if os.path.exists(src):
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(src, dst)
        print(f"ONNX saved to: {dst}")
    
    # 使用Python API转换 - 修正版
    print("\nConverting to TensorRT...")
    
    python_code = '''
import tensorrt as trt
import numpy as np
import os

# 配置
onnx_path = "/home/wheeltec/Ebike_Human_Follower/src/yolov8_pytorch_detect_tracking/models/yolov8.onnx"
engine_path = "/home/wheeltec/Ebike_Human_Follower/src/yolov8_pytorch_detect_tracking/models/yolov8.engine"

# 创建logger
logger = trt.Logger(trt.Logger.INFO)

# 创建builder
builder = trt.Builder(logger)

# 创建网络
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# 创建解析器
parser = trt.OnnxParser(network, logger)

# 解析ONNX
print("Parsing ONNX...")
with open(onnx_path, 'rb') as f:
    if not parser.parse(f.read()):
        print("Parse failed")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        exit(1)

print("Parse success")

# 打印网络信息
print(f"Network inputs: {network.num_inputs}")
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
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
    print("FP16 enabled")

# 创建优化配置文件
profile = builder.create_optimization_profile()

# 设置输入形状
for i in range(network.num_inputs):
    inp = network.get_input(i)
    shape = inp.shape
    # 转换为具体的形状
    min_shape = tuple([1 if dim == -1 else dim for dim in shape])
    opt_shape = tuple([1 if dim == -1 else dim for dim in shape])
    max_shape = tuple([1 if dim == -1 else dim for dim in shape])
    
    profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
    print(f"Set shape for {inp.name}: {min_shape}")

config.add_optimization_profile(profile)

# 构建引擎 - 使用build_serialized_network而不是build_engine
print("Building engine...")
serialized_network = builder.build_serialized_network(network, config)

if serialized_network is None:
    print("Build failed")
    exit(1)

# 保存引擎
print(f"Saving engine to {engine_path}")
with open(engine_path, 'wb') as f:
    f.write(serialized_network)

print(f"Success! Engine size: {os.path.getsize(engine_path) / 1024 / 1024:.2f} MB")
'''
    
    # 执行转换
    with open('temp_convert.py', 'w') as f:
        f.write(python_code)
    
    result = os.system('python temp_convert.py')
    os.remove('temp_convert.py')
    
    if result == 0:
        print("\n✅ Conversion completed!")
        return True
    else:
        print("\n❌ Conversion failed")
        return False

if __name__ == '__main__':
    success = main()
    if not success:
        print("\nTrying alternative conversion method...")
        
        # 备用方法：使用更简单的配置
        python_code_alt = '''
import tensorrt as trt
import os

onnx_path = "/home/wheeltec/Ebike_Human_Follower/src/yolov8_pytorch_detect_tracking/models/yolov8.onnx"
engine_path = "/home/wheeltec/Ebike_Human_Follower/src/yolov8_pytorch_detect_tracking/models/yolov8.engine"

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

print("Parsing ONNX...")
with open(onnx_path, 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        exit(1)

print("ONNX parsed successfully")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
    print("FP16 enabled")

# 不使用优化配置文件，直接构建
print("Building engine (this may take a few minutes)...")
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    print("Build failed")
    exit(1)

with open(engine_path, 'wb') as f:
    f.write(serialized_engine)

print(f"Engine saved to {engine_path}")
print(f"Size: {os.path.getsize(engine_path) / 1024 / 1024:.2f} MB")
'''
        
        with open('temp_convert_alt.py', 'w') as f:
            f.write(python_code_alt)
        
        result = os.system('python temp_convert_alt.py')
        os.remove('temp_convert_alt.py')
        
        if result == 0:
            print("\n✅ Conversion completed with alternative method!")
        else:
            print("\n❌ Both conversion methods failed")
            sys.exit(1)