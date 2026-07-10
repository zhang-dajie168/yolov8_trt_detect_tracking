#!/usr/bin/env python3
import tensorrt as trt
import os
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_path, engine_path, fp16=False, workspace_mb=1024, input_shape=(1,3,640,640)):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Loading ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("Parse errors:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False

    # 获取输入张量
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    print(f"Input: {input_name}, shape: {input_tensor.shape}")

    # 固定输入形状（设置 optimization profile）
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, input_shape, input_shape, input_shape)

    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")
    elif fp16:
        print("Warning: FP16 not supported, using FP32")

    # 打印输出张量信息
    print("Output tensors:")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  {i}: {out.name}, shape: {out.shape}")

    print("Building engine... (may take a few minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Build failed")
        return False

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    size_mb = os.path.getsize(engine_path) / (1024*1024)
    print(f"Engine saved to {engine_path} ({size_mb:.2f} MB)")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', default='/home/wheeltec/yolov8_pytorch_detect_tracking/models/yolov8n_0701.onnx')
    parser.add_argument('--engine', default='/home/wheeltec/yolov8_pytorch_detect_tracking/models/yolov8n_0701.engine')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16')
    parser.add_argument('--workspace', type=int, default=1024, help='Workspace in MB')
    parser.add_argument('--input_size', type=int, default=640, help='Input size (width=height)')
    args = parser.parse_args()

    input_shape = (1, 3, args.input_size, args.input_size)
    build_engine(args.onnx, args.engine, args.fp16, args.workspace, input_shape)

if __name__ == '__main__':
    main()
