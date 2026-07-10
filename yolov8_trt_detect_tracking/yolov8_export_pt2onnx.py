from ultralytics import YOLO

# 加载模型
model = YOLO('/home/peng/runs/detect/train-2/weights/yolov8n_0701.pt')

# 导出单输出 ONNX
# 使用 dynamic=False 固定输入尺寸，避免动态轴导致多输出
model.export(
    format='onnx',
    imgsz=640,
    opset=11,
    dynamic=False,          # 固定 batch 和尺寸
    simplify=False,         # 保持原图结构，避免简化导致输出拆分
    nms=False,              # 确保不添加 NMS 后处理节点
    batch=1,                # 固定 batch
    device=0                # 使用 GPU
)

