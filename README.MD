
# 1.build
colcon build --packages-select yolov8_trt_detect_tracking

# 2. run
 source install/setup.bash
 ros2 launch yolov8_trt_detect_tracking yolov8_trt_detect_tracking.launch.py 
 
 
# 3. echo

ros2 topic echo /keypoint_tracks


# 4.result

header:
  stamp:
    sec: 1764656961
    nanosec: 514896896
  frame_id: camera_link
polygon:
  points:
  - x: 1.0
    y: 1.0
    z: 2.0
  - x: 309.0
    y: 73.0
    z: 0.0
  - x: 639.0
    y: 471.0
    z: 0.0



# 跟踪找回逻辑：

目标跟踪中
    ↓ (目标消失)
目标丢失 → 启动主动找回
    ↓ (搜索候选目标)
ReID匹配 → 相似度≥阈值 → 匹配成功 → 身份切换 → 恢复跟踪
    ↓ (匹配失败)          ↓ (验证失败)
保持丢失状态 ←────── 目标重现 → ReID验证 → 验证成功 → 恢复跟踪
