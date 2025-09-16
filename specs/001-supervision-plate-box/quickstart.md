# Quickstart: Supervision库可视化集成

**Feature**: 使用Supervision库增强可视化功能
**Date**: 2025-09-15
**Estimated Time**: 10-15分钟

## 概述

本快速开始指南演示如何使用新的supervision库增强的可视化功能来绘制车辆检测和车牌OCR结果。新实现提供更专业的视觉效果和更好的性能，同时保持完全的向后兼容性。

## 前置条件

### 环境要求
```bash
# Python环境
python >= 3.10

# 必需依赖
pip install supervision opencv-python pillow numpy

# 可选：用于性能基准测试
pip install matplotlib seaborn
```

### 字体要求
```bash
# 确保中文字体可用
ls -la SourceHanSans-VF.ttf  # 项目根目录
# 或系统字体
ls -la /usr/share/fonts/truetype/  # Linux
ls -la /System/Library/Fonts/      # macOS
```

## 快速开始

### 1. 基本使用 (零代码修改)

现有代码保持完全不变，supervision库作为内部实现升级：

```python
# 现有调用方式 - 无需修改
from utils.drawing import draw_detections

# 示例数据
image = cv2.imread("test_image.jpg")
detections = [[[100, 150, 300, 400, 0.95, 0], [350, 200, 500, 350, 0.87, 1]]]
class_names = ["vehicle", "plate"]
colors = [(255, 0, 0), (0, 255, 0)]  # 红色车辆，绿色车牌

# 基础检测绘制
result_image = draw_detections(image, detections, class_names, colors)
cv2.imshow("Detection Result", result_image)
cv2.waitKey(0)
```

### 2. 带OCR结果的完整示例

```python
from utils.drawing import draw_detections
import cv2

# 准备测试数据
def prepare_test_data():
    """准备测试数据"""
    # 加载测试图像
    image = cv2.imread("sample_vehicle.jpg")

    # 模拟检测结果: [x1, y1, x2, y2, confidence, class_id]
    detections = [[
        [100, 150, 300, 400, 0.95, 0],  # 车辆检测
        [350, 320, 450, 360, 0.89, 1]   # 车牌检测
    ]]

    # 类别配置
    class_names = ["vehicle", "plate"]
    colors = [(255, 0, 0), (0, 255, 0)]  # BGR格式

    # 车牌OCR结果
    plate_results = [
        None,  # 车辆无OCR结果
        {      # 车牌OCR结果
            "plate_text": "京A12345",
            "color": "蓝牌",
            "layer": "单层",
            "confidence": 0.92,
            "should_display_ocr": True
        }
    ]

    return image, detections, class_names, colors, plate_results

# 执行完整示例
def main():
    """主函数"""
    # 准备数据
    image, detections, class_names, colors, plate_results = prepare_test_data()

    # 使用supervision增强绘制
    result_image = draw_detections(
        image=image,
        detections=detections,
        class_names=class_names,
        colors=colors,
        plate_results=plate_results,
        font_path="SourceHanSans-VF.ttf"
    )

    # 显示结果
    cv2.imshow("Enhanced Visualization", result_image)
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    cv2.imwrite("enhanced_result.jpg", result_image)
    print("Result saved to enhanced_result.jpg")

if __name__ == "__main__":
    main()
```

### 3. 性能基准测试

验证supervision库的性能提升：

```python
from utils.drawing import benchmark_drawing_performance
import numpy as np

def run_performance_test():
    """运行性能基准测试"""
    # 创建测试图像
    test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # 创建大量检测对象进行压力测试
    detections = [[
        [i*50, j*50, (i+1)*50, (j+1)*50, 0.9, i%2]
        for i in range(20) for j in range(10)
    ]]  # 200个检测对象

    class_names = ["vehicle", "plate"]
    colors = [(255, 0, 0), (0, 255, 0)]

    # 运行基准测试
    results = benchmark_drawing_performance(
        image=test_image,
        detections_data=detections,
        class_names=class_names,
        colors=colors,
        iterations=50
    )

    # 打印结果
    print("\n=== Performance Benchmark Results ===")
    print(f"PIL Backend: {results['pil_avg_time']:.2f}ms")
    print(f"Supervision Backend: {results['supervision_avg_time']:.2f}ms")
    print(f"Improvement Ratio: {results['improvement_ratio']:.2f}x")
    print(f"Objects/Second: {results.get('objects_per_second', 'N/A')}")

# 运行性能测试
run_performance_test()
```

### 4. 配置自定义

通过环境变量或配置文件自定义行为：

```python
import os

# 环境变量配置
os.environ["USE_SUPERVISION"] = "true"        # 启用supervision
os.environ["FALLBACK_TO_PIL"] = "true"        # 启用PIL回退
os.environ["PERFORMANCE_LOGGING"] = "false"   # 关闭性能日志

# 使用自定义配置
from utils.drawing import draw_detections

result = draw_detections(
    image, detections, class_names, colors,
    use_supervision=True,  # 明确启用supervision
    font_path="custom_font.ttf"  # 自定义字体
)
```

## 验证检查清单

### ✅ 基础功能验证
- [ ] 现有代码无需修改即可运行
- [ ] 检测框正确显示（颜色、粗细、位置）
- [ ] 类别标签正确显示（名称、置信度）
- [ ] 中文字体正确加载和显示

### ✅ OCR功能验证
- [ ] 车牌文字正确显示（中文+数字+字母）
- [ ] 车牌颜色信息正确显示
- [ ] 车牌层级信息正确显示
- [ ] OCR文字位置智能调整（避免重叠）

### ✅ 性能验证
- [ ] 绘制时间 < 50ms（20个对象）
- [ ] 性能提升 > 2倍（相比PIL）
- [ ] 内存使用稳定（无明显增长）
- [ ] 错误处理正常（fallback机制）

### ✅ 兼容性验证
- [ ] 输出格式为BGR numpy数组
- [ ] 与cv2.imshow()兼容
- [ ] 与cv2.imwrite()兼容
- [ ] 现有pipeline.py集成无问题

## 故障排除

### 常见问题及解决方案

#### 问题1: 中文字体显示为方块
```bash
# 解决方案：检查字体文件
ls -la SourceHanSans-VF.ttf
# 如果不存在，下载字体文件
wget https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSans.ttc
```

#### 问题2: supervision导入失败
```bash
# 解决方案：安装supervision库
pip install supervision>=0.16.0
# 或更新到最新版本
pip install --upgrade supervision
```

#### 问题3: 性能没有提升
```python
# 解决方案：检查是否启用supervision
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看日志输出确认使用的后端
result = draw_detections(image, detections, class_names, colors)
# 日志应显示 "Using supervision backend" 而不是 "Fallback to PIL"
```

#### 问题4: OCR文字位置不正确
```python
# 解决方案：检查plate_results格式
plate_results = [
    None,  # 非车牌检测
    {
        "plate_text": "京A12345",           # 必需
        "color": "蓝牌",                    # 必需
        "layer": "单层",                    # 必需
        "confidence": 0.92,                 # 必需
        "should_display_ocr": True          # 必需：控制是否显示
    }
]
```

#### 问题5: 性能测试失败
```python
# 解决方案：降低测试强度
results = benchmark_drawing_performance(
    image, detections, class_names, colors,
    iterations=10  # 从100降低到10
)
```

## 进阶使用

### 1. 自定义注释器配置

```python
# 高级配置示例（内部实现，用户一般不需要修改）
from utils.drawing_config import SupervisionConfig

config = SupervisionConfig(
    box_thickness=5,
    font_size=20,
    smart_position=True,
    enable_shadows=True
)

# 使用自定义配置
result = draw_detections_with_config(image, detections, config)
```

### 2. 批量图像处理

```python
import glob
from pathlib import Path

def process_image_batch(image_dir: str, output_dir: str):
    """批量处理图像"""
    Path(output_dir).mkdir(exist_ok=True)

    for img_path in glob.glob(f"{image_dir}/*.jpg"):
        image = cv2.imread(img_path)
        # ... 进行检测和OCR ...

        # 使用supervision绘制
        result = draw_detections(image, detections, class_names, colors, plate_results)

        # 保存结果
        output_path = Path(output_dir) / Path(img_path).name
        cv2.imwrite(str(output_path), result)
        print(f"Processed: {img_path} -> {output_path}")

# 批量处理
process_image_batch("input_images/", "output_images/")
```

### 3. 实时视频处理

```python
def process_video_stream(video_path: str):
    """实时视频流处理"""
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ... 进行检测和OCR ...

        # 使用supervision绘制（高性能）
        result_frame = draw_detections(frame, detections, class_names, colors, plate_results)

        # 显示结果
        cv2.imshow("Real-time Detection", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 处理视频
process_video_stream("test_video.mp4")
```

## 下一步

1. **运行完整测试**: 执行上述所有示例代码
2. **性能基准**: 对比新旧实现的性能差异
3. **集成测试**: 在完整pipeline中验证功能
4. **生产部署**: 逐步切换到supervision后端

## 支持和反馈

如果遇到问题或有改进建议：

1. 检查日志输出：`logging.getLogger("utils.drawing").setLevel(logging.DEBUG)`
2. 运行基准测试：确认性能提升符合预期
3. 提交Issue：描述问题和重现步骤
4. 查看文档：参考data-model.md和contracts/了解详细API

---

**快速开始完成**: 🎉 您已经准备好使用supervision库增强的可视化功能！