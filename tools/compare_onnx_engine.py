#!/usr/bin/env python3
"""
测试RT-DETR模型的compare_engine方法 - 多图片支持版本
比较ONNX模型和TensorRT引擎的推理结果精度，支持多张图片的可视化
"""

import sys
import logging
from pathlib import Path
import cv2
import numpy as np
import yaml
import os
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 添加项目路径到系统路径
project_root = Path(__file__).parent.parent  # 获取父目录作为项目根目录
sys.path.insert(0, str(project_root))

from infer_onnx import create_detector, RUN
from utils.drawing import draw_detections

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='比较ONNX模型和TensorRT引擎的推理结果精度，支持多张图片的可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
            使用示例:
            # 传统方式（无后缀，自动生成.onnx和.engine）
            %(prog)s --model-path models/rtdetr-20250729 --data-path data/苏州图片
            
            # 指定ONNX文件（自动推导.engine文件）
            %(prog)s --model-path models/yolov8s_640.onnx --data-path data/sample.jpg
            
            # 指定Engine文件（自动推导.onnx文件）  
            %(prog)s --model-path models/rtdetr-20250729.engine --conf-thres 0.5
            
            # 分别指定ONNX和Engine文件
            %(prog)s --model-path models/model.onnx --engine-path models/custom.engine
            
            # 其他参数组合（输入形状自动从ONNX模型读取）
            %(prog)s --model-path models/rtdetr-20250729.onnx --conf-thres 0.5 --rtol 1e-2
            '''
    )
    
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        default='models/rtdetr-20250729',
        help='模型路径（可带.onnx/.engine后缀，或不带后缀自动生成）'
    )
    
    parser.add_argument(
        '--engine-path', '-e',
        type=str,
        default=None,
        help='TensorRT引擎文件路径（可选，未指定时自动推导）'
    )
    
    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default=None,
        help='图片数据路径（可以是文件夹或具体图片路径，默认为None使用合成数据）'
    )
    
    parser.add_argument(
        '--model-type', '-t',
        type=str,
        default='rtdetr',
        choices=['rtdetr', 'yolo', 'rfdetr'],
        help='模型类型 (默认: rtdetr)'
    )
    
    
    parser.add_argument(
        '--conf-thres',
        type=float,
        default=0.25,
        help='置信度阈值 (默认: 0.25)'
    )
    
    parser.add_argument(
        '--rtol',
        type=float,
        default=1e0,
        help='相对容差 (默认: 1e-3)'
    )
    
    parser.add_argument(
        '--atol',
        type=float,
        default=1e-1,
        help='绝对容差 (默认: 1e-3)'
    )
    
    parser.add_argument(
        '--save-engine',
        action='store_true',
        help='当从ONNX构建引擎时是否保存引擎文件'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细输出'
    )
    
    return parser.parse_args()

def load_colors_and_class_names():
    """加载类别名称和颜色配置"""
    try:
        with open("configs/det_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        class_names = config["class_names"]
        colors = config["visual_colors"]
        return class_names, colors
    except Exception as e:
        logging.warning(f"无法加载配置文件: {e}, 使用默认配置")
        # 默认配置
        class_names = ["car", "plate", "person", "truck", "bus"]  
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        return class_names, colors

def post_process_raw_outputs(raw_outputs, detector, test_img, conf_thres=0.5):
    """对 compare_engine 返回的原始输出进行后处理，参考 __call__ 方法"""
    try:
        # 获取图像的原始尺寸
        original_shape = test_img.shape
        h_img, w_img, _ = original_shape
        
        # 预处理以获取scale和ratio_pad参数
        preprocess_result = detector._preprocess(test_img)
        if len(preprocess_result) == 3:
            _, scale, _ = preprocess_result
            ratio_pad = None
        else:
            _, scale, _, ratio_pad = preprocess_result
        
        # 调试：输出原始输出的统计信息
        if isinstance(raw_outputs, list) and len(raw_outputs) > 0:
            prediction = raw_outputs[0]
            logging.debug(f"原始输出形状: {prediction.shape}")
            if len(prediction.shape) == 3 and prediction.shape[2] > 4:
                # 检查分类分数部分的统计
                scores = prediction[:, :, 4:]
                logging.debug(f"分类分数统计: min={np.min(scores):.4f}, max={np.max(scores):.4f}, "
                            f"mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
        
        # 使用检测器的_postprocess方法，参考 __call__ 中的处理
        if type(detector).__name__ == 'RFDETROnnx':
            detections = detector._postprocess(raw_outputs, conf_thres)
        else:
            # 对于RT-DETR等模型，使用第一个输出
            prediction = raw_outputs[0] if isinstance(raw_outputs, list) else raw_outputs
            detections = detector._postprocess(prediction, conf_thres, scale=scale, ratio_pad=ratio_pad)
        
        # RFDETR和RT-DETR特殊处理：需要将坐标从输入尺寸缩放回原始尺寸
        if type(detector).__name__ in ['RFDETROnnx', 'RTDETROnnx'] and detections and len(detections) > 0:
            # RFDETR和RT-DETR的 _postprocess 返回的坐标是在输入图像尺寸上的
            # 需要缩放回原始图像尺寸
            input_h, input_w = detector.input_shape
            scale_x = w_img / input_w  # 原始宽度 / 输入宽度
            scale_y = h_img / input_h  # 原始高度 / 输入高度
            
            for detection in detections:
                if len(detection) > 0:
                    # detection的格式是 [x1, y1, x2, y2, conf, cls]
                    detection[:, [0, 2]] *= scale_x  # 缩放 x1, x2
                    detection[:, [1, 3]] *= scale_y  # 缩放 y1, y2
                    
                    # 确保坐标在图像边界内
                    detection[:, 0] = np.clip(detection[:, 0], 0, w_img)  # x1
                    detection[:, 1] = np.clip(detection[:, 1], 0, h_img)  # y1  
                    detection[:, 2] = np.clip(detection[:, 2], 0, w_img)  # x2
                    detection[:, 3] = np.clip(detection[:, 3], 0, h_img)  # y2
        
        return detections, original_shape
    except Exception as e:
        logging.error(f"后处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def visualize_compare_results(test_images, run_results, class_names, colors, detector=None, engine_path=None):
    """可视化 compare_engine 的结果，支持多张图片"""
    try:
        # 创建带引擎文件名的输出目录
        if engine_path:
            engine_name = Path(engine_path).stem  # 获取不带后缀的引擎文件名
        else:
            engine_name = "default"
        output_dir = f"{RUN}/{engine_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 确保 test_images 是列表
        if not isinstance(test_images, list):
            test_images = [test_images]
        
        # 获取运行器名称
        runner_names = list(run_results.keys())
        onnx_runner_name = None
        trt_runner_name = None
        
        for name in runner_names:
            if 'onnxrt-runner' in name:
                onnx_runner_name = name
            elif 'trt-runner' in name:
                trt_runner_name = name
        
        logging.info(f"找到运行器: ONNX={onnx_runner_name}, TRT={trt_runner_name}")
        logging.info(f"将处理 {len(test_images)} 张图片")
        
        # 处理每张图片
        all_comparison_images = []
        
        if onnx_runner_name and detector:
            # run_results中每个runner的值是IterationResult列表
            onnx_iterations = run_results[onnx_runner_name]
            trt_iterations = run_results[trt_runner_name] if trt_runner_name else None
            
            num_iterations = len(onnx_iterations)
            num_images = len(test_images)
            logging.info(f"迭代次数: {num_iterations}, 图片数量: {num_images}")
            
            # 处理每个迭代(每张图片)
            for iteration_idx in range(min(num_iterations, num_images)):
                test_img = test_images[iteration_idx]
                logging.info(f"处理第 {iteration_idx + 1} 张图片...")
                
                # 获取当前迭代的ONNX结果
                current_iteration = onnx_iterations[iteration_idx]
                onnx_raw = [current_iteration[output_name] for output_name in detector.output_names]
                
                logging.info(f"图片 {iteration_idx + 1} ONNX原始输出形状: {[arr.shape for arr in onnx_raw]}")
                
                # 尝试对ONNX原始输出进行后处理和画图
                onnx_result_img = None
                try:
                    onnx_detections, _ = post_process_raw_outputs(onnx_raw, detector, test_img)
                    if onnx_detections and len(onnx_detections) > 0 and len(onnx_detections[0]) > 0:
                        onnx_result_img = draw_detections(test_img.copy(), onnx_detections, class_names, colors)
                        logging.info(f"图片 {iteration_idx + 1} ONNX检测到 {len(onnx_detections[0])} 个目标")
                    else:
                        onnx_result_img = test_img.copy()
                        cv2.putText(onnx_result_img, f"ONNX: No Detection (Image {iteration_idx + 1})", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        logging.info(f"图片 {iteration_idx + 1} ONNX未检测到目标")
                except Exception as e:
                    logging.warning(f"图片 {iteration_idx + 1} ONNX后处理失败: {e}")
                    onnx_result_img = test_img.copy()
                    cv2.putText(onnx_result_img, f"ONNX: Processing Failed (Image {iteration_idx + 1})", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 如果有 TensorRT 结果，也进行比较
                trt_result_img = None
                if trt_iterations and iteration_idx < len(trt_iterations):
                    trt_current_iteration = trt_iterations[iteration_idx]
                    trt_raw = [trt_current_iteration[output_name] for output_name in detector.output_names]
                    
                    logging.info(f"图片 {iteration_idx + 1} TensorRT原始输出形状: {[arr.shape for arr in trt_raw]}")
                    
                    try:
                        trt_detections, _ = post_process_raw_outputs(trt_raw, detector, test_img)
                        if trt_detections and len(trt_detections) > 0 and len(trt_detections[0]) > 0:
                            trt_result_img = draw_detections(test_img.copy(), trt_detections, class_names, colors)
                            logging.info(f"图片 {iteration_idx + 1} TensorRT检测到 {len(trt_detections[0])} 个目标")
                            
                            # 调试：比较ONNX和TensorRT的检测结果
                            logging.info(f"图片 {iteration_idx + 1} ONNX vs TensorRT 检测结果比较:")
                            onnx_count = len(onnx_detections[0]) if onnx_detections and len(onnx_detections) > 0 else 0
                            trt_count = len(trt_detections[0])
                            logging.info(f"  ONNX: {onnx_count} 个目标")
                            logging.info(f"  TRT:  {trt_count} 个目标")
                            
                            # 检查原始输出的差异
                            onnx_output = onnx_raw[0]
                            trt_output = trt_raw[0] 
                            max_diff = np.abs(onnx_output - trt_output).max()
                            logging.info(f"  原始输出最大差异: {max_diff:.6f}")
                            
                            # 详细比较前几个检测结果
                            if onnx_detections and trt_detections and len(onnx_detections[0]) > 0 and len(trt_detections[0]) > 0:
                                logging.info(f"图片 {iteration_idx + 1} 详细检测结果比较 (前3个):")
                                for i in range(min(3, len(onnx_detections[0]), len(trt_detections[0]))):
                                    onnx_det = onnx_detections[0][i]
                                    trt_det = trt_detections[0][i]
                                    logging.info(f"  检测 {i+1}:")
                                    logging.info(f"    ONNX: [{onnx_det[0]:.1f}, {onnx_det[1]:.1f}, {onnx_det[2]:.1f}, {onnx_det[3]:.1f}] conf:{onnx_det[4]:.3f} cls:{int(onnx_det[5])}")
                                    logging.info(f"    TRT:  [{trt_det[0]:.1f}, {trt_det[1]:.1f}, {trt_det[2]:.1f}, {trt_det[3]:.1f}] conf:{trt_det[4]:.3f} cls:{int(trt_det[5])}")
                                    coord_diff = np.abs(onnx_det[:4] - trt_det[:4]).max()
                                    conf_diff = abs(onnx_det[4] - trt_det[4])
                                    logging.info(f"    差异: 坐标最大差异={coord_diff:.3f}, 置信度差异={conf_diff:.6f}")
                            
                            # 检查是否真的使用了不同的原始输出
                            logging.info(f"图片 {iteration_idx + 1} 原始输出形状: ONNX={onnx_output.shape}, TRT={trt_output.shape}")
                            if np.array_equal(onnx_output, trt_output):
                                logging.warning(f"图片 {iteration_idx + 1} ⚠️ 警告: ONNX和TensorRT的原始输出完全相同！")
                            else:
                                logging.info(f"图片 {iteration_idx + 1} ✅ 原始输出确实不同，均值差异: {np.abs(onnx_output.mean() - trt_output.mean()):.6f}")
                        else:
                            trt_result_img = test_img.copy()
                            cv2.putText(trt_result_img, f"TensorRT: No Detection (Image {iteration_idx + 1})", (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            logging.info(f"图片 {iteration_idx + 1} TensorRT未检测到目标")
                    except Exception as e:
                        logging.warning(f"图片 {iteration_idx + 1} TensorRT后处理失败: {e}")
                        trt_result_img = test_img.copy()
                        cv2.putText(trt_result_img, f"TensorRT: Processing Failed (Image {iteration_idx + 1})", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # 没有TensorRT结果
                    trt_result_img = test_img.copy()
                    cv2.putText(trt_result_img, f"TensorRT: No Result (Image {iteration_idx + 1})", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                
                # 创建当前图片的对比图像
                h, w = test_img.shape[:2]
                comparison_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
                
                if onnx_result_img is not None:
                    comparison_img[:, :w] = onnx_result_img
                else:
                    comparison_img[:, :w] = test_img
                    
                if trt_result_img is not None:
                    comparison_img[:, w:] = trt_result_img
                else:
                    comparison_img[:, w:] = test_img
                
                all_comparison_images.append(comparison_img)
                
                # 保存单张图片的比较结果
                single_output_path = f"{output_dir}/compare_result_image_{iteration_idx + 1:03d}.jpg"
                cv2.imwrite(single_output_path, comparison_img)
                logging.info(f"图片 {iteration_idx + 1} 比较结果已保存: {single_output_path}")
                
                # 保存ONNX单独结果
                if onnx_result_img is not None:
                    onnx_output_path = f"{output_dir}/onnx_result_image_{iteration_idx + 1:03d}.jpg"
                    cv2.imwrite(onnx_output_path, onnx_result_img)
                    logging.info(f"图片 {iteration_idx + 1} ONNX单独结果已保存: {onnx_output_path}")
                
                # 保存TensorRT单独结果
                if trt_result_img is not None and trt_iterations:
                    trt_output_path = f"{output_dir}/trt_result_image_{iteration_idx + 1:03d}.jpg"
                    cv2.imwrite(trt_output_path, trt_result_img)
                    logging.info(f"图片 {iteration_idx + 1} TensorRT单独结果已保存: {trt_output_path}")
            
            # 创建综合对比图像（如果有多张图片）
            if len(all_comparison_images) > 1:
                # 创建网格布局显示所有图片
                grid_cols = min(2, len(all_comparison_images))
                grid_rows = (len(all_comparison_images) + grid_cols - 1) // grid_cols
                
                single_h, single_w = all_comparison_images[0].shape[:2]
                grid_img = np.zeros((grid_rows * single_h, grid_cols * single_w, 3), dtype=np.uint8)
                
                for idx, comp_img in enumerate(all_comparison_images):
                    row = idx // grid_cols
                    col = idx % grid_cols
                    y_start = row * single_h
                    y_end = y_start + single_h
                    x_start = col * single_w
                    x_end = x_start + single_w
                    grid_img[y_start:y_end, x_start:x_end] = comp_img
                
                grid_output_path = f"{output_dir}/all_compare_results_grid.jpg"
                cv2.imwrite(grid_output_path, grid_img)
                logging.info(f"所有图片的网格对比结果已保存: {grid_output_path}")
                
            elif len(all_comparison_images) == 1:
                # 只有一张图片，保存为主结果
                main_output_path = f"{output_dir}/compare_results.jpg"
                cv2.imwrite(main_output_path, all_comparison_images[0])
                logging.info(f"比较结果已保存: {main_output_path}")
        else:
            logging.warning("未找到ONNX运行器结果或检测器为空")
            
    except Exception as e:
        logging.error(f"可视化结果失败: {e}")
        import traceback
        traceback.print_exc()

def test_rtdetr_compare_engine(args):
    """测试RT-DETR的compare_engine方法，支持多张图片"""
    
    # 智能路径处理：支持带后缀的单路径和双路径模式
    def determine_paths(model_path_str, engine_path_str=None):
        """智能确定ONNX和Engine文件路径"""
        model_path = Path(model_path_str)
        
        # 如果用户指定了engine路径，直接使用
        if engine_path_str:
            return str(model_path), str(Path(engine_path_str))
        
        # 检查model_path是否有后缀
        if model_path.suffix:
            if model_path.suffix.lower() == '.onnx':
                # 如果是.onnx文件，生成对应的.engine文件
                onnx_path = str(model_path)
                engine_path = str(model_path.with_suffix('.engine'))
            elif model_path.suffix.lower() == '.engine':
                # 如果是.engine文件，生成对应的.onnx文件
                engine_path = str(model_path)
                onnx_path = str(model_path.with_suffix('.onnx'))
            else:
                # 其他后缀，保持原有逻辑
                onnx_path = str(model_path.with_suffix('.onnx'))
                engine_path = str(model_path.with_suffix('.engine'))
        else:
            # 没有后缀，保持原有逻辑
            onnx_path = str(model_path.with_suffix('.onnx'))
            engine_path = str(model_path.with_suffix('.engine'))
        
        return onnx_path, engine_path
    
    onnx_path, engine_path = determine_paths(args.model_path, args.engine_path)
    
    # 图片数据路径
    test_image_paths = [args.data_path] if args.data_path else None
    
    logging.info(f"使用模型路径: {args.model_path}")
    logging.info(f"ONNX文件: {onnx_path}")
    logging.info(f"Engine文件: {engine_path}")
    logging.info(f"图片数据路径: {test_image_paths}")
    
    # 检查ONNX模型文件是否存在
    if not Path(onnx_path).exists():
        logging.error(f"ONNX模型文件不存在: {onnx_path}")
        return False
        
    # TensorRT引擎文件可以不存在，会自动构建
    if not Path(engine_path).exists():
        logging.info(f"TensorRT引擎文件不存在: {engine_path}，将从 ONNX 模型构建")
    
    # 检查测试图像文件夹是否存在
    if test_image_paths:
        test_folder_exists = False
        for path_str in test_image_paths:
            if Path(path_str).exists():
                test_folder_exists = True
                break
        
        if not test_folder_exists:
            logging.warning(f"指定的图片路径不存在: {test_image_paths}")
            test_image_paths = None
    
    if not test_image_paths:
        # 创建默认测试图片
        default_test_path = "data/sample.jpg"
        os.makedirs(os.path.dirname(default_test_path), exist_ok=True)
        test_img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.rectangle(test_img, (100, 100), (300, 200), (0, 255, 0), 2)
        cv2.rectangle(test_img, (400, 250), (550, 350), (255, 0, 0), 2)
        cv2.imwrite(default_test_path, test_img)
        test_image_paths = [default_test_path]
        logging.info(f"创建默认测试图片: {default_test_path}")
    
    try:
        # 创建检测器（使用参数中的配置）
        logging.info(f"创建{args.model_type.upper()}检测器...")
        detector = create_detector(
            model_type=args.model_type,
            onnx_path=onnx_path,
            conf_thres=args.conf_thres
        )
        
        logging.info(f"成功创建检测器: {type(detector).__name__}")
        logging.info(f"检测器实际输入形状: {detector.input_shape}")
        detector.create_engine_dataloader(image_paths=test_image_paths)
        
        # 确保数据加载器已创建
        if detector.engine_dataloader is None:
            logging.error("数据加载器未成功创建")
            return False
        # 加载测试图像（使用修改后的engine_dataloader的路径处理逻辑）
        from infer_onnx.engine_dataloader import CustomEngineDataLoader
        temp_loader = CustomEngineDataLoader(
            detector_class=type(detector),
            input_shape=detector.input_shape,
            input_name=detector.input_name,
            image_paths=test_image_paths,
            iterations=1
        )
        
        # 获取处理后的图片路径
        processed_image_paths = temp_loader.image_paths
        
        if not processed_image_paths:
            logging.error("未找到任何有效的测试图像")
            return False
        
        logging.info(f"找到 {len(processed_image_paths)} 张测试图片")
        
        # 加载所有图片
        test_images = []
        for img_path in processed_image_paths:
            img = cv2.imread(str(img_path))
            if img is not None:
                test_images.append(img)
                logging.info(f"加载图片: {img_path}")
            else:
                logging.warning(f"无法加载图片: {img_path}")
        
        if not test_images:
            logging.error("未成功加载任何测试图片")
            return False
        
        # 获取检测器的类别名称，如果没有则加载配置文件
        if detector.class_names and len(detector.class_names) > 0:
            # 从检测器获取类别名称字典，转换为列表格式
            max_class_id = max(detector.class_names.keys())
            class_names = [detector.class_names.get(i, f"class_{i}") for i in range(max_class_id + 1)]
            logging.info(f"从检测器获取类别名称: {detector.class_names}")
        else:
            # 回退到配置文件
            class_names, _ = load_colors_and_class_names()
            logging.info("从配置文件加载类别名称")
        
        # 加载颜色配置
        _, colors = load_colors_and_class_names()
        
        # 使用compare_engine获取原始结果（传入图片路径列表）
        logging.info("开始比较ONNX模型和TensorRT引擎...")
        logging.info(f"ONNX模型: {onnx_path}")
        logging.info(f"将使用 {len(processed_image_paths)} 张图片进行比较")
        
        # 先尝试使用现有的TensorRT引擎
        try:
            logging.info(f"尝试使用现有TensorRT引擎: {engine_path}")
            accuracy_match, run_results = detector.compare_engine(
                engine_path=engine_path,
                save_engine=False,
                rtol=args.rtol,
                atol=args.atol
            )
        except Exception as e:
            logging.warning(f"现有引擎文件无效: {e}")
            logging.info("将从ONNX模型重新构建TensorRT引擎...")
            
            # 从 ONNX构建新引擎并保存
            accuracy_match, run_results = detector.compare_engine(
                engine_path=None,     # 不指定引擎路径，从 ONNX构建
                save_engine=args.save_engine,  # 使用参数中的保存设置
                rtol=args.rtol,       # 使用参数中的相对容差
                atol=args.atol        # 使用参数中的绝对容差
            )
        
        # 获取compare_engine的结果进行后处理和画图
        if run_results:
            logging.info("对比较结果进行后处理和可视化...")
            logging.info(f"run_results 键: {list(run_results.keys())}")
            visualize_compare_results(test_images, run_results, class_names, colors, detector, engine_path)
        else:
            logging.info("使用检测器进行单独推理和可视化...")
            for idx, test_img in enumerate(test_images):
                detections, _ = detector(test_img)
                if detections and len(detections[0]) > 0:
                    result_img = draw_detections(test_img.copy(), detections, class_names, colors)
                    
                    # 保存结果 - 使用引擎文件名
                    engine_name = Path(engine_path).stem
                    output_dir = f"{RUN}/{engine_name}"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = f"{output_dir}/test_detection_result_{idx + 1:03d}.jpg"
                    cv2.imwrite(output_path, result_img)
                    logging.info(f"图片 {idx + 1} 检测结果已保存: {output_path}")
                    
                else:
                    logging.info(f"图片 {idx + 1} 未检测到任何目标")
        
        # 输出结果
        if accuracy_match:
            logging.info("✅ 精度比较通过！ONNX模型和TensorRT引擎的推理结果在指定容差范围内匹配。")
        else:
            logging.warning("⚠️  精度比较未通过！ONNX模型和TensorRT引擎的推理结果存在差异。")
        
        logging.info("数据加载器改进完成，现在支持文件夹和具体图片路径")
        return accuracy_match
        
    except Exception as e:
        logging.error(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logging.info("=" * 60)
    logging.info("ONNX vs TensorRT 引擎精度比较测试 - 多图片支持版本")
    logging.info("=" * 60)
    logging.info(f"模型类型: {args.model_type.upper()}")
    logging.info(f"模型路径: {args.model_path}")
    logging.info(f"置信度阈值: {args.conf_thres}")
    logging.info(f"容差设置: rtol={args.rtol}, atol={args.atol}")
    if args.data_path:
        logging.info(f"数据路径: {args.data_path}")
    else:
        logging.info("数据路径: 使用合成数据")
    logging.info("=" * 60)
    
    success = test_rtdetr_compare_engine(args)
    
    logging.info("=" * 60)
    if success:
        logging.info("测试完成: 成功 ✅")
        sys.exit(0)
    else:
        logging.error("测试完成: 失败 ❌")
        sys.exit(1)

if __name__ == "__main__":
    main()