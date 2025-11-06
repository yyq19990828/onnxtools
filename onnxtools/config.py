"""
统一配置管理模块

本模块集中管理所有推理模型的配置常量
配置优先级: 构造函数参数 > config.py全局配置
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml


# 检测模型配置
DET_CLASS_NAMES: List[str] = [
    "car", "truck", "heavy_truck", "van", "bus",
    "bicycle", "cyclist", "tricycle", "trolley", "pedestrain",
    "cone", "animal", "other", "plate", "motorcycle",
]

DET_VISUAL_COLORS: List[str] = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#8763DE", "#FF00FF",
]

# OCR配置
OCR_CHARACTER_DICT: List[str] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z",
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
    "学", "警", "使", "领", "港", "澳", "挂", "临", "时", "入",
    "境", "民", "航", "危", "险", "品", "应", "急",
]

COLOR_MAP: Dict[int, str] = {
    0: "black", 1: "blue", 2: "green", 3: "white", 4: "yellow",
}

LAYER_MAP: Dict[int, str] = {
    0: "single", 1: "double",
}


# ============================================================================
# 配置加载工具函数
# ============================================================================

def load_det_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载检测模型配置

    优先级:
    1. config_path显式指定的外部YAML文件(如果提供)
    2. 硬编码的配置常量(DET_CLASS_NAMES, DET_VISUAL_COLORS)

    Args:
        config_path: 可选的外部配置文件路径(YAML格式)

    Returns:
        包含class_names和visual_colors的字典

    Note:
        configs/det_config.yaml仅作为外部配置示例保留,
        默认情况下使用本模块的硬编码常量。
    """
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"检测配置文件不存在: {config_path}")

    # 默认使用硬编码配置
    return {'class_names': DET_CLASS_NAMES, 'visual_colors': DET_VISUAL_COLORS}


def load_plate_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载车牌配置

    优先级:
    1. config_path显式指定的外部YAML文件(如果提供)
    2. 硬编码的配置常量(OCR_CHARACTER_DICT, COLOR_MAP, LAYER_MAP)

    Args:
        config_path: 可选的外部配置文件路径(YAML格式)

    Returns:
        包含ocr_dict, color_dict, layer_dict的字典

    Note:
        configs/plate.yaml仅作为外部配置示例保留,
        默认情况下使用本模块的硬编码常量。
    """
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"车牌配置文件不存在: {config_path}")

    # 默认使用硬编码配置
    return {'ocr_dict': OCR_CHARACTER_DICT, 'color_dict': COLOR_MAP, 'layer_dict': LAYER_MAP}


def get_ocr_character_list(
    config_path: Optional[str] = None,
    add_blank: bool = True,
    add_space: bool = True
) -> List[str]:
    """获取OCR字符列表"""
    plate_config = load_plate_config(config_path)
    character = plate_config['ocr_dict']
    
    if add_blank:
        character = ["blank"] + character
    if add_space:
        character = character + [" "]
    
    return character
