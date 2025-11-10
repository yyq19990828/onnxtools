"""
统一配置管理模块

本模块集中管理所有推理模型的配置常量
配置优先级: 构造函数参数 > config.py全局配置
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml


# 检测模型配置
DET_CLASSES: Dict[int, str] = {
    0: "car",
    1: "truck",
    2: "heavy_truck",
    3: "van",
    4: "bus",
    5: "bicycle",
    6: "cyclist",
    7: "tricycle",
    8: "trolley",
    9: "pedestrain",
    10: "cone",
    11: "animal",
    12: "other",
    13: "plate",
    14: "motorcycle",
}

DET_COLORS: List[str] = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#8763DE", "#FF00FF",
]

COCO_CLASSES: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

COCO_COLORS: List[str] = [
    "#F21818", "#1857F2", "#97F218", "#F218D7", "#18F2CD", "#F28E18", "#4E18F2", "#21F218",
    "#F21861", "#18A0F2", "#E0F218", "#C418F2", "#18F284", "#F24518", "#182AF2", "#6AF218",
    "#F218AA", "#18E9F2", "#F2BB18", "#7B18F2", "#18F23B", "#F21834", "#1873F2", "#B3F218",
    "#F118F2", "#18F2B1", "#F27218", "#3218F2", "#3DF218", "#F2187C", "#18BCF2", "#F2E818",
    "#A818F2", "#18F269", "#F22918", "#1846F2", "#86F218", "#F218C5", "#18F2DF", "#F29F18",
    "#5F18F2", "#18F220", "#F2184F", "#188FF2", "#CFF218", "#D518F2", "#18F296", "#F25618",
    "#1819F2", "#59F218", "#F21898", "#18D8F2", "#F2CC18", "#8C18F2", "#18F24D", "#F21822",
    "#1862F2", "#A2F218", "#F218E1", "#18F2C3", "#F28318", "#4318F2", "#2CF218", "#F2186B",
    "#18ABF2", "#EAF218", "#B918F2", "#18F27A", "#F23A18", "#1835F2", "#75F218", "#F218B4",
    "#18F2F0", "#F2B018", "#7018F2", "#18F231", "#F2183E", "#187EF2", "#BDF218", "#E618F2",
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
# 可视化预设配置
# ============================================================================

# 5种预定义的可视化预设
VISUALIZATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "standard": {
        "name": "标准检测模式",
        "description": "默认边框+标签，适用于通用检测场景",
        "annotators": [
            {"type": "box_corner", "thickness": 2},
            {"type": "rich_label", "font_size": 25},
        ]
    },
    "lightweight": {
        "name": "简洁轻量模式",
        "description": "点标记+简单标签，最小视觉干扰",
        "annotators": [
            {"type": "dot", "radius": 5, "position": "CENTER"},
            {"type": "rich_label", "font_size": 14},
        ]
    },
    "privacy": {
        "name": "隐私保护模式",
        "description": "边框+车牌模糊，保护敏感信息",
        "annotators": [
            {"type": "box", "thickness": 2},
            {"type": "blur", "kernel_size": 15},
        ]
    },
    "debug": {
        "name": "调试分析模式",
        "description": "圆角框+置信度条+详细标签，展示所有信息",
        "annotators": [
            {"type": "round_box", "thickness": 3, "roundness": 0.3},
            {"type": "percentage_bar", "height": 16, "width": 80},
            {"type": "rich_label", "font_size": 18},
        ]
    },
    "high_contrast": {
        "name": "高对比展示模式",
        "description": "光晕效果+背景变暗，突出检测对象",
        "annotators": [
            {"type": "halo", "opacity": 0.8},
            {"type": "background_overlay", "opacity": 0.5, "color": "black"},
        ]
    },
}


# ============================================================================
# 配置加载工具函数
# ============================================================================

def load_det_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载检测模型配置

    优先级:
    1. config_path显式指定的外部YAML文件(如果提供)
    2. 硬编码的配置常量(DET_CLASSES, DET_COLORS)

    Args:
        config_path: 可选的外部配置文件路径(YAML格式)

    Returns:
        包含class_names(Dict[int, str])和visual_colors的字典

    Note:
        configs/det_config.yaml仅作为外部配置示例保留,
        默认情况下使用本模块的硬编码常量。
        class_names 格式已改为字典 {class_id: class_name}
    """
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                # 兼容处理：如果 class_names 是列表，转换为字典
                if 'class_names' in config and isinstance(config['class_names'], list):
                    config['class_names'] = {i: name for i, name in enumerate(config['class_names'])}
                return config
        else:
            raise FileNotFoundError(f"检测配置文件不存在: {config_path}")

    # 默认使用硬编码配置
    return {'class_names': DET_CLASSES, 'visual_colors': DET_COLORS}


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


def load_visualization_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载可视化预设配置

    优先级:
    1. config_path显式指定的外部YAML文件(如果提供)
    2. 硬编码的配置常量(VISUALIZATION_PRESETS)

    Args:
        config_path: 可选的外部配置文件路径(YAML格式)

    Returns:
        包含presets字典的配置

    Note:
        configs/visualization_presets.yaml仅作为外部配置示例保留,
        默认情况下使用本模块的硬编码常量。
    """
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"可视化配置文件不存在: {config_path}")

    # 默认使用硬编码配置
    return {'presets': VISUALIZATION_PRESETS}
