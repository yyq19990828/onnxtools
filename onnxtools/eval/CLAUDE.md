[根目录](../../CLAUDE.md) > [onnxtools](../CLAUDE.md) > **eval**

# 评估子模块 (onnxtools.eval)

## 模块职责

COCO 检测、OCR、分类、MOT(多目标跟踪)四类数据集评估,产出 mAP / 编辑距离 /
HOTA·MOTA·IDF1 等指标,支持置信度过滤与表格/JSON 输出。

`__init__.py` 采用 PEP 562 `__getattr__` 惰性导入:检测/OCR/分类评估器依赖 ONNX 推理栈;
`MOTEvaluator` 仅依赖 numpy + motmetrics/trackeval(`[mot]` extra),故纯 `[tracking]`/`[mot]`
安装也能 `from onnxtools.eval import MOTEvaluator`,不触发 `onnxruntime` 导入。

## 四个评估器 - 最小 API

### DetDatasetEvaluator (COCO/YOLO 检测)

```python
from onnxtools import DetDatasetEvaluator, create_detector

ev = DetDatasetEvaluator(create_detector('rtdetr', 'models/rtdetr.onnx'))
res = ev.evaluate_dataset(dataset_path, conf_threshold=0.25, iou_threshold=0.7,
                          max_images=100, exclude_files=None,
                          exclude_labels_containing=None)
# res: map50 / map50_95 / map75 / precision / recall / f1_score /
#      per_class_ap / speed_{preprocess,inference,postprocess} / total_images
```

注:`conf_threshold` 默认 0.25(非 0.001),与 Ultralytics 验证模式对齐,避免指标差异。

### OCRDatasetEvaluator

```python
from onnxtools import OCRDatasetEvaluator, OcrORT

ev = OCRDatasetEvaluator(OcrORT('models/ocr.onnx', character=char_dict))
res = ev.evaluate_dataset(label_file='data/val.txt', dataset_base_path='data/',
                          conf_threshold=0.5, max_images=None,
                          output_format='table', min_width=40)  # 'table'|'json'
# res: accuracy / normalized_edit_distance / edit_distance_similarity /
#      total_samples / evaluated_samples / filtered_samples / skipped_samples /
#      evaluation_time / avg_inference_time_ms / per_sample_results
```

`load_label_file(label_file, base)` 支持单图 `path<TAB>gt` 与 JSON 数组
`["a.jpg","b.jpg"]<TAB>gt`(自动展开)。`SampleEvaluation` 为单样本结果数据类。

### ClsDatasetEvaluator (分类)

```python
from onnxtools.eval import ClsDatasetEvaluator
res = ClsDatasetEvaluator(classifier).evaluate_dataset(dataset_path)
```

### MOTEvaluator (MOT 跟踪)

MOTChallenge 格式评估,用两套现成库出指标:**motmetrics**(CLEAR-MOT/IDF1)+
**TrackEval**(HOTA)。两库惰性导入、缺失时优雅降级(只装其一也能跑对应指标子集)。

```python
from onnxtools.eval import MOTEvaluator, run_tracker_on_gt

ev = MOTEvaluator("data/track/MOT_dataset")        # 读 seqmap/seqinfo/gt.txt

# 预测来源三选一:
#  A) 用 GT 框当理想检测,现场跑跟踪后端 → 纯关联质量(无需检测器):
preds = run_tracker_on_gt("data/track/MOT_dataset", "bytetrack_native", frame_rate=5)
#  B) 已有结果目录:preds = "runs/tracker_out"(每序列一个 <seq>.txt)
#  C) 内存 dict:{seq: {frame: ndarray(N,7)}} 或 {seq: MOTSequence}

result = ev.evaluate(preds,
    iou_threshold=0.5,                 # CLEAR/Identity 匹配阈值(HOTA 不受影响)
    metrics=("clear", "identity", "hota"),
    classes=None,                      # None=池化所有类别;[1] 只评 pedestrian
    conf_threshold=0.0)
print(result.summary_table())          # 每序列 + OVERALL 表格
result.overall["HOTA"]                 # 标量 0~100
result.to_dict()                       # 可 JSON 序列化
```

**指标语义**:百分比型(HOTA/MOTA/IDF1/...)存 `[0,100]`;`MOTP` 转为平均 IoU 重叠%
(`(1-motmetrics.motp)*100`,越大越好);计数型(IDsw/FP/FN/Frag/MT/ML/GT_IDs)为整数。

**关键实现(绕过两库的坑)**:
- IoU 距离矩阵自算后喂 `mm.MOTAccumulator`,**绕过 motmetrics 的 `np.asfarray`**
  (NumPy 2.0 已移除,本仓库 numpy≥2.2.6)。
- HOTA 直接构造 TrackEval `eval_sequence` 的 data dict(连续 0 基 id + 逐帧 IoU 相似度),
  **绕过其文件式数据集机制**,跨序列用 `combine_sequences` 聚合。
- 评估期间临时抬高 root logger 级别屏蔽 motmetrics 的 `partials: x seconds` 计时日志。
- result 文件无 class 列(第 8 列起 -1 占位),故文件来源一律池化;需分类别评估用内存 dict。

CLI:`python tools/eval/eval_mot.py --gt-root <dir> (--tracker <algo> | --predictions <dir>)`

## 模块结构

```
onnxtools/eval/
├── __init__.py     # PEP 562 惰性导出全部评估器
├── eval_coco.py    # DetDatasetEvaluator (COCO/YOLO)
├── eval_ocr.py     # OCRDatasetEvaluator + SampleEvaluation + load_label_file
├── eval_cls.py     # ClsDatasetEvaluator
├── mot_data.py     # MOTChallenge 解析 (load_gt/load_predictions/write_mot_file)
└── eval_mot.py     # MOTEvaluator / MOTResult / run_tracker_on_gt
```

## 数据集格式

| 类型 | 路径/格式 | 说明 |
|------|-----------|------|
| COCO/YOLO 目录 | `images/{test,val,train}/` + `labels/{...}/` + `classes.yaml` | 子集优先级 test>val>train(回退) |
| YOLO 标签 | `class_id xc yc w h`(归一化,每行一框) | `labels/*.txt` |
| OCR 标签 | `image_path<TAB>gt` 或 `["a.jpg","b.jpg"]<TAB>gt` | `val.txt`/`train.txt` |
| MOTChallenge GT | `gt.txt` 9 列:`frame,id,x,y,w,h,conf,class,vis` | 见 `data/track/MOT_dataset/README.md` |
| MOTChallenge result | 无 class 列(第 8 列起 -1 占位) | 文件来源池化评估 |

## 依赖与相关文件

- 核心依赖:`infer_onnx.BaseORT`、`utils.detection_metrics`(mAP)、`utils.ocr_metrics`、
  `python-levenshtein>=0.25.0`、`[mot]` extra(motmetrics/trackeval)。
- CLI 工具:`tools/eval/eval.py`(COCO)、`tools/eval/eval_ocr.py`(OCR)、`tools/eval/eval_mot.py`(MOT)。
- 配置:`configs/det_config.yaml`、`configs/plate.yaml`。

## FAQ

- **COCO conf_threshold 为何默认 0.25?** 与 Ultralytics 验证模式对齐(其会把过低阈值重置为
  0.25),用 0.001 会导致指标显著偏差。详见 `eval_coco.py` 注释。
- **如何分类别评估 MOT?** 文件来源(目录/result.txt)无 class 信息一律池化;需按类别评估时以
  内存 dict 形态传入预测。
