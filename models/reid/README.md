# ReID ONNX 模型清单

这里存放给 BoT-SORT / StrongSORT 这类跟踪器使用的第三方 ReID 外观特征模型。`*.onnx` 已被仓库 `.gitignore` 忽略，所以模型文件只保存在本机；这个 README 记录已下载模型的来源、输入输出和校验信息。

## 目录结构

```text
models/reid/
├── person/
│   ├── osnet/
│   │   └── msmt17/
│   ├── mobilenetv2/
│   │   └── msmt17/
│   ├── resnet50/
│   │   └── msmt17/
│   └── fastreid/
│       └── botsort_sbs_s50/
└── rider_vehicle/
    └── README.md
```

`person/` 只表示“以人形目标为主”的 ReID 模型。你的骑行者检测框如果是**人和车在同一个框里**，不要直接把它当成纯 person ReID 场景；这类目标后续放到 `rider_vehicle/` 更合适。

## 已下载模型

| 路径 | 来源 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| `person/osnet/msmt17/osnet_x0_25_msmt17_hf_anriha.onnx` | Hugging Face `anriha/osnet_x0_25_msmt17` | `[16,3,256,128]` `input` | `[16,512]` `output` | 很小的固定 batch OSNet x0.25，适合快速冒烟测试。 |
| `person/osnet/msmt17/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_Nx3x256x128.onnx` | PINTO model zoo `429_OSNet` | `[N,3,256,128]` `base_images` | `[N,512]` `features` | 轻量 OSNet，MSMT17 combineall，适合作为边缘端基线。 |
| `person/osnet/msmt17/osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_Nx3x256x128.onnx` | PINTO model zoo `429_OSNet` | `[N,3,256,128]` `base_images` | `[N,512]` `features` | 中等尺寸 OSNet。 |
| `person/osnet/msmt17/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_Nx3x256x128.onnx` | PINTO model zoo `429_OSNet` | `[N,3,256,128]` `base_images` | `[N,512]` `features` | 较强的 OSNet 基线；行人/骑行者 ReID 建议优先试这个。 |
| `person/osnet/msmt17/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0_Nx3x256x128.onnx` | PINTO model zoo `429_OSNet` | `[N,3,256,128]` `base_images` | `[N,512]` `features` | OSNet-AIN 变体，偏跨域鲁棒性。 |
| `person/mobilenetv2/msmt17/mobilenetv2_1dot0_msmt_Nx3x256x128.onnx` | PINTO model zoo `429_OSNet` | `[N,3,256,128]` `base_images` | `[N,1280]` `features` | 轻量非 OSNet 对照模型。 |
| `person/resnet50/msmt17/resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_Nx3x256x128.onnx` | PINTO model zoo `429_OSNet` | `[N,3,256,128]` `base_images` | `[N,2048]` `features` | 较重的 ResNet50 对照基线。 |

## 模型来源

- Hugging Face 小模型：<https://huggingface.co/anriha/osnet_x0_25_msmt17>
- PINTO OSNet ONNX 模型集合：<https://github.com/PINTO0309/PINTO_model_zoo/tree/main/429_OSNet>
- Torchreid / OSNet 原始项目：<https://github.com/KaiyangZhou/deep-person-reid>

PINTO OSNet 动态 batch 包下载地址：

```bash
https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/429_OSNet/resources_Nx3xHxW.tar.gz
```

PINTO 还提供 FastReID / BoT-SORT SBS-S50 的 ONNX 包，位于 `430_FastReID`，但发布包大约 7GB，默认没有下载。只有明确需要 MOT17/MOT20 SBS-S50 时再拉：

```bash
https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/430_FastReID/resources_mot17_mot20_sbs_s50.tar.gz
```

## SHA256 校验

```text
ec998add63fbd7536873ef78b3b7a1ca3297e12eb6c6332f8c5e9f80248cfed8  person/osnet/msmt17/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0_Nx3x256x128.onnx
588ed0906b08bab6952e44ddda82c131e24b792025a1a7d8003587d67da8546d  person/osnet/msmt17/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_Nx3x256x128.onnx
e78604f4ccda49b8f41cd0f8f7303800ce75d2361895ebb0729513c1bf53d277  person/osnet/msmt17/osnet_x0_25_msmt17_hf_anriha.onnx
aa0dc1b13989e3ad6604cdb2e989e26095c19da446ffffa6e0bef4295589f7b7  person/osnet/msmt17/osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_Nx3x256x128.onnx
65e4b18b8a2e6c6b27d3c7720a15c0d49cfe7b226b6a2250c223fbfd24e991c7  person/osnet/msmt17/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_Nx3x256x128.onnx
9fa59a23e00a42638623aae992a84ab91b56612df2f1e9f813b4a204158eefbe  person/mobilenetv2/msmt17/mobilenetv2_1dot0_msmt_Nx3x256x128.onnx
363ae4031ed4a5119224251b2421a73aa59ac325fa1f3207791976d479f327b1  person/resnet50/msmt17/resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0_Nx3x256x128.onnx
```

## 已做验证

- 所有已下载 ONNX 文件均通过 `onnx.checker.check_model()`。
- 所有已下载 ONNX 文件均通过 ONNX Runtime CPU 随机输入前向测试，输出为有限值。

接入 `BoTSORT` 时，ReID 模型需要在外部运行，再通过 `detections.data["features"]` 或 `reid_encoder(frame, xyxy)` 把 embedding 传给 tracker。

## 关于骑行者框

当前已下载模型主要是 person ReID，训练数据形态通常是竖直行人 crop，输入尺寸也多为 `[N,3,256,128]`。如果骑行者检测框包含“人 + 自行车/电动车/摩托车”整体，直接 resize 到 256x128 会明显压缩车体和背景，特征未必稳定。

建议策略：

1. 先用 `person/osnet/msmt17/osnet_x1_0_msmt17_combineall_...onnx` 做基线，观察 ID switch 是否改善。
2. 如果骑行者框偏宽、车体占比大，把这类模型单独归到 `rider_vehicle/`，不要和纯行人模型混用。
3. 最稳的路线是用本项目的骑行者 crop 做 ReID 微调或蒸馏，再导出 ONNX。
4. 如果检测器能拆出“人框”和“车框”，可以分别提 person embedding 和 vehicle/rider embedding，再在 tracker 外部拼接或加权融合。
