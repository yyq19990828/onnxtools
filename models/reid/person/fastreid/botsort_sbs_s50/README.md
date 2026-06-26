# BoT-SORT FastReID SBS-S50

这个目录预留给 PINTO `430_FastReID` 里导出的 BoT-SORT MOT17/MOT20 SBS-S50 ReID ONNX 模型。

发布包大约 7GB：

```bash
curl -L \
  https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/430_FastReID/resources_mot17_mot20_sbs_s50.tar.gz \
  -o resources_mot17_mot20_sbs_s50.tar.gz
```

我没有默认下载这个大包，因为 `models/reid/person/` 下已经有 OSNet / MobileNetV2 / ResNet50 这些更小的 ONNX 基线，足够先做 BoT-SORT ReID 集成测试。

来源：

- <https://github.com/PINTO0309/PINTO_model_zoo/tree/main/430_FastReID>
- <https://github.com/NirAharon/BoT-SORT>
- <https://github.com/JDAI-CV/fast-reid>
