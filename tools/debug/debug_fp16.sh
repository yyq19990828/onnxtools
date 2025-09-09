 polygraphy run models/rtdetr-20250729.onnx \
    --trt \
        --trt-outputs mark all \
        --fp16 \
        --precision-constraints obey \
        --trt-network-postprocess-script tools/network_postprocess.py \
    --onnxrt \
        --onnx-outputs mark all \
