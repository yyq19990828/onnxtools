polygraphy run  models/rtdetr-20250729.onnx \
                --onnxrt \
                --onnx-outputs mark all \
                --trt \
                --atol 1e-3 --rtol 1e-3 \
                --trt-outputs mark all \
                > rtdetr-polygraphy.log 2>&1