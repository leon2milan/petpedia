# Cuda model deployment

1. onnx
`python deployment/convert_onnx.py -m ../pretrained_model/simCSE/ -n simCSE`
add config file in triton_models
2. tensorRT 
```
docker run -it --rm --gpus all -v $PWD/models/onnx_models/simCSE/:/models nvcr.io/nvidia/tritonserver:21.10-py3 \
    /usr/src/tensorrt/bin/trtexec \
    --onnx=/models/model.onnx \
    --best \
    --minShapes=input_ids:1x1,attention_mask:1x1,token_type_ids:1x1 \
    --optShapes=input_ids:32x128,attention_mask:32x128,token_type_ids:32x128 \
    --maxShapes=input_ids:32x128,attention_mask:32x128,token_type_ids:32x128  \
    --saveEngine="/models/model.plan" \
    --workspace=6000 \
    --useCudaGraph
```
`cp models/onnx_models/simCSE/model.plan triton_models/unsup_simcse/1/`
`cp -r ../pretrained_model/simCSE triton_models/tokenize/1`
You then need to update you `config.pbtxt`, replace all `TYPE_INT64` tensor type by `TYPE_INT32`. replace platform: "onnxruntime_onnx" by platform: "tensorrt_plan" Finally convert the numpy tensors to int32 in the tokenizer python code, like below:

input_ids = pb_utils.Tensor("INPUT_IDS", tokens['input_ids'].astype(np.int32))
attention = pb_utils.Tensor("ATTENTION", tokens['attention_mask'].astype(np.int32))

3. Launch Nvidia Triton inference server to play with both ONNX and TensorRT models:
```
docker run -it --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 256m \
  -v $PWD/triton_models:/models nvcr.io/nvidia/tritonserver:21.10-py3 \
  bash -c "pip install transformers && tritonserver --model-repository=/models"

```

4. Then you can query inference service.
`python triton_request.py`