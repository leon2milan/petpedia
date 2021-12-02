# Cuda model deployment

1. run `make build_docker` to pull docker.

2. then run `convert_model -m <your model path> -n <your model name> --seq-len 16 128 128 --batch-size 1 32 32`  
    > 16 128 128 -> minimum, optimal, maximum sequence length, to help TensorRT better optimize your model  
    > 1 32 32 -> batch size, same as above

3. Launch Nvidia Triton inference server to play with both ONNX and TensorRT models:
```
docker run -it --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 --shm-size 256m \
  -v $PWD/triton_models:/models nvcr.io/nvidia/tritonserver:21.11-py3 \
  bash -c "pip install transformers sentencepiece && tritonserver --model-repository=/models"
```

4. Then you can query inference service.