# Pet Robot
## introduce
This project is used for pet domain question answering. For now, it support pet domain `segmentation`, `spell correct`, `question answering`. Also it uses `knowledge graph` which build by ourself.   
For retrieving, we use `ElasticSeach` for term retrieval, which support `fine` and `corse` two route search. And we also use `hnsw` as our vertor-based retrieval method.
For matching, we use `simCSE` which finetune on pet-domain continue trained bert. It returns similarity between two sentences, which will use as rank score.

## Start
For easy use, we use `makefile` to simiplify operation. 
- run `make run` to start service, then you can use curl way to use.  
For using `simCSE` during matching stage, you need put `pretrained_model` at your project same directory. And fellow procedure in [deploy](./doc/deploy.md) doc.
## More info
For other detail info, you can check doc directory.
- [deploy](./doc/deploy.md) for deploy deeplearning model use `onnx` + `tensorRT` + `fastertransformer`
- [interface](./doc/interface.md) for interface interacted with java
- [finetune](./doc/finetune.md) for continue train bert related cmd
- [first_runtime](./doc/first_runtime.md) for all the produce to recurrent 
- [mongo](./doc/mongo.md) for mange mongo service related cmd
- [es](./doc/es.md) for mange es service related cmd