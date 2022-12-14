import os

from iopath.common.file_io import PathManager as PathManagerBase

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))



PathManager = PathManagerBase()

directories = [
    'logs', 'gunicorn', 'data', 'models', 'data/knowledge_graph',
    'data/segmentation', 'data/dictionary', 'data/intent/',
    'data/dictionary/specialize/', 'data/dictionary/sensitive/',
    'data/dictionary/synonym/', 'data/dictionary/segmentation/',
    'models/basic_structure/', 'models/representation/embedding/',
    'models/representation/language_model/', 'models/representation/ngram/',
    'models/representation/sif/', 'models/retrieval', 'models/correction',
    'models/intent/', 'models/gpt/', 'models/pretrained_model',
    'models/onnx_models', 'models/entity_link/'
]
for i in directories:
    path = os.path.join(ROOT, i)
    if not os.path.exists(path):
        os.mkdir(path)

from .config import CfgNode, configurable, get_cfg, global_cfg, set_global_cfg
