from .config import CfgNode, get_cfg, global_cfg, set_global_cfg, configurable
import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
directories = [
    'data',
    'models',
    'data/knowledge_graph',
    'data/segmentation',
    'data/dictionary',
    'data/basic_info',
    'data/intent/',
    'data/dictionary/specialize/',
    'data/dictionary/sensitive/',
    'data/dictionary/synonym/',
    'data/dictionary/segmentation/',
    'models/embedding/',
    'models/representation/',
    'models/retrieval',
    'models/correction',
    'models/intent/',
    'models/pretrained_model',
]
for i in directories:
    path = os.path.join(ROOT, i)
    if not os.path.exists(path):
        os.mkdir(path)
