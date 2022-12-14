import os
from .config import CfgNode as CN
from config import ROOT
from qa.tools.utils import get_host_ip

ip = get_host_ip()
env = 'product'
if ip == '172.28.29.249':
    env = 'test'

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 2

# 基本配置
_C.BASE = CN()
_C.BASE.ROOT = ROOT
_C.BASE.START_TOKEN = "<SOS>"
_C.BASE.END_TOKEN = "<EOS>"
_C.BASE.PAD_TOKEN = "<PAD>"
_C.BASE.ENVIRONMENT = env

_C.BASE.QA_COLLECTION = 'qa'
_C.BASE.KNOWLEDGE_COLLECTION = 'DOG'
_C.BASE.SQL_COLLECTION = 'Neo4jQueryStatement'
_C.BASE.SPECIALIZE_COLLECTION = 'AliasMapTABLE'
_C.BASE.SENSETIVE_COLLECTION = 'sensetiveWord'
_C.BASE.TSC_COLLECTION = 'toneShapeCode'
_C.BASE.QUERY_DURING_GUIDE = 'querySuggest'

_C.BASE.MODEL_PATH = os.path.join(ROOT, 'models')
_C.BASE.QA_DATA = os.path.join(ROOT, 'data/knowledge_graph/all_qa.csv')

######### deprecated #########
_C.BASE.CAT_DATA = os.path.join(ROOT, 'data/knowledge_graph/chat_name.csv')
_C.BASE.DOG_DATA = os.path.join(ROOT, 'data/knowledge_graph/chien_name.csv')
_C.BASE.DISEASE_DATA = os.path.join(ROOT, 'data/knowledge_graph/disease.txt')
_C.BASE.SYMPTOM_DATA = os.path.join(ROOT, 'data/knowledge_graph/symptom.csv')

_C.BASE.DATA_PATH = os.path.join(ROOT, 'data/')
_C.BASE.CHAR_FILE = os.path.join(ROOT, 'data/segmentation/all_char.txt')
_C.BASE.FINE_WORD_FILE = os.path.join(ROOT,
                                      'data/segmentation/all_fine_word.txt')
_C.BASE.ROUGH_WORD_FILE = os.path.join(ROOT,
                                       'data/segmentation/all_rough_word.txt')

_C.BASE.FINE_WORD2VEC = os.path.join(
    ROOT, 'models/representation/embedding/fine_word2vec.bin')
_C.BASE.ROUGH_WORD2VEC = os.path.join(
    ROOT, 'models/representation/embedding/rough_word2vec.bin')
_C.BASE.CHAR_WORD2VEC = os.path.join(
    ROOT, 'models/representation/embedding/char_word2vec.bin')

_C.BASE.KEY_POS_INDEX = 'POS'
_C.BASE.KEY_TF_INDEX = 'TF'
_C.BASE.KEY_LD_INDEX = 'LD'

# 数据结构
_C.BASE.DATA_STRUCTURE = CN()

_C.BASE.DATA_STRUCTURE.TRIE = CN()
_C.BASE.DATA_STRUCTURE.TRIE.SAVE_PATH = os.path.join(
    ROOT, 'models/basic_structure/')

_C.WEB = CN()
_C.WEB.PORT = 6400
_C.WEB.HOST = ip
_C.WEB.THREADS = 3
_C.WEB.WORKER = 1
_C.WEB.DAEMON = 'true'
_C.WEB.WORK_CLASS = 'gevent'  # [sync, eventlet, gevent, tornado, gthread, gaiohttp]
_C.WEB.LOG = os.path.join(ROOT, 'logs/info.log')
_C.WEB.ERROR_LOG = os.path.join(ROOT, 'logs/error.log')
_C.WEB.PID_FILE = os.path.join(ROOT, 'gunicorn/gunicorn.pid')
_C.WEB.LOG_LEVEL = 'debug'

_C.MONGO = CN()
_C.MONGO.HOST = ip
_C.MONGO.PORT = 27017
_C.MONGO.USER = 'qa'
_C.MONGO.PWD = 'ABCabc123'

_C.ES = CN()
_C.ES.HOST = ip
_C.ES.PORT = 9200
_C.ES.USER = 'qa'
_C.ES.PWD = 'ABCabc123'

_C.TRITON = CN()
_C.TRITON.HOST = ip
_C.TRITON.PORT = 8000

_C.NEO4J = CN()
_C.NEO4J.HOST = ip
_C.NEO4J.PORT = 7687
_C.NEO4J.USER = 'neo4j'
_C.NEO4J.PWD = 'Neo4j123'
_C.NEO4J.DB = 'neo4j'

_C.DICTIONARY = CN()
_C.DICTIONARY.PATH = os.path.join(ROOT, 'data/dictionary/')
_C.DICTIONARY.SYNONYM_PATH = os.path.join(ROOT, 'data/dictionary/synonym/')
_C.DICTIONARY.SENSITIVE_PATH = os.path.join(ROOT, 'data/dictionary/sensitive/')
_C.DICTIONARY.SPECIALIZE_PATH = os.path.join(ROOT,
                                             'data/dictionary/specialize/')
_C.DICTIONARY.STOP_WORDS = os.path.join(ROOT, 'data/dictionary/stop_words.txt')
_C.DICTIONARY.CUSTOM_WORDS = os.path.join(
    ROOT, 'data/dictionary/segmentation/custom.txt')
_C.DICTIONARY.SAME_PINYIN_PATH = os.path.join(
    ROOT, 'data/correction/same_pinyin.txt')
_C.DICTIONARY.SAME_STROKE_PATH = os.path.join(
    ROOT, 'data/correction/same_stroke.txt')

# 文本表达
_C.REPRESENTATION = CN()

_C.REPRESENTATION.WORD2VEC = CN()
_C.REPRESENTATION.WORD2VEC.EMBEDDING_SIZE = 300
_C.REPRESENTATION.WORD2VEC.POOLING = 'avg'  # one of ['avg', 'sum']
_C.REPRESENTATION.WORD2VEC.MIN_COUNT = 5
_C.REPRESENTATION.WORD2VEC.WINDOW = 5
_C.REPRESENTATION.WORD2VEC.USE_LMDB = True
_C.REPRESENTATION.WORD2VEC.PRETRAINED = os.path.join(
    ROOT, 'models/pretrained_model/sgns.wiki.word')

#sif
_C.REPRESENTATION.SIF = CN()
_C.REPRESENTATION.SIF.FINE_PCA_PATH = os.path.join(
    ROOT, 'models/representation/sif/fine_pac.pkl')
_C.REPRESENTATION.SIF.FINE_WEIGHTFILE_PATH = os.path.join(
    ROOT, 'models/representation/sif/rough_freq.txt')
_C.REPRESENTATION.SIF.ROUGH_PCA_PATH = os.path.join(
    ROOT, 'models/representation/sif/rough_pac.pkl')
_C.REPRESENTATION.SIF.ROUGH_WEIGHTFILE_PATH = os.path.join(
    ROOT, 'models/representation/sif/rough_freq.txt')
_C.REPRESENTATION.SIF.A = 1e-3
_C.REPRESENTATION.SIF.RMPC = 1


_C.REPRESENTATION.KENLM = CN()
_C.REPRESENTATION.KENLM.SAVE_PATH = os.path.join(
    ROOT, 'models/representation/language_model/')
_C.REPRESENTATION.KENLM.PROJECT = "pet_ngram"
_C.REPRESENTATION.KENLM.MEMORY = "10%"  # 运行预占用内存
_C.REPRESENTATION.KENLM.MIN_COUNT = 2  # n-grams考虑的最低频率
_C.REPRESENTATION.KENLM.ORDER = 4  # n-grams的数量
_C.REPRESENTATION.KENLM.SKIP_SYMBOLS = '"<unk>"'
_C.REPRESENTATION.KENLM.KENLM_MODEL = os.path.join(ROOT, '../kenlm/build/bin/')

_C.REPRESENTATION.CUSTOM_W2V = CN()
_C.REPRESENTATION.CUSTOM_W2V.EMB_SIZE = 200
_C.REPRESENTATION.CUSTOM_W2V.SAVE_PATH = os.path.join(
    ROOT, 'models/embedding/custom_w2v.bin')

_C.REPRESENTATION.NGRAM = CN()
_C.REPRESENTATION.NGRAM.SAVE_PATH = os.path.join(
    ROOT, 'models/representation/ngram')

# self pet pretrain model
_C.REPRESENTATION.PRE_TRAIN = CN()
_C.REPRESENTATION.PRE_TRAIN.USE_WORD = False
_C.REPRESENTATION.PRE_TRAIN.MODE = 'base'  #['base', 'medium', 'small', 'tiny']

# simCSE
_C.REPRESENTATION.SIMCSE = CN()
# 基本参数
_C.REPRESENTATION.SIMCSE.TYPE = 'unsup'  #['unsup', 'sup']
_C.REPRESENTATION.SIMCSE.EPOCHS = 20
_C.REPRESENTATION.SIMCSE.BATCH_SIZE = 64
_C.REPRESENTATION.SIMCSE.LR = 1e-5
_C.REPRESENTATION.SIMCSE.DROPOUT = 0.3
_C.REPRESENTATION.SIMCSE.MAXLEN = 64
_C.REPRESENTATION.SIMCSE.POOLING = 'cls'  # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']
_C.REPRESENTATION.SIMCSE.DEVICE = 'cuda'  #['cuda', 'cpu']
_C.REPRESENTATION.SIMCSE.TRAIN_DATA = _C.BASE.CHAR_FILE
_C.REPRESENTATION.SIMCSE.EVAL_DATA = os.path.join(
    ROOT, "data/similarity/cnsd-sts-dev.txt")
# _C.REPRESENTATION.SIMCSE.TEST_DATA =

# 预训练模型目录
if _C.REPRESENTATION.PRE_TRAIN.USE_WORD:
    model = [
        x for x in os.listdir(os.path.join(ROOT, "../pretrained_model"))
        if _C.REPRESENTATION.PRE_TRAIN.MODE in x and 'word' in x
    ][0]
else:
    model = [
        x for x in os.listdir(os.path.join(ROOT, "../pretrained_model"))
        if _C.REPRESENTATION.PRE_TRAIN.MODE in x and 'word' not in x
    ][0]
_C.REPRESENTATION.SIMCSE.PRETRAINED_MODEL = os.path.join(
    ROOT, "../pretrained_model", model)

# 微调后参数存放位置
_C.REPRESENTATION.SIMCSE.SAVE_PATH = os.path.join(
    ROOT, f"models/representation/simcse_{_C.REPRESENTATION.SIMCSE.TYPE}.pt")

_C.QUERY_SUGGESTION = CN()

_C.QUERY_SUGGESTION.GPT = CN()
_C.QUERY_SUGGESTION.GPT.TRAIN_DATA = os.path.join(ROOT,
                                                  "data/similarity/train.csv")
_C.QUERY_SUGGESTION.GPT.SAVE_PATH = os.path.join(ROOT, "models/gpt/")
_C.QUERY_SUGGESTION.GPT.MAX_LEN = 150
_C.QUERY_SUGGESTION.GPT.IGNORE_INDEX = -100
_C.QUERY_SUGGESTION.GPT.EPOCH = 100
_C.QUERY_SUGGESTION.GPT.BATCH_SIZE = 32
_C.QUERY_SUGGESTION.GPT.LR = 2.6e-5
_C.QUERY_SUGGESTION.GPT.EPS = 1.0e-09
_C.QUERY_SUGGESTION.GPT.GRADIENT_ACC_STEPS = 2
_C.QUERY_SUGGESTION.GPT.MAX_GRAD_NORM = 2.0
_C.QUERY_SUGGESTION.PRETRAINED_MODEL = ''
_C.QUERY_SUGGESTION.PATIENT = 0
_C.QUERY_SUGGESTION.VAL_NUM = 8000
_C.QUERY_SUGGESTION.WARMUP_STEPS = 4000
_C.QUERY_SUGGESTION.LOG_STEPS = 4000
_C.QUERY_SUGGESTION.TEMPERATURE = 1
_C.QUERY_SUGGESTION.TOPK = 5
_C.QUERY_SUGGESTION.TOPP = 0  # 最高积累概率
_C.QUERY_SUGGESTION.REPETITION_PENALTY = 1.0
_C.QUERY_SUGGESTION.PREDICT_MAX_LEN = 25

# 同义词挖掘
_C.SYNONYM = CN()
_C.SYNONYM.PROCESS_NUM = 30
_C.SYNONYM.USE_PINYIN = True
_C.SYNONYM.PINYIN_WEIGHT = 0.2
_C.SYNONYM.TOPK = 5
_C.SYNONYM.WIN_LEN = 5
_C.SYNONYM.USE_SN_MODEL = True
_C.SYNONYM.USE_W2V_MODEL = True
_C.SYNONYM.USE_LEVEN_MODEL = True
_C.SYNONYM.USE_BAIKE_CRAWLER = True
_C.SYNONYM.INPUT_WORD = os.path.join(ROOT, 'data/dictionary/synonym/init.txt')

# 纠错
_C.CORRECTION = CN()
_C.CORRECTION.MODEL_FILE = os.path.join(ROOT, 'models/correction/')
_C.CORRECTION.THRESHOLD = 5
_C.CORRECTION.DATS_PATH = os.path.join(ROOT, 'models/correction/dats.dat')
_C.CORRECTION.BKTREE_PATH = os.path.join(ROOT, 'models/correction/bktree.pkl')

# query 标准化
_C.QUERY_NORMALIZATION = CN()
_C.QUERY_NORMALIZATION.INIT_FOR_DETECTION = True

# 检索
_C.RETRIEVAL = CN()
_C.RETRIEVAL.ES_INDEX = 'qa_v1'
_C.RETRIEVAL.LIMIT = 10
_C.RETRIEVAL.TOPK = 1
_C.RETRIEVAL.PART_MATCH_RATIO = 0.5

_C.RETRIEVAL.DO_NORMALIZE = False
_C.RETRIEVAL.DO_CORRECT = False
_C.RETRIEVAL.DO_EXPANSION = False
_C.RETRIEVAL.DO_TERM_WEIGHT = False
_C.RETRIEVAL.DO_CLOSE_WEIGHT = False
_C.RETRIEVAL.TIME_PERFORMENCE = False

_C.RETRIEVAL.HANDLE_EMOGI = False
_C.RETRIEVAL.HANDLE_TRANS = True

_C.RETRIEVAL.BEST_ROUTE = CN()
_C.RETRIEVAL.BEST_ROUTE.DO_KBQA = False
_C.RETRIEVAL.BEST_ROUTE.USE_ES = True

_C.RETRIEVAL.WELL_ROUTE = CN()
_C.RETRIEVAL.WELL_ROUTE.USE_HNSW = True
_C.RETRIEVAL.WELL_ROUTE.USE_ES = True
_C.RETRIEVAL.WELL_ROUTE.SEG_GRAINED = 'both'  # one of ['fine', 'rough', 'both']
_C.RETRIEVAL.WELL_ROUTE.WORD_RANK_THRESHOLD = 1  # greater than

_C.RETRIEVAL.PART_ROUTE = CN()
_C.RETRIEVAL.PART_ROUTE.USE_CONTENT_PROFILE = False

_C.RETRIEVAL.USE_ES = any(
    [_C.RETRIEVAL.WELL_ROUTE.USE_ES, _C.RETRIEVAL.BEST_ROUTE.USE_ES])

# HNSW 模型
_C.RETRIEVAL.HNSW = CN()
_C.RETRIEVAL.HNSW.SENT_EMB_METHOD = 'W2V'  # ['W2V', 'SIF']
_C.RETRIEVAL.HNSW.SPACE = 'cosine'  # ['l2', 'cosine', 'ip']
_C.RETRIEVAL.HNSW.ROUGH_HNSW_PATH = os.path.join(
    ROOT, 'models/retrieval/{}_rough_hnsw.bin').format(
        _C.RETRIEVAL.HNSW.SENT_EMB_METHOD)
_C.RETRIEVAL.HNSW.FINE_HNSW_PATH = os.path.join(
    ROOT, 'models/retrieval/{}_fine_hnsw.bin').format(
        _C.RETRIEVAL.HNSW.SENT_EMB_METHOD)
_C.RETRIEVAL.HNSW.KNOWLEDGE_HNSW_PATH = os.path.join(
    ROOT, 'models/retrieval/{}_knowledge_hnsw.bin').format(
        _C.RETRIEVAL.HNSW.SENT_EMB_METHOD)
_C.RETRIEVAL.HNSW.DATA_PATH = _C.BASE.QA_DATA
_C.RETRIEVAL.HNSW.FORCE_TRAIN = False
_C.RETRIEVAL.HNSW.EF_CONSTRUCTION = 3000
_C.RETRIEVAL.HNSW.EF = 50
_C.RETRIEVAL.HNSW.M = 64
_C.RETRIEVAL.HNSW.THREAD = 20
_C.RETRIEVAL.HNSW.FILTER_THRESHOLD = 5.0

_C.RETRIEVAL.TERM = CN()
_C.RETRIEVAL.TERM.K1 = 1.5
_C.RETRIEVAL.TERM.K3 = 7.0
_C.RETRIEVAL.TERM.B = 0.75

# intent
_C.INTENT = CN()
_C.INTENT.DATA_PATH = os.path.join(ROOT, 'data/intent/')
_C.INTENT.MODEL_PATH = os.path.join(ROOT, 'models/intent/')

# 匹配
_C.MATCH = CN()
# one of ['cosine', 'edit', 'jaccard', 'simcse', 'ts_ss', 'ts', 'ss'] or all
# ts ss in paper <A Hybrid Geometric Approach for Measuring Similarity Level Among Documents and Document Clustering>
# or https://arxiv.org/pdf/1911.00262.pdf or https://github.com/taki0112/Vector_Similarity
_C.MATCH.METHODS = ['edit', 'jaccard', 'cosine']  

_C.MATCH.BERT = CN()
_C.MATCH.BERT.MODEL_PATH = os.path.join(ROOT, 'models/matching/')
_C.MATCH.BERT.MAX_SEQ_LEN = 128
_C.MATCH.BERT.NUM_LABELS = 2
_C.MATCH.BERT.BATCH_SIZE = 8

# 实体链接
_C.ENTITYLINK = CN()
_C.ENTITYLINK.TOPK = 2
_C.ENTITYLINK.USE_RANK = 2
_C.ENTITYLINK.USE_KG = False
_C.ENTITYLINK.MODEL_PATH = os.path.join(ROOT, 'models/entity_link/')
_C.ENTITYLINK.ENTITY_MODEL = os.path.join(_C.MATCH.BERT.MODEL_PATH,
                                          'entity_sim')
_C.ENTITYLINK.ENTITY_NORM_MODEL = os.path.join(_C.MATCH.BERT.MODEL_PATH,
                                               'entity_norm_sim')
_C.ENTITYLINK.PATH_MODEL = os.path.join(_C.MATCH.BERT.MODEL_PATH, 'path_sim')

#KBQA
_C.KBQA = CN()
_C.KBQA.TEMPLATE = ['']

_C.CONTENTUNDERSTANDING = CN()
_C.CONTENTUNDERSTANDING.KEYWORD_FILE = os.path.join(ROOT,
                                                    'config/keyword.toml')
_C.CONTENTUNDERSTANDING.RULE_FILE = os.path.join(ROOT, 'config/rule.toml')

# deploy
_C.DEPLOY = CN()
_C.DEPLOY.SAVE_PATH = os.path.join(ROOT,
                                   'models/onnx_models/')  # triton_models
_C.DEPLOY.OUPUT = 'triton_models'
_C.DEPLOY.BATCH_SIZE = [1, 32, 32
                        ]  # "batch sizes to optimize for (min, optimal, max)"
_C.DEPLOY.SEQ_LEN = [16, 128,
                     128]  # sequence lengths to optimize for (min, opt, max)
_C.DEPLOY.WORKSPACE_SIZE = 10000  # workspace size in MiB (TensorRT)
_C.DEPLOY.VERBOSE = True
_C.DEPLOY.BACKEND = [
    "onnx", "tensorrt", "pytorch"
]  # backend to use. One of [onnx,tensorrt, pytorch] or all
_C.DEPLOY.NB_INSTANCE = 1  # # of model instances, may improve troughput
_C.DEPLOY.WARMUP = 100  # # of inferences to warm each model
_C.DEPLOY.NB_MEASURES = 1000  # # of inferences for benchmarks
_C.DEPLOY.SEED = 123
_C.DEPLOY.ATOL = 1e-1  # tolerance when comparing outputs to Pytorch ones


_C.HEALTHY = CN()
_C.HEALTHY.DATA = CN()
_C.HEALTHY.DATA.HEALTHY = os.path.join(ROOT, 'data/healthy/')
_C.HEALTHY.DATA.CAT = os.path.join(_C.HEALTHY.DATA.HEALTHY, '猫variety.txt')
_C.HEALTHY.DATA.DOG = os.path.join(_C.HEALTHY.DATA.HEALTHY, '犬variety.txt')
_C.HEALTHY.DATA.CAT_INFO = os.path.join(_C.HEALTHY.DATA.HEALTHY, 'cat_message.csv')
_C.HEALTHY.DATA.DOG_INFO = os.path.join(_C.HEALTHY.DATA.HEALTHY, 'dog_message.csv')
_C.HEALTHY.DATA.DISEASE = os.path.join(_C.HEALTHY.DATA.HEALTHY, 'Disease_select.xlsx')
_C.HEALTHY.DATA.CAT_DISEASE = os.path.join(_C.HEALTHY.DATA.HEALTHY, 'avgratenew猫.csv')
_C.HEALTHY.DATA.CAT_DISEASE = os.path.join(_C.HEALTHY.DATA.HEALTHY, 'avgratenew犬.csv')
_C.HEALTHY.DATA.TRAIN_RAW = os.path.join(_C.HEALTHY.DATA.HEALTHY, 'hiscasephyexam_20220622.csv')
_C.HEALTHY.DATA.DISEASE_RATE = os.path.join(_C.HEALTHY.DATA.HEALTHY, 'disease_rate.json')
_C.HEALTHY.DATA.DISEASE_COUNT = os.path.join(_C.HEALTHY.DATA.HEALTHY, 'disease_count.json')
_C.HEALTHY.DATA.DISEASE_DICT = os.path.join(_C.HEALTHY.DATA.HEALTHY, 'disease_dict.json')
_C.HEALTHY.DATA.DEL_DISEASE = os.path.join(_C.HEALTHY.DATA.HEALTHY, 'del_disease.json')
_C.HEALTHY.DATA.CAT_WEIGHT = os.path.join(_C.HEALTHY.DATA.HEALTHY, 'cat_weight.json')
_C.HEALTHY.DATA.DOG_WEIGHT = os.path.join(_C.HEALTHY.DATA.HEALTHY, 'dog_weight.json')
_C.HEALTHY.DATA.CAT_MAX_WEIGHT = os.path.join(_C.HEALTHY.DATA.HEALTHY, 'cat_max_weight.json')
_C.HEALTHY.DATA.DOG_MAX_WEIGHT = os.path.join(_C.HEALTHY.DATA.HEALTHY, 'dog_max_weight.json')
_C.HEALTHY.DATA.CAT_MIN_WEIGHT = os.path.join(_C.HEALTHY.DATA.HEALTHY, 'cat_min_weight.json')
_C.HEALTHY.DATA.DOG_MIN_WEIGHT = os.path.join(_C.HEALTHY.DATA.HEALTHY, 'dog_min_weight.json')
_C.HEALTHY.DATA.TRAIN = os.path.join(_C.HEALTHY.DATA.HEALTHY, 'train')

_C.HEALTHY.MODEL = CN()
_C.HEALTHY.MODEL.PATH = os.path.join(ROOT, 'models/healthy/')
_C.HEALTHY.MODEL.DATA_THRESHOLD = 100