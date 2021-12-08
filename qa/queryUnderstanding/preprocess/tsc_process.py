import pickle

import pandas as pd
from config import get_cfg
from qa.queryUnderstanding.querySegmentation import Segmentation, Words
from qa.tools.mongo import Mongo
from tqdm.notebook import tqdm

tqdm.pandas(desc="Data Process")

cfg = get_cfg()
mongo = Mongo(cfg, cfg.INVERTEDINDEX.DB_NAME)
seg = Segmentation(cfg)
stopwords = Words(cfg).get_stopwords
specialize = Words(cfg).get_specializewords

tmp = pd.read_table(cfg.BASE.CHAR_FILE, names=['text_a'])
char = list(set([x for y in tmp['text_a'].values.tolist() for x in y]))
ex = [
    "管理", "管里", "管理员", "服务管理", "服务器", "活动管理员", "冬季热", "官方", "维护", "系统", "系统公告",
    "审查", "巡查", "监督", "监管", "game", "master", "GAMEMASTER", "GameMaster", "GM",
    "Gm", "gm", "游戏管理员", "Client", "Server", "CS", "Cs", "cs", "cS", "KEFU",
    "kefu", "Kefu", "KeFu", "助理", "客户服务", "客服", "服务天使", "TEsT", "tESt", "test",
    "TeSt", "tEST", "Test", "测试", "辅助程序", "运营", "运营者", "运营组", "运营商", "运营长",
    "运营官", "运营人", "犬", "毒"
]
data = [
    x.strip() for x in open(
        'data/dictionary/sensitive/sensitive_words_lines.txt').readlines()
    if x not in ex and len(x) > 1
]
for x in data:
    mongo.insert_one(cfg.BASE.SENSETIVE_COLLECTION, {'word': x})

yunmuDict = {
    'a': '1',
    'o': '2',
    'e': '3',
    'i': '4',
    'u': '5',
    'v': '6',
    'ai': '7',
    'ei': '7',
    'ui': '8',
    'ao': '9',
    'ou': 'A',
    'iou': 'B',  #有：you->yiou->iou->iu
    'ie': 'C',
    've': 'D',
    'er': 'E',
    'an': 'F',
    'en': 'G',
    'in': 'H',
    'un': 'I',
    'vn': 'J',  #晕：yun->yvn->vn->ven
    'ang': 'F',
    'eng': 'G',
    'ing': 'H',
    'ong': 'K'
}
yunmuDict = pd.DataFrame(list(yunmuDict.items()), columns=['content', 'code'])
yunmuDict['class'] = 'yunmu_encode'
shengmuDict = {
    'b': '1',
    'p': '2',
    'm': '3',
    'f': '4',
    'd': '5',
    't': '6',
    'n': '7',
    'l': '7',
    'g': '8',
    'k': '9',
    'h': 'A',
    'j': 'B',
    'q': 'C',
    'x': 'D',
    'zh': 'E',
    'ch': 'F',
    'sh': 'G',
    'r': 'H',
    'z': 'E',
    'c': 'F',
    's': 'G',
    'y': 'I',
    'w': 'J',
    '0': '0'
}

shengmuDict = pd.DataFrame(list(shengmuDict.items()),
                           columns=['content', 'code'])
shengmuDict['class'] = 'shenmu_encode'
shapeDict = {
    '⿰': '1',
    '⿱': '2',
    '⿲': '3',
    '⿳': '4',
    '⿴': '5',  #左右结构、上下、左中右、上中下、全包围
    '⿵': '6',
    '⿶': '7',
    '⿷': '8',
    '⿸': '9',
    '⿹': 'A',  #上三包、下三包、左三包、左上包、右上包
    '⿺': 'B',
    '⿻': 'C',
    '0': '0'
}  #左下包、镶嵌、独体字：0

# shapeDict = pd.DataFrame(list(shapeDict.items()), columns=['content', 'code'])
# shapeDict['class'] = 'shape_encode'
strokesDict = {
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'A',
    11: 'B',
    12: 'C',
    13: 'D',
    14: 'E',
    15: 'F',
    16: 'G',
    17: 'H',
    18: 'I',
    19: 'J',
    20: 'K',
    21: 'L',
    22: 'M',
    23: 'N',
    24: 'O',
    25: 'P',
    26: 'Q',
    27: 'R',
    28: 'S',
    29: 'T',
    30: 'U',
    31: 'V',
    32: 'W',
    33: 'X',
    34: 'Y',
    35: 'Z',
    0: '0'
}

# strokesDict = {v:k for k, v in strokesDict.items()}
# strokesDict = pd.DataFrame(list(strokesDict.items()), columns=['content', 'code'])
strokesDict['class'] = 'stroke_encode'

with open('data/dictionary/sensitive/data.pkl', 'rb') as f:
    fourcorner = pickle.load(f)

fourcorner = pd.DataFrame(list(fourcorner.items()),
                          columns=['content', 'code'])
fourcorner['class'] = 'fourcorner_encode'

stroke_map = [
    x.strip().split()[1:3]
    for x in open('data/dictionary/sensitive/utf8_strokes.txt').readlines()
]
fourCorner_map = [
    x.strip().split()[1:3] for x in open(
        'data/dictionary/sensitive/unihan_structure.txt').readlines()
]
tsc_map = [
    x.strip().split()[1:3]
    for x in open('data/dictionary/sensitive/hanzi_ssc_res.txt').readlines()
]

fourCorner_map = {x[0]: x[1][0] for x in fourCorner_map}
stroke_map = {x[0]: x[1] for x in stroke_map}
tsc_map = {x[0]: x[1] for x in tsc_map}

stroke_map = pd.DataFrame(list(stroke_map.items()),
                          columns=['content', 'code'])
stroke_map['class'] = 'stroke'
fourCorner_map = pd.DataFrame(list(fourCorner_map.items()),
                              columns=['content', 'code'])
fourCorner_map['class'] = 'han_structure'

tsc_map = pd.DataFrame(list(tsc_map.items()), columns=['content', 'code'])
tsc_map['class'] = 'tsc_map'
tsc_map

hanzi = pd.merge(stroke_map.pivot(index='content',
                                  columns='class',
                                  values='code').reset_index(),
                 fourCorner_map.pivot(index='content',
                                      columns='class',
                                      values='code').reset_index(),
                 how='outer',
                 on='content')
hanzi = pd.merge(hanzi,
                 tsc_map.pivot(index='content', columns='class',
                               values='code').reset_index(),
                 how='outer',
                 on='content')
fourcorner_filter = fourcorner[fourcorner['content'].isin(
    hanzi.content.tolist())]
hanzi = pd.merge(hanzi,
                 fourcorner_filter.pivot(index='content',
                                         columns='class',
                                         values='code').reset_index(),
                 how='outer',
                 on='content')
hanzi['stroke_map'] = hanzi['stroke'].map(
    {str(k): v
     for k, v in strokesDict.items()})
hanzi['structure_map'] = hanzi['han_structure'].map(shapeDict)
hanzi = hanzi[hanzi['content'].isin(char)].reset_index(drop=True)

hanzi = pd.concat([
    yunmuDict.pivot(index='content', columns='class',
                    values='code').reset_index(),
    shengmuDict.pivot(index='content', columns='class',
                      values='code').reset_index(), hanzi
])

hanzi.progress_apply(lambda row: mongo.insert_one(
    'toneShapeCode', dict(zip(hanzi.columns, row.values))),
                     axis=1)
