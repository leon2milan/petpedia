from functools import reduce
import pandas as pd
from core.tools.mongo import Mongo
from core.queryUnderstanding.preprocess import clean, normal_cut_sentence
from core.queryUnderstanding.querySegmentation import Segmentation, Words
from config import get_cfg
from core.tools import flatten

from collections import Counter
from functools import partial
from core.queryUnderstanding.representation import SIF, W2V
from tqdm.notebook import tqdm

tqdm.pandas(desc="Data Process")


cfg = get_cfg()
mongo = Mongo(cfg, cfg.INVERTEDINDEX.DB_NAME)
seg = Segmentation(cfg)
stopwords = Words(cfg).get_stopwords
specialize = Words(cfg).get_specializewords

rough = W2V(cfg, is_rough=True)

qa = pd.read_csv('data/knowledge_graph/all_qa.csv').fillna('')
qa = qa[qa['answer'] != ''].reset_index(drop=True)
breed = {**specialize['cat'], **specialize['dog']}
cat = specialize['cat']
dog = specialize['dog']
symptom = specialize['symptom']
disease = specialize['disease']


def extract_tag(s, dic):
    res = []
    for k, v in dic.items():
        if k.strip() in s:
            if k == '缅甸猫' and '欧洲缅甸猫' in s:
                break
            if k == '獒犬' and '斗牛獒犬' in s:
                break
            if k == '斗牛犬' and ('法国' in s or '英国' in s):
                break
            if k == '雪纳瑞' and '迷你雪纳瑞' in s:
                break
            if k == '可卡犬' and ('美国可卡犬' in s or '英国可卡犬' in s):
                break
            if k == '雪纳瑞' and '巨型雪纳瑞' in s:
                break
            if k == '贵宾犬' and ('茶杯' in s or '玩具' in s or '迷你' in s):
                break
            res.append(k.strip())
    if not res:
        for k, v in dic.items():
            for i in v:
                if i in s:
                    if k == '哈威那伴随犬' and ('哈瓦那棕猫' in s or '哈瓦那猫' in s):
                        break
                    elif i == '拿破仑' and '猫' in s:
                        break
                    elif i == '黑背' and '黑背白肚子的猫' in s:
                        break
                    elif i == '折耳' and ('喜乐蒂' in s or '拉布拉多' in s
                                        or '土狗' in s):
                        break
                    elif i == '布偶' and '布偶背包' in s:
                        break
                    elif i == '狼犬' and ('黑狼犬' in s or '爱尔兰大猎犬' in s):
                        break
                    elif i == '比特犬' and '惠比特犬' in s:
                        break
                    elif i == '罗威' and '罗威士梗' in s:
                        break
                    elif i == '波利' and '波利顿犬' in s:
                        break
                    elif i == '森林猫' and '西伯利亚森林猫' in s:
                        break
                    elif i == '查理王' and ('骑士查尔斯王小猎犬' in s or '骑士查理王小猎犬' in s):
                        break
                    elif i == '猎鹬犬' and ('英国玩具猎鹬犬' in s or '克伦伯猎鹬犬' in s):
                        break
                    elif i == '贵妇' and '贵妇人' in s:
                        break
                    elif i == '曼彻斯特' and '玩具曼彻斯特犬' in s:
                        break
                    elif i == '波利' and '纽波利顿' in s:
                        break
                    elif i == '泰迪' and '玩具泰迪' in s:
                        break
                    elif i == '贵宾' and ('茶杯' in s or '玩具' in s or '迷你' in s):
                        break
                    res.append(k)
    return list(set(res))


def extract_animal(s):
    res = []
    c = extract_tag(s, cat)
    d = extract_tag(s, dog)
    if c:
        res.extend('猫')
    if d:
        res.extend('犬')
    if '猫' in s:
        res.append('猫')
    if ('狗' in s and '猫中狗' not in s) or ('犬' in s and '狂犬' not in s):
        res.append('犬')
    return list(set(res))


# qa['animal'] = qa['question'].progress_apply(lambda x: "|".join(extract_animal(x)))
# qa['breed'] = qa['question'].progress_apply(lambda x: "|".join(extract_tag(x, breed)))
# qa['disease'] = qa['question'].progress_apply(lambda x: "|".join(extract_tag(x, disease)))
# qa['symptom'] = qa['question'].progress_apply(lambda x: "|".join(extract_tag(x, symptom)))
qa['unrelevent'] = False
qa.iloc[20511:34295]['unrelevent'] = True
qa.iloc[44770:67108]['unrelevent'] = True
# qa.iloc[76523: 79428]['unrelevent'] = True

qa = qa[~qa['unrelevent']].reset_index(drop=True)
qa.shape
