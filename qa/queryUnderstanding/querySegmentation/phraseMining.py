import re
from collections import Counter

import pandas as pd
import thulac
from embeddings import sent_emb_sif, word_emb_elmo
from model.method import SIFRank, SIFRank_plus
from pymongo import MongoClient
from tqdm import tqdm

from config import get_cfg
from qa.queryUnderstanding.querySegmentation import Segmentation, Words
from qa.tools import flatten, substringSieve

tqdm.pandas(desc="Data Process")
conn = MongoClient(f'mongodb://qa:ABCabc123@10.2.0.55:27017/qa')
db = conn['qa']
qa = pd.DataFrame(list(db['qa'].find({})))

#download from https://github.com/HIT-SCIR/ELMoForManyLangs
model_file = r'./auxiliary_data/zhs.model/'
cfg = get_cfg()
stopwords = Words(cfg).get_stopwords
specialize = Words(cfg).get_specializewords
entity_word = flatten([[
    k, v
] for k, v in reduce(lambda a, b: dict(a, **b), specialize.values()).items()])

ELMO = word_emb_elmo.WordEmbeddings(model_file)
SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)

#download from http://thulac.thunlp.org/
zh_model = thulac.thulac(model_path=r'./auxiliary_data/models/',
                         user_dict=r'./auxiliary_data/user_dict.txt')
elmo_layers_weight = [0.5, 0.5, 0.0]

qa['keyphrases'] = qa['question'].progress_apply(lambda x: SIFRank(
    x, SIF, zh_model, N=15, elmo_layers_weight=elmo_layers_weight))

qa['keyphrases_'] = qa['question'].progress_apply(lambda x: SIFRank_plus(
    x, SIF, zh_model, N=15, elmo_layers_weight=elmo_layers_weight))

all_word = list(
    set([y for x in qa['question_rough_cut'].values.tolist() for y in x]))
df1 = sorted(Counter([
    y[0].strip().replace('\u200b', '')
    for x in qa['keyphrases_'].values.tolist() for y in x
    if y[0] not in all_word
]).items(),
             key=lambda x: x[1],
             reverse=True)
df2 = sorted(Counter([
    y[0].strip().replace('\u200b', '')
    for x in qa['keyphrases'].values.tolist() for y in x
    if y[0] not in all_word
]).items(),
             key=lambda x: x[1],
             reverse=True)

df = pd.concat([df1, df2]).groupby(['word']).sum().reset_index()
df['word'] = df['word'].apply(lambda x: re.sub('\s+', '', x))
df['flag'] = df.apply(
    lambda row: True
    if not row['word'] or row['word'] in stopwords or row['word'] in
    entity_word or row['word'].startswith('～') or row['word'].startswith('%')
    or row['word'].startswith('-') or row['word'].startswith('.') else False,
    axis=1)

phrase_mining = [
    x for x in df[~df['flag']]['word'].tolist()
    if len(re.sub('[^\u4e00-\u9fa5]+', '', x)) > 1 and not x.endswith('怎')
    and not x.startswith('么') and not x.startswith('儿')
    and not x.startswith('月') and not x.endswith('价') and not x.endswith('会')
    and not x.endswith('什') and len(x) > 1 and x not in all_word
    and not x.startswith('么') and not x.startswith('儿')
    and not x.startswith('月') and not x.endswith('价') and not x.endswith('会')
    and not x.endswith('什') and len(x) > 1 and x not in entity_word
]

entity_word = flatten([[
    k, v
] for k, v in reduce(lambda a, b: dict(a, **b), specialize.values()).items()])

phrase_mining = substringSieve(phrase_mining)

with open('phrase_mining.csv', 'w') as f:
    for i in phrase_mining:
        f.write(i + '\n')
