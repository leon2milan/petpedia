import pandas as pd
from core.tools.mongo import Mongo
from core.queryUnderstanding.preprocess import clean, normal_cut_sentence
from core.queryUnderstanding.querySegmentation import Segmentation, Words
from config import get_cfg
from core.tools import flatten
from core.tools.es import ES
from tqdm import tqdm

tqdm.pandas(desc="Data Process")

cfg = get_cfg()
mongo = Mongo(cfg, cfg.INVERTEDINDEX.DB_NAME)
es = ES(cfg)
seg = Segmentation(cfg)
stopwords = Words(cfg).get_stopwords
qa = pd.read_csv(cfg.BASE.QA_DATA)[['question', 'answer']].fillna('')
qa = qa[qa['answer'] != ''].reset_index(drop=True)

qa['unrelevent'] = False
qa.iloc[20511:34295]['unrelevent'] = True
qa.iloc[44770:67108]['unrelevent'] = True
qa = qa[~qa['unrelevent']].reset_index()[['question', 'answer']]

qa['question_fine_cut'] = qa['question'].progress_apply(
    lambda x:
    [x for x in seg.cut(clean(x), is_rough=False) if x not in stopwords])
qa['question_rough_cut'] = qa['question'].progress_apply(
    lambda x:
    [x for x in seg.cut(clean(x), is_rough=True) if x not in stopwords])

qa['answer_fine_cut'] = qa['answer'].progress_apply(
    lambda x: [[x for x in line if x not in stopwords] for line in seg.cut(
        [clean(y) for y in normal_cut_sentence(x)], is_rough=False)])
qa['answer_rough_cut'] = qa['answer'].progress_apply(
    lambda x: [[x for x in line if x not in stopwords] for line in seg.cut(
        [clean(y) for y in normal_cut_sentence(x)], is_rough=True)])

mongo.clean('qa')
mongo.insert_many('qa', qa.fillna('').to_dict('record'))

es.insert_mongo('qa_v1')

data =  qa['answer'].dropna().drop_duplicates().progress_apply(lambda x: normal_cut_sentence(x)).tolist() + \
        qa['question'].unique().tolist()
data = flatten(data)

with open(cfg.BASE.CHAR_FILE, 'w') as f:
    for x in data:
        if x:
            f.write(" ".join(x))
            f.write('\n')

data = qa['question_fine_cut'].values.tolist() + [
    y for x in qa['answer_fine_cut'].values.tolist() for y in x if y
]
data = list(set([" ".join(x) for x in data if x]))
with open(cfg.BASE.FINE_WORD_FILE, 'w') as f:
    for x in tqdm(data):
        if x:
            f.write(x)
            f.write('\n')

data = qa['question_rough_cut'].values.tolist() + [
    y for x in qa['answer_rough_cut'].values.tolist() for y in x if y
]
data = list(set([" ".join(x) for x in data if x]))
with open(cfg.BASE.ROUGH_WORD_FILE, 'w') as f:
    for x in tqdm(data):
        if x:
            f.write(x)
            f.write('\n')