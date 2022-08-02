import os
import pandas as pd
from qa.tools.mongo import Mongo
from qa.queryUnderstanding.preprocess import clean, normal_cut_sentence
from config import get_cfg, ROOT
from tqdm import tqdm
from pymongo import MongoClient
from qa.tools import flatten


tqdm.pandas(desc="Data Process")

cfg = get_cfg()

mongo = Mongo(cfg, cfg.BASE.QA_COLLECTION)

try:

    from qa.tools.es import ES
    from qa.queryUnderstanding.querySegmentation import Segmentation, Words
except:

    mongo = Mongo(cfg, cfg.BASE.QA_COLLECTION)

    conn = MongoClient(
        f'mongodb://{cfg.MONGO.USER}:{cfg.MONGO.PWD}@172.28.29.249:{cfg.MONGO.PORT}/qa')

    db = conn['qa']
    need_transite = [cfg.BASE.SPECIALIZE_COLLECTION, cfg.BASE.SENSETIVE_COLLECTION,
                    cfg.BASE.TSC_COLLECTION, cfg.BASE.QUERY_DURING_GUIDE]
    for nt in need_transite:
        data = pd.DataFrame(list(db[nt].find({})))
        print('data', data)
        mongo.insert_many(nt, data.to_dict('record'))

    from qa.tools.es import ES
    from qa.queryUnderstanding.querySegmentation import Segmentation, Words


es = ES(cfg)
seg = Segmentation(cfg)
stopwords = Words(cfg).get_stopwords

if not os.path.exists(cfg.BASE.QA_DATA):
    qa = pd.read_csv(os.path.join(ROOT, 'data/knowledge_graph/qa_no_problem_data.csv'))
    print('qa size:', qa.shape)
    qa.to_csv(cfg.BASE.QA_DATA, index=False)
else:
    qa = pd.read_csv(cfg.BASE.QA_DATA)[['question', 'answer']].dropna().drop_duplicates().reset_index(drop=True)

qa['question_fine_cut'] = qa['question'].progress_apply(
    lambda x:
    [x for x in seg.cut(clean(x), is_rough=False) if x not in stopwords])
qa['question_rough_cut'] = qa['question'].progress_apply(
    lambda x:
    [x for x in seg.cut(clean(x), is_rough=True) if x not in stopwords])
qa = qa.reset_index(drop=True).reset_index()


from qa.contentUnderstanding.content_profile import ContentUnderstanding

cs = ContentUnderstanding(cfg)

qa['content_tag'] = qa.apply(lambda row: cs.understanding(row['question']),
                             axis=1)
qa = pd.concat(
    [qa.drop(['content_tag'], axis=1), qa['content_tag'].apply(pd.Series)],
    axis=1)


def format_content(x):
    if isinstance(x, list):
        return "|".join(x)
    return x


for col in qa.columns:
    if col not in ['question_rough_cut', 'question_fine_cut', 'index', '_id']:
        qa[col] = qa[col].fillna('').apply(lambda x: format_content(x))
mongo.clean(cfg.BASE.QA_COLLECTION)
mongo.insert_many(cfg.BASE.QA_COLLECTION, qa.fillna('').to_dict('record'))

qa = pd.DataFrame(list(mongo.find(cfg.BASE.QA_COLLECTION, {})))
es.insert_mongo('qa_v1', qa)


qa['answer_fine_cut'] = qa['answer'].progress_apply(
    lambda x: [[x for x in line if x not in stopwords] for line in
        [seg.cut(clean(y), is_rough=False) for y in normal_cut_sentence(x)]])
qa['answer_rough_cut'] = qa['answer'].progress_apply(
    lambda x: [[x for x in line if x not in stopwords] for line in
        [seg.cut(clean(y), is_rough=True) for y in normal_cut_sentence(x)]])

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