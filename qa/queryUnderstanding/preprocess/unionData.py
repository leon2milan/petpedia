import pandas as pd
from qa.tools.mongo import Mongo
from qa.queryUnderstanding.preprocess import clean, normal_cut_sentence
from qa.queryUnderstanding.querySegmentation import Segmentation, Words
from config import get_cfg
from qa.tools import flatten
from qa.tools.es import ES
from tqdm import tqdm

tqdm.pandas(desc="Data Process")

cfg = get_cfg()
mongo = Mongo(cfg, cfg.INVERTEDINDEX.DB_NAME)
es = ES(cfg)
seg = Segmentation(cfg)
stopwords = Words(cfg).get_stopwords

if cfg.BASE.FROM_FILE:
    qa = pd.read_csv(cfg.BASE.QA_DATA)[['question', 'answer']].fillna('')
    qa = qa[qa['answer'] != '']

    qa['unrelevent'] = False
    qa.loc[7001:8017, 'unrelevent'] = True
    qa.loc[8027:9519, 'unrelevent'] = True
    qa.loc[9530:11563, 'unrelevent'] = True
    qa.loc[11565:13519, 'unrelevent'] = True
    qa.loc[34119:34128, 'unrelevent'] = True
    qa.loc[40129:49027, 'unrelevent'] = True
    qa.loc[54431:80919, 'unrelevent'] = True
    qa.loc[91413:109696, 'unrelevent'] = True
    qa.loc[110844:113751, 'unrelevent'] = True
    qa.loc[113881:160111, 'unrelevent'] = True
    qa = qa[~qa['unrelevent']][['question', 'answer'
                                ]].reset_index(drop=True).reset_index()

    qa['question_fine_cut'] = qa['question'].progress_apply(
        lambda x:
        [x for x in seg.cut(clean(x), is_rough=False) if x not in stopwords])
    qa['question_rough_cut'] = qa['question'].progress_apply(
        lambda x:
        [x for x in seg.cut(clean(x), is_rough=True) if x not in stopwords])
else:
    from pymongo import MongoClient
    conn = MongoClient(
        f'mongodb://{cfg.MONGO.USER}:{cfg.MONGO.PWD}@10.2.0.55:27017/qa')

    db = conn['qa']
    qa = pd.DataFrame(list(db['qa'].find({})))
mongo.clean(cfg.self.cfg.BASE.QA_COLLECTION)
mongo.insert_many(cfg.self.cfg.BASE.QA_COLLECTION,
                  qa.fillna('').to_dict('record'))

es.insert_mongo('qa_v1')

# if cfg.BASE.FROM_FILE:
#     qa['answer_fine_cut'] = qa['answer'].progress_apply(
#         lambda x: [[x for x in line if x not in stopwords] for line in seg.cut(
#             [clean(y) for y in normal_cut_sentence(x)], is_rough=False)])
#     qa['answer_rough_cut'] = qa['answer'].progress_apply(
#         lambda x: [[x for x in line if x not in stopwords] for line in seg.cut(
#             [clean(y) for y in normal_cut_sentence(x)], is_rough=True)])

#     data =  qa['answer'].dropna().drop_duplicates().progress_apply(lambda x: normal_cut_sentence(x)).tolist() + \
#             qa['question'].unique().tolist()
#     data = flatten(data)

#     with open(cfg.BASE.CHAR_FILE, 'w') as f:
#         for x in data:
#             if x:
#                 f.write(" ".join(x))
#                 f.write('\n')

#     data = qa['question_fine_cut'].values.tolist() + [
#         y for x in qa['answer_fine_cut'].values.tolist() for y in x if y
#     ]
#     data = list(set([" ".join(x) for x in data if x]))
#     with open(cfg.BASE.FINE_WORD_FILE, 'w') as f:
#         for x in tqdm(data):
#             if x:
#                 f.write(x)
#                 f.write('\n')

#     data = qa['question_rough_cut'].values.tolist() + [
#         y for x in qa['answer_rough_cut'].values.tolist() for y in x if y
#     ]
#     data = list(set([" ".join(x) for x in data if x]))
#     with open(cfg.BASE.ROUGH_WORD_FILE, 'w') as f:
#         for x in tqdm(data):
#             if x:
#                 f.write(x)
#                 f.write('\n')