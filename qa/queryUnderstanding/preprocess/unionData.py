import os
import pandas as pd
from qa.tools.mongo import Mongo
from qa.queryUnderstanding.preprocess import clean
from config import get_cfg, ROOT
from tqdm import tqdm
from pymongo import MongoClient


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
    filtered = pd.read_csv(os.path.join(ROOT, 'data/knowledge_graph/filtered_qa.csv'))
    own = pd.read_csv(os.path.join(ROOT, 'data/knowledge_graph/final_qa.csv'))
    own = own.drop('question', axis=1).join(own.question.str.split('|', expand=True).stack().reset_index(drop=True, level=1).rename('question'))

    qa = pd.concat([filtered[['question', 'answer']], own[['question', 'answer']]]).dropna().drop_duplicates().reset_index(drop=True)
    print('qa size:', qa.shape)
    qa.to_csv(cfg.BASE.QA_DATA, index=False)
else:
    qa = pd.read_csv(cfg.BASE.QA_DATA)[['question', 'answer']].dropna().drop_duplicates().reset_index(drop=True)

qa = qa[qa['answer'] != '']

qa[~qa['question'].isin([
    '金毛chd症状', '狗总是便秘怎么解决', '银狐犬跟日本尖嘴有什么区别吗', '泡狗粮应该用温水还是开水【图】',
    '怎样训练狗狗捡东西送回【图】', '养狗你需要注意哪些【图】'
])]

qa['question'] = qa['question'].apply(lambda x: x.strip(
    '[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+').replace(
        '【图】', ''))

qa['answer'] = qa['answer'].apply(lambda x: x.split('标签：')[0])
qa['answer'] = qa['answer'].apply(lambda x: x.split('友情提示：')[0])
qa['answer'] = qa['answer'].apply(lambda x: x.split('文章来自：')[0])
qa['answer'] = qa['answer'].apply(lambda x: x.split('友情链接：')[0])
qa['answer'] = qa['answer'].apply(lambda x: x.split('购买宠物，请复制客服微信号')[0])
qa['answer'] = qa['answer'].apply(lambda x: x.split('建议咨询在线兽医')[0])
qa['answer'] = qa['answer'].apply(lambda x: x.split('温馨提示：')[0])
qa['answer'] = qa['answer'].apply(
    lambda x: x.replace('导读：', '').replace('导读', ''))
qa = qa[~qa['answer'].str.contains('派多格')]
qa = qa[~qa['answer'].str.contains('美容学校')]
qa = qa[~qa['answer'].str.contains('微信')]
qa = qa[~qa['answer'].str.contains('公众号')]
qa = qa[~((qa['answer'].str.startswith('2.')) |
            (qa['answer'].str.startswith('2、')) |
            (qa['answer'].str.startswith('3.')) |
            (qa['answer'].str.startswith('第二')) |
            (qa['answer'].str.startswith('二、')) |
            (qa['answer'].str.startswith('三、')))]
qa = qa[~(qa['question'].str.contains('中新网：') |
            (qa['question'].str.contains('中网资讯：')) |
            (qa['question'].str.contains('中国日报网：') |
            (qa['question'].str.contains('眉山网：')) |
            (qa['question'].str.contains('大众网报道：')) |
            (qa['question'].str.contains('培训学校')) |
            (qa['question'].str.contains('投稿指南')) |
            (qa['question'].str.contains('派多格'))))]

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