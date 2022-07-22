import pandas as pd
from qa.tools.mongo import Mongo
from qa.queryUnderstanding.preprocess import clean
from qa.queryUnderstanding.querySegmentation import Segmentation, Words
from config import get_cfg
from qa.tools.es import ES
from tqdm import tqdm

tqdm.pandas(desc="Data Process")

cfg = get_cfg()
mongo = Mongo(cfg, cfg.BASE.QA_COLLECTION)
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
    qa = qa[~qa['unrelevent']][['question', 'answer']]

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
else:
    from pymongo import MongoClient
    conn = MongoClient(
        f'mongodb://{cfg.MONGO.USER}:{cfg.MONGO.PWD}@10.2.0.55:27017/qa')

    db = conn['qa']
    qa = pd.DataFrame(list(db['qa'].find({})))

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

# if cfg.BASE.FROM_FILE:
#     qa['answer_fine_cut'] = qa['answer'].progress_apply(
#         lambda x: [[x for x in line if x not in stopwords] for line in
#             [seg.cut(clean(y), is_rough=False) for y in normal_cut_sentence(x)]])
#     qa['answer_rough_cut'] = qa['answer'].progress_apply(
#         lambda x: [[x for x in line if x not in stopwords] for line in
#             [seg.cut(clean(y), is_rough=True) for y in normal_cut_sentence(x)]])

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