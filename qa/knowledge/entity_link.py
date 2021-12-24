import os
import re
import time

import lightgbm as lgb
from config import get_cfg
from qa.matching.semantic.bert_sim import BertSim
from qa.queryUnderstanding.queryReformat.queryNormalization.normalize import \
    Normalization
from qa.queryUnderstanding.querySegmentation import Segmentation, Words
from qa.queryUnderstanding.representation import W2V
from qa.tools.logger import setup_logger
from qa.tools.neo import NEO4J
from qa.tools.utils import Singleton

logger = setup_logger(name='entity_linking')
__all__ = ['EntityLink']


class QuestionEntityPair():
    def __init__(self, question, entitys):
        self.question = question
        self.entitys = entitys
        self.pairs = [[question, entitys[i]] for i in range(len(entitys))]

    def score(self, score, mention):
        assert len(score) == len(self.pairs)
        self.pairs_score = [{
            "candiate_entity": self.pairs[j][1],
            "score": score[j],
            "mention": mention
        } for j in range(len(score))]
        return self.pairs_score


def caculate_entity_feature(entity):
    result_special_char = re.findall("[《（“(\"]", entity)
    result_number = re.findall("[0-9]", entity)
    result_english = re.findall("[a-zA-Z]", entity)
    char_flag = 1 if len(result_special_char) > 0 else 0
    english_flag = 1 if len(result_english) > 0 else 0
    number_flag = 1 if len(result_number) > 0 else 0
    result = [char_flag, number_flag, english_flag]
    return result


def max_common_string(s1, s2):
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    maxstr = s1
    substr_maxlen = max(len(s1), len(s2))
    for sublen in range(substr_maxlen, -1, -1):
        for i in range(substr_maxlen - sublen + 1):
            if maxstr[i:i + sublen] in s2:
                return maxstr[i:i + sublen]


question_words = [
    '谁', '什么', '哪儿', '哪里', '几时', '多少', '多大', '多高', '何时', '何地', '怎么', '怎的',
    '怎样', '怎么样', '怎么着', '如何', '为什么', '哪', '多', '何', '怎', '吗', '呢', '吧', '啊'
]


def extractor_question_words(query):
    for question_word in question_words:
        question_word_index = query.find(question_word)
        if question_word_index != -1:
            return [
                question_word_index, question_word_index + len(question_word)
            ]
    return [0, 0]


def caculate_match_feature(query, entity):
    query_length = len(query)
    entity_length = len(entity)
    entity_in_query = 0
    if entity in query:
        entity_in_query = 1
    common_string_entity = len(max_common_string(
        query, entity)) / (len(query) + len(entity))
    result = [
        query_length, entity_length, entity_in_query, common_string_entity
    ]
    return result


def caculate_match_mention_query(query, mention):
    mention_index = query.find(mention)
    mention_index_feature = [mention_index, mention_index + len(mention)]
    mention_query_feature = mention_index / len(query)
    question_word_index = extractor_question_words(query)
    mention_question_distance = question_word_index[0] - mention_index_feature[
        1] if mention_index < question_word_index[
            0] else mention_index - question_word_index[1]
    if question_word_index[0] == mention_index or question_word_index[
            1] == mention_index_feature[1]:
        mention_question_distance = 0
    mention_start_or_end = 1 if mention_index == 0 or mention_index == len(
        query) else 0
    common_string_mention = len(mention) / len(query)
    return [
        mention_query_feature, mention_question_distance,
        common_string_mention, mention_start_or_end
    ]


class SimFeture():
    def __init__(self, query, entity, mention):
        self.query = query
        self.entity = entity
        self.entity_feature = caculate_entity_feature(entity)
        self.query_entity = caculate_match_feature(query, entity)
        self.query_mention = caculate_match_mention_query(query, mention)
        self.feature = self.entity_feature + self.query_entity + self.query_mention


class CandiateEntityScorePair():
    def __init__(self, pairs_score, score):
        assert len(pairs_score) == len(score)
        self.candiate_entity_score = [{
            "candiate_entity":
            pairs_score[i]['candiate_entity'],
            "score":
            score[i],
            "mention":
            pairs_score[i]['mention']
        } for i in range(len(score))]


@Singleton
class EntityLink(object):
    __slot__ = [
        'cfg', 'w2v', 'word', 'normalization', 'seg', 'es', 'neo4j', 'gbm'
    ]

    def __init__(self, cfg):
        self.cfg = cfg
        self.w2v = W2V(self.cfg, is_rough=True)
        self.word = Words(self.cfg)
        self.normalization = Normalization(self.cfg)
        self.seg = Segmentation(self.cfg)
        if self.cfg.ENTITYLINK.USE_KG:
            self.es = BertSim(self.cfg, self.cfg.ENTITYLINK.ENTITY_MODEL)
            self.neo4j = NEO4J(self.cfg)
            self.gbm = lgb.Booster(model_file=os.path.join(
                self.cfg.ENTITYLINK.MODEL_PATH, 'lgb_model_link.txt'))

    def get_mentions(self, query):
        normalize = self.normalization.detect(query)
        normalize = [
            x for x in normalize
            if not any(x in dup and x != dup for dup in normalize)
        ]
        species = '（犬）' if len([
            self.word.get_class(y) for x in normalize
            for y in self.word.get_name(x)
        ]) > 0 else '（猫）'
        logger.debug(f"query: {query}, normalize: {normalize}")
        normalize = [
            y for x in normalize for y in self.word.get_name(x)
            if (len(re.findall(r'（.*）', y)) > 0 and re.findall(r'（.*）', y)[0]
                == species) or (len(re.findall(r'（.*）', y)) == 0)
        ]
        logger.debug(f"query: {query}, normalize1: {normalize}")
        normalize = [(x, self.word.get_class(x)) for x in normalize if x]
        logger.debug(f"query: {query}, normalize2: {normalize}")
        return list(set(normalize))

    def search_graph_feature(self, entity):
        cypher_sql = "match(n)-[r]->(m) where n.name ='{}' return r.name, keys(n)".format(
            entity)
        result_list, cypher_result = [], []
        try:
            cypher_result = self.neo4j.run(cypher_sql)
            result_list = [res['r.name'] for res in cypher_result]
        except Exception as e:
            print(e)

        return [len(set(result_list)), len(cypher_result)]

    def entity_link2(self, query, mention_candiate_entitys=None):
        mention_time = time.time()
        if mention_candiate_entitys is None:
            mention_candiate_entitys = self.get_mentions(query)
        candiate_time = time.time()
        logger.info("得到候选实体时间：{:.8f}".format(candiate_time - mention_time))
        question_entity_pair = QuestionEntityPair(query,
                                                  mention_candiate_entitys)
        question_entity_sims = self.es.predict_sim(question_entity_pair.pairs)
        pairs_score = question_entity_pair.score(question_entity_sims, mention)
        pairs_score = sorted(pairs_score,
                             key=lambda x: x['score'],
                             reverse=True)[:self.cfg.ENTITYLINK.TOPK]
        if self.cfg.ENTITYLINK.USE_RANK:
            text_x = []
            for i in range(len(pairs_score)):
                candiate_entity = pairs_score[i]['candiate_entity']

                graph_hot = self.search_graph_feature(candiate_entity)
                logger.info("graph: {:.5f}".format(time.time() -
                                                   candiate_time))

                sim = pairs_score[i]['score']
                sim_feature = SimFeture(query, candiate_entity, mention)
                features = sim_feature.feature
                features.append(sim)
                features.extend(graph_hot)
                text_x.append(features)
            candiate_entity_rank_score = self.gbm.predict(
                text_x, num_iteration=self.gbm.best_iteration)
            candiate_entity_score_pair = CandiateEntityScorePair(
                pairs_score, candiate_entity_rank_score)
            candiate_entity_list = candiate_entity_score_pair.candiate_entity_score
            logger.info("精排候选实体时间：", str(time.time() - candiate_time))
        candiate_entity_list = sorted(candiate_entity_list,
                                      key=lambda x: x['score'],
                                      reverse=True)
        candiate_entity_list = candiate_entity_list[:self.cfg.ENTITYLINK.TOPK]
        return candiate_entity_list

    def entity_link(self, query):
        # candidate = self.knowledge_hnsw.search(self.seg(query, is_rough=True))
        extracted = self.get_mentions(query)
        logger.debug(f"query: {query}, extracted: {extracted}")
        # candidate = [x if x['entity'] in extracted else x['score'] * 0.5 for x in candidate]
        # candidate = sorted(candidate, key=lambda x: x['score'])
        disease = [x for x in extracted if self.word.is_disease(x[0])]
        logger.debug(f"query: {query}, disease: {disease}")
        if len(disease) > 0:
            return disease[0]
        symptom = [x for x in extracted if self.word.is_symptom(x[0])]
        logger.debug(f"query: {query}, symptom: {symptom}")
        if len(symptom) > 0:
            return symptom[0]
        return extracted[0] if extracted else ('', None)


if __name__ == '__main__':
    cfg = get_cfg()
    mention = EntityLink(cfg)
    queries = [
        "狗乱吃东西怎么办", "边牧偶尔尿血怎么办", "猫咪经常拉肚子怎么办", "哈士奇拆家怎么办", "英短不吃东西怎么办？",
        "拉布拉多和金毛谁聪明", "折耳怀孕不吃东西怎么办？", "阿提桑诺曼底短腿犬", "阿尔卑斯达切斯勃拉克犬", "狗狗骨折了怎么办",
        "金毛一直咳嗽检查说是支气管肺炎及支气管扩张怎么治"
    ]
    for query in queries:
        print(query, mention.entity_link(query))
