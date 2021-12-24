import difflib
import os
import time
from copy import deepcopy
from random import sample

import Levenshtein
import lightgbm as lgb
from config import get_cfg
from qa.matching.semantic.bert_sim import BertSim
from qa.tools import setup_logger
from qa.tools.neo import NEO4J

logger = setup_logger()


def classifier_question(question):
    person = ['谁', '作者', '人']
    location = ['哪国人', '哪里', '在哪', '地址', '位置']
    time = ['时候', '日期', '时代']
    number = ['多少', '多高', '多大', '', '']
    works = ['代表作品', '奖项', '']
    return []


def classifier_answer(answer):
    return []


class Similarity():
    def __init__(self, str1, str2):
        self.difflib_sim = difflib.SequenceMatcher(None, str1,
                                                   str2).quick_ratio()
        self.edit_sim = Levenshtein.ratio(str1, str2)
        self.jaro_sim = Levenshtein.jaro_winkler(str1, str2)


class PathSimFeature():
    def __init__(self, query, path, mention, answer):
        path_mention_similarity = Similarity(path, mention)
        path_query_similarity = Similarity(path, query)
        self.path_mention_feature = [
            path_mention_similarity.edit_sim,
            path_mention_similarity.difflib_sim,
            path_mention_similarity.jaro_sim
        ]
        self.path_query_similarity = [
            path_query_similarity.edit_sim, path_query_similarity.difflib_sim,
            path_query_similarity.jaro_sim
        ]
        self.answer = classifier_answer(answer)
        self.query = classifier_question(query)
        self.feature = self.path_query_similarity + self.path_mention_feature + self.query + self.answer


def path_rule_score(pairs_score, query):
    discrible_words = ['是什么', '是谁', '怎么样']
    score = 0
    for word in discrible_words:
        if pairs_score[0]['cypher_result']['mention'] + word == query:
            score = 1
            continue
    if score > 0:
        discribles = []
        key_value = ''
        for i in range(len(pairs_score)):
            if pairs_score[i]['cypher_result']['path'] == '描述':
                pairs_score[i]['score'][0] = score
                print(pairs_score[i]['cypher_result'])
                key_value = '描述'
            if key_value != '描述' and pairs_score[i]['cypher_result'][
                    'path'] != '标签':
                key_value = pairs_score[i]['cypher_result'][
                    'path'] + ":" + pairs_score[i]['cypher_result']['answer']
                discribles.append(key_value)
        if len(discribles) > 0:
            pairs_score.append({
                "cypher_result": {
                    "answer": '、'.join(discribles),
                    "template": '1aa',
                    "path": '详细信息',
                    'entity': pairs_score[0]['cypher_result']['entity'],
                    "mention": pairs_score[0]['cypher_result']['mention'],
                    "score": pairs_score[0]['cypher_result']['score']
                },
                "score": [score, 1, 1]
            })

    # print(pairs_score)
    return pairs_score


class PathRank(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.bs = BertSim(self.cfg, self.cfg.ENTITYLINK.PATH_MODEL)
        self.neo4j = NEO4J(self.cfg)
        self.gbm = lgb.Booster(model_file=os.path.join(
            self.cfg.ENTITYLINK.MODEL_PATH, 'lgb_model_path_rank.txt'))

    def predict_path_rank(self, query, cypher_result):
        test_x = []
        question_path_pair = QuestionPathPair(query, cypher_result)
        question_path_sims = self.bs.predict_sim(question_path_pair.pairs)
        pairs_score = question_path_pair.score(question_path_sims)
        pairs_score = path_rule_score(pairs_score, query)

        if len(pairs_score) > 3:
            pairs_score = sorted(pairs_score,
                                 key=lambda x: x['score'][0],
                                 reverse=True)
        pairs_score = pairs_score[:2]
        # path_entity_pair.extend(pairs_score)
        if self.cfg.ENTITYLINK.USE_RANK:
            for i in range(len(pairs_score)):
                res = pairs_score[i]['cypher_result']
                path = res['path']
                answer = res['answer']
                mention = res['mention']
                entity_score = res['score']
                sim_feature = PathSimFeature(query, path, mention, answer)
                path_sim = pairs_score[i]['score']
                features = sim_feature.feature
                features.extend(path_sim[:1])
                features.append(entity_score)
                test_x.append(features)
            # path_rank_score = predict_path(text_x)
            path_rank_score = self.gbm.predict(
                test_x, num_iteration=self.gbm.best_iteration)
            question_path_score_pair = QusetionPathScorePair(
                pairs_score, path_rank_score)
            candiate_path_list = sorted(
                question_path_score_pair.path_rank_score,
                key=lambda x: x['score'],
                reverse=True)
        else:
            candiate_path_list = pairs_score
        # candiate_entity_list = candiate_entity_list[:top_k]
        print(candiate_path_list[:4])
        return candiate_path_list[0] if len(candiate_path_list) > 0 else {}

    def predict_path_rank_list(self, query, cypher_result):
        test_x = []
        question_path_pair = QuestionPathPair(query, cypher_result)
        question_path_sims = self.bs.predict_sim(question_path_pair.pairs)
        pairs_score = question_path_pair.score(question_path_sims)
        pairs_score = path_rule_score(pairs_score, query)
        pairs_score = sorted(pairs_score,
                             key=lambda x: x['score'],
                             reverse=True)
        pairs_score = pairs_score[:2]
        # path_entity_pair.extend(pairs_score)
        for i in range(len(pairs_score)):
            res = pairs_score[i]['cypher_result']
            path = res['path']
            answer = res['answer']
            mention = res['mention']
            entity_score = res['score']
            sim_feature = PathSimFeature(query, path, mention, answer)
            path_sim = pairs_score[i]['score']
            features = sim_feature.feature
            features.append(path_sim)
            features.append(entity_score)
            test_x.append(features)
        # path_rank_score = predict_path(text_x)
        path_rank_score = self.gbm.predict(
            test_x, num_iteration=self.gbm.best_iteration)
        question_path_score_pair = QusetionPathScorePair(
            pairs_score, path_rank_score)
        candiate_path_list = sorted(question_path_score_pair.path_rank_score,
                                    key=lambda x: x['score'],
                                    reverse=True)
        # candiate_entity_list = candiate_entity_list[:top_k]
        return candiate_path_list


class QusetionPathScorePair():
    def __init__(self, pairs_score, score):
        assert len(pairs_score) == len(score)
        self.path_rank_score = [{
            "cypher_result":
            pairs_score[i]['cypher_result'],
            "score":
            score[i]
        } for i in range(len(score))]


class QuestionPathPair():
    def __init__(self, question, cypher_result):
        self.question = question
        # self.pairs = [[question, cypher_result[i]['path']]for i in range(len(cypher_result))]
        pairs = []
        for i in range(len(cypher_result)):
            if cypher_result[i]['template'] != '1a':
                pairs.append(
                    [question, cypher_result[i]['path'].replace('\t', '')])
                pairs.append(
                    [question, cypher_result[i]['path'].split('\t')[0]])
                pairs.append(
                    [question, cypher_result[i]['path'].split('\t')[1]])
            else:
                pairs.append([question, cypher_result[i]['path']])
                pairs.append([question, ''])
                pairs.append([question, ''])
        self.pairs = pairs
        self.cypher_result = cypher_result

    def score(self, score):
        assert len(score) == len(self.pairs)
        assert len(score) == len(self.cypher_result) * 3
        # self.pairs_score = [{"cypher_result": self.cypher_result[j], "score": score[j]} for j in range(len(score))]
        self.pairs_score = []
        for j in range(len(self.cypher_result)):
            if self.cypher_result[j]['template'] != '1a':
                self.pairs_score.append({
                    "cypher_result":
                    self.cypher_result[j],
                    "score":
                    [score[j * 3], score[j * 3 + 1], score[j * 3 + 2]]
                })
            else:
                self.pairs_score.append({
                    "cypher_result": self.cypher_result[j],
                    "score": [score[j * 3], 0.8, 0.8]
                })
        return self.pairs_score


if __name__ == "__main__":
    cfg = get_cfg()
    pr = PathRank(cfg)
    start_time = time.time()
    # template_dir = r'构建数据/ccks_kbqa/模版分类数据/train_1aa.txt'
    # entity = r'洪七公'
    # # result = get_graph_template(entity, template='1aa')
    # feature_dir = r'构建数据/ccks_kbqa/train_feature.txt'
    # generate_path_rank_corpus(feature_dir)
    query = r'香港的英文名'
    cypher_result = [
        {
            'answer': 'Hong Kong',
            'template': '1a',
            'path': '外文名称',
            'entity': '香港（中国特别行政区）',
            'mention': '香港',
            'score': 0.17222082860199425
        },
    ]

    result = pr.predict_path_rank(query, cypher_result)
    print(result)
    print("完成")
    print("时间：", str(time.time() - start_time))
