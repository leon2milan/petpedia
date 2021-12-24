import itertools
from copy import deepcopy
from config.config import get_cfg
from qa.matching.semantic.bert_sim import BertSim
from qa.knowledge.graph_template import Template
from qa.knowledge.path_rank import PathRank
from qa.knowledge.entity_link import EntityLink
from qa.tools import setup_logger

logger = setup_logger()


class KBQA:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.el = EntityLink(self.cfg)
        self.temp = Template(self.cfg)
        self.pr = PathRank(self.cfg)
        self.ens = BertSim(self.cfg, self.cfg.ENTITYLINK.ENTITY_NORM_MODEL)

    def duplientity_removal(self, result):
        dupliate_entitys = result['cypher_result']['answer'].split('\t')
        if len(dupliate_entitys) != 1:
            entity_data = list(itertools.combinations(dupliate_entitys, 2))
            entity_query = []
            for data in entity_data:
                entity_query.append(deepcopy([data[0], data[1]]))
            scores = self.ens.predict_sim(entity_query)
            scores_index = [
                index if scores[index] > 0.9 else -1
                for index in range(len(scores))
            ]
            scores_index = list(set(scores_index))
            if -1 in scores_index:
                scores_index.remove(-1)
            remove_entity = [
                entity_query[index][1]
                if len(entity_query[index][0]) > len(entity_query[index][1])
                else entity_query[index][0] for index in scores_index
            ]
            for entity in remove_entity:
                if entity in dupliate_entitys:
                    dupliate_entitys.remove(entity)
        result['cypher_result']['answer'] = '\t'.join(dupliate_entitys)
        return result

    def get_answer(self, query):
        logger.info("开始抽取实体...")
        mention_list = self.el.get_mentions(query)
        if len(mention_list) == 0:
            result = {}
            return result
        logger.info("开始生成候选实体...")
        candiate_entitys = self.el.entity_link2(query, mention_list)
        logger.info("开始neo4j模版查询...")
        cypher_result = self.temp.get_graph_template(candiate_entitys)
        # print(cypher_result[:4])
        if len(cypher_result) == 0:
            result = {}
            return result
        logger.info("开始查询结果排序...")
        result = self.pr.predict_path_rank(query, cypher_result)
        logger.info("查询完成！")
        result = self.duplientity_removal(result)
        logger.info("查询结果归一化完成！")
        return result


if __name__ == '__main__':
    cfg = get_cfg()
    kbqa = KBQA(cfg)
    queries = [
        "狗乱吃东西怎么办", "边牧偶尔尿血怎么办", "猫咪经常拉肚子怎么办", "哈士奇拆家怎么办", "英短不吃东西怎么办？",
        "拉布拉多和金毛谁聪明", "折耳怀孕不吃东西怎么办？", "阿提桑诺曼底短腿犬", "阿尔卑斯达切斯勃拉克犬"
    ]
    for query in queries:
        print(query, kbqa.get_answer(query))
