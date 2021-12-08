import torch
import math
import torch.nn.functional as F
from config.config import get_cfg
from qa.tools import setup_logger, Singleton
from transformers import BertConfig, BertModel, BertTokenizer
from qa.matching.semantic.simCSE.unsup import SimcseModel

logger = setup_logger(name='simCSE')

@Singleton
class SIMCSE:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model_path = self.cfg.REPRESENTATION.SIMCSE.PRETRAINED_MODEL
        self.device = torch.device(self.cfg.REPRESENTATION.SIMCSE.DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = SimcseModel(self.cfg).to(self.device)
        logger.info('load simcse model ...')
        self.model.load_state_dict(
            torch.load(self.cfg.REPRESENTATION.SIMCSE.SAVE_PATH))
        self.model.eval()

    def get_embedding(self, text) -> float:
        """模型评估函数 
        批量预测, 计算cos_sim, 转成numpy数组拼接起来, 一次性求spearman相关度
        """
        squeeze_flag = True if isinstance(text, str) else False
        source = self.tokenizer(
            text,
            max_length=self.cfg.REPRESENTATION.SIMCSE.MAXLEN,
            truncation=True,
            padding='max_length',
            return_tensors='pt')
        with torch.no_grad():
            if squeeze_flag:
                source_input_ids = source['input_ids'].to(self.device)
                source_attention_mask = source['attention_mask'].to(
                    self.device)
                source_token_type_ids = source['token_type_ids'].to(
                    self.device)
                source_pred = self.model(source_input_ids, source_attention_mask,
                                        source_token_type_ids)
            
            else:
                # source        [batch, 1, seq_len] -> [batch, seq_len]
                source_input_ids = source['input_ids'].squeeze(1).to(self.device)
                source_attention_mask = source['attention_mask'].squeeze(1).to(
                    self.device)
                source_token_type_ids = source['token_type_ids'].squeeze(1).to(
                    self.device)
                source_pred = self.model(source_input_ids, source_attention_mask,
                                        source_token_type_ids)
        return source_pred

    def similarity(self, str1, str2):
        if isinstance(str1, str):
            str1 = self.get_embedding(str1)
        if isinstance(str2, str):
            str2 = self.get_embedding(str2)
        sim = F.cosine_similarity(str1, str2, dim=-1).cpu().numpy()[0]
        if math.isnan(sim):
            sim = 0.0
        return sim
    
    def query_sim(self, query: str, candidate_list: list):
        query = self.get_embedding(query)
        candidate_list = self.get_embedding(candidate_list)
        return F.cosine_similarity(query, candidate_list, dim=-1).cpu().numpy().astype('float64',casting='same_kind')



if __name__ == '__main__':
    cfg = get_cfg()
    simcse = SIMCSE(cfg)
    import time
    test = [['哈士奇拆家怎么办', '猫拉肚子不吃东西'], 
            ['哈士奇拆家怎么办', '哈士奇总是乱咬东西怎么办'], 
            ['哈士奇拆家怎么办', '狗拆家怎么办'], 
            ['猫癣', '什么是猫癣'],
            ['什么是猫癣', '得了猫癣怎么办'], 
            ['猫癣', '猫癣转阴'],
            ['聪明', '智商']]

    for i in test:
        s = time.time()
        print(
            f"query: {i[0]}, candidate: {i[1]}, score: {simcse.similarity(i[0], i[1])}, time: {time.time() - s}"
        )

    test = [('哈士奇拆家吗', 0.7548817), ('哈士奇会拆家吗', 0.64450085), ('哈士奇为什么喜欢拆家', 0.5734687), ('萨摩耶拆家吗', 0.5666889), ('雪纳瑞拆家吗', 0.4354), ('为什么哈士奇喜欢拆家', 0.3191795), ('买宠物去宠物店OR家庭认养？', 0.3084151), ('柯基拆家吗', 0.24455209), ('拆家狂魔——哈士奇', 0.23657143), ('有不拆家的哈士奇吗', 0.20618913)]
    
    query = '哈士奇老拆家怎么办'
    for i in test:
        s = time.time()
        print(
            f"query: {query}, candidate: {i[0]}, score: {simcse.similarity(query, i[0])}, default_score: {i[1]}, time: {time.time() - s}"
        )
    query = '犬瘟热'
    candidate = [('犬瘟热转阴', 0.88323), ('犬瘟热咳喘', 0.78485936), ('犬瘟热怎么得的', 0.6642784), ('犬瘟热血常规', 0.6492873), ('犬瘟热尿血', 0.6390894), ('犬瘟热是什么,犬瘟热症状有哪些', 0.6343337), ('犬瘟热初期症状,犬瘟热怎么治', 0.62487304), ('犬瘟热后遗症有哪些', 0.61606777), ('犬瘟热临床症状有哪些,犬瘟热症状', 0.60905147), ('呕泻犬瘟热', 0.59413147)] 
    candidate_list = [i[0] for i in candidate]
    for i in candidate:
        s = time.time()
        print(
            f"query: {query}, candidate: {i[0]}, score: {simcse.similarity(query, i[0])}, default_score: {i[1]}, time: {time.time() - s}"
        )
    s = time.time()
    print(f'score: {simcse.query_sim(query, candidate_list)}, time: {time.time() - s}')