import torch
import torch.nn.functional as F
from config.config import get_cfg
from qa.tools import setup_logger
from transformers import BertConfig, BertModel, BertTokenizer

logger = setup_logger()


class SIMCSE:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = self.cfg.REPRESENTATION.SIMCSE.DEVICE
        self.model_path = self.cfg.REPRESENTATION.SIMCSE.PRETRAINED_MODEL.PRETRAINED_MODEL
        config = BertConfig.from_pretrained(self.model_path)
        config.attention_probs_dropout_prob = self.cfg.REPRESENTATION.SIMCSE.DROPOUT  # 修改config的dropout系数
        config.hidden_dropout_prob = self.cfg.REPRESENTATION.SIMCSE.DROPOUT
        self.model = BertModel.from_pretrained(
            self.cfg.REPRESENTATION.SIMCSE.PRETRAINED_MODEL.SAVE_PATH,
            config=config)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)

    def get_embedding(self, text) -> float:
        """模型评估函数 
        批量预测, 计算cos_sim, 转成numpy数组拼接起来, 一次性求spearman相关度
        """
        source = self.tokenizer(
            text,
            max_length=self.cfg.REPRESENTATION.SIMCSE.MAXLEN,
            truncation=True,
            padding='max_length',
            return_tensors='pt')
        with torch.no_grad():
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
        str1 = self.get_embedding(str1)
        str2 = self.get_embedding(str2)
        return F.cosine_similarity(str1, str2, dim=-1)


if __name__ == '__main__':
    cfg = get_cfg()
    simcse = SIMCSE(cfg)
    print(simcse('哈士奇拆家怎么办', '猫拉肚子不吃东西'))
    print(simcse('猫癣', '什么是猫癣'))
    print(simcse('猫癣', '猫癣转阴'))