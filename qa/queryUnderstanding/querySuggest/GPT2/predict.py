import os

import torch
import torch.nn.functional as F

from transformers import (BertTokenizerFast, GPT2LMHeadModel)
from config import get_cfg
from qa.tools import setup_logger

logger = setup_logger()
PAD = '[PAD]'
pad_id = 0


def top_k_top_p_filtering(logits,
                          top_k=0,
                          top_p=0.0,
                          filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocab size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim(
    ) == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1,
                                                                  None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                        dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class GPT:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = torch.device(
            'cuda' if cfg.REPRESENTATION.SIMCSE.DEVICE else 'cpu')
        logger.info('using device:{}'.format(self.device))
        self.tokenizer = BertTokenizerFast(
            vocab_file='qa/queryUnderstanding/querySuggest/GPT2/vocab.txt',
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]")
        model_path = os.path.join(self.cfg.QUERY_SUGGESTION.GPT.SAVE_PATH,
                                  'epoch46')
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, text):

        input_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # input_ids = [self.tokenizer.cls_token_id]  # 每个input以[CLS]为开头
        # input_ids.extend(text_ids)
        # input_ids.append(self.tokenizer.sep_token_id)

        input_ids = torch.tensor(input_ids).long().to(self.device)
        input_ids = input_ids.unsqueeze(0)
        response = []  # 根据context，生成的response
        # 最多生成max_len个token
        for _ in range(self.cfg.QUERY_SUGGESTION.PREDICT_MAX_LEN):
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
            for id in set(response):
                next_token_logits[
                    id] /= self.cfg.QUERY_SUGGESTION.REPETITION_PENALTY
            next_token_logits = next_token_logits / self.cfg.QUERY_SUGGESTION.TEMPERATURE
            # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
            next_token_logits[self.tokenizer.convert_tokens_to_ids(
                '[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(
                next_token_logits,
                top_k=self.cfg.QUERY_SUGGESTION.TOPK,
                top_p=self.cfg.QUERY_SUGGESTION.TOPP)
            # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1),
                                           num_samples=1)
            if next_token == self.tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                break
            response.append(next_token.item())
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)

        text = self.tokenizer.convert_ids_to_tokens(response)
        return "".join(text)


if __name__ == '__main__':

    cfg = get_cfg()
    gpt = GPT(cfg)
    test = [
        '哈士奇老拆家怎么办', '犬瘟热', '狗发烧', '金毛', '拉布拉多不吃东西怎么办', '犬细小病毒的症状', '犬细小',
        '我和的'
    ]
    for i in test:
        print(f"source: {i}, 'predict': {gpt.predict(i)}")
