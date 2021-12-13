import os
import sys

from config.config import get_cfg

import os

import torch
import torch.nn as nn
from config import get_cfg
from qa.tools import setup_logger
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm
from transformers import (BertConfig, BertForSequenceClassification,
                          BertTokenizer)
from transformers.data.processors.utils import DataProcessor

logger = setup_logger()


class BertSim:
    def __init__(self, cfg, model_path) -> None:
        self.cfg = cfg

        self.max_len = self.cfg.MATCH.BERT.MAX_SEQ_LEN
        tokenizer_kwards = {
            'do_lower_case': False,
            'max_len': self.max_len,
            'vocab_file': os.path.join(model_path, 'vocab.txt')
        }
        self.tokenizer = BertTokenizer(*(), **tokenizer_kwards)
        bert_config = BertConfig.from_pretrained(model_path)
        bert_config.num_labels = self.cfg.MATCH.BERT.NUM_LABELS
        self.device = torch.device(
            "cuda" if self.cfg.REPRESENTATION.SIMCSE.DEVICE else "cpu")
        self.model = BertForSequenceClassification.from_pretrained(
            model_path, **{
                'config': bert_config
            }).to(self.device)
        self.model.eval()

    def load_and_cache_example(self, data, processor):
        label_list = processor.get_labels()
        examples = processor.get_test_examples(data)

        features = self.sim_convert_examples_to_features(examples=examples,
                                                         label_list=label_list)
        all_input_ids = torch.tensor([f.input_ids for f in features],
                                     dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features],
                                          dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features],
                                          dtype=torch.long)
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label)
        return dataset

    def sim_convert_examples_to_features(self,
                                         examples,
                                         label_list=None,
                                         pad_token=0,
                                         pad_token_segment_id=0,
                                         mask_padding_with_zero=True):
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        for (ex_index, example) in enumerate(examples):
            inputs = self.tokenizer.encode_plus(
                text=example.question,
                text_pair=example.attribute,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=
                True  # We're truncating the first sequence in priority if True
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs[
                "token_type_ids"]
            attention_mask = [1 if mask_padding_with_zero else 0
                              ] * len(input_ids)

            padding_length = self.max_len - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + (
                [0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] *
                                               padding_length)

            assert len(
                input_ids
            ) == self.max_len, "Error with input length {} vs {}".format(
                len(input_ids), self.max_len)
            assert len(
                attention_mask
            ) == self.max_len, "Error with input length {} vs {}".format(
                len(attention_mask), self.max_len)
            assert len(
                token_type_ids
            ) == self.max_len, "Error with input length {} vs {}".format(
                len(token_type_ids), self.max_len)

            # label = label_map[example.label]
            label = example.label

            if ex_index < 5:
                logger.debug("*** Example ***")
                logger.debug("guid: %s" % (example.guid))
                logger.debug("input_ids: %s" %
                            " ".join([str(x) for x in input_ids]))
                logger.debug("attention_mask: %s" %
                            " ".join([str(x) for x in attention_mask]))
                logger.debug("token_type_ids: %s" %
                            " ".join([str(x) for x in token_type_ids]))
                logger.debug("label: %s " % str(label))

            features.append(
                SimInputFeatures(input_ids, attention_mask, token_type_ids,
                                 label))
        return features

    def evaluate(self, eval_dataset):

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset,
                                     sampler=eval_sampler,
                                     batch_size=self.cfg.MATCH.BERT.BATCH_SIZE)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", self.cfg.MATCH.BERT.BATCH_SIZE)

        all_pred_label = []  # 记录所有的预测标签列表
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            token_type_ids = batch[2].to(self.device)
            label_ids = batch[3].to(self.device)
            with torch.no_grad():

                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     labels=label_ids)
                _, logits = outputs[0], outputs[1]
                logits = logits.softmax(dim=-1)

                logits = logits.tolist()
                logits_all = [logit[1] for logit in logits]

                all_pred_label.extend(logits_all)  # 记录预测的 label
        return all_pred_label

    def predict_sim(self, data):
        processor = SimProcessor()
        eval_dataset = self.load_and_cache_example(data, processor)
        pred_label = self.evaluate(eval_dataset)
        return pred_label


class SimInputExample(object):
    def __init__(self, guid, question, attribute, label=None):
        self.guid = guid
        self.question = question
        self.attribute = attribute
        self.label = label


class SimInputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


class SimProcessor(DataProcessor):
    """Processor for the FAQ problem
        modified from https://github.com/huggingface/transformers/blob/master/transformers/data/processors/glue.py#L154
    """
    def get_test_examples(self, datas):
        logger.info("*******  test  ********")
        return self._create_examples(datas)

    def get_labels(self):
        return [0, 1]

    @classmethod
    def _create_examples(cls, datas):
        examples = []
        uid = 0
        for tokens in datas:
            uid += 1
            if 2 == len(tokens):
                examples.append(
                    SimInputExample(guid=int(uid),
                                    question=tokens[0],
                                    attribute=tokens[1],
                                    label=int(0)))
        return examples


if __name__ == '__main__':
    data = [['散文《风筝》是哪一年创作的', '风筝（鲁迅著散文）'],
            ['散文《风筝》是哪一年创作的？', '风筝（孙燕姿歌曲专辑《风筝》'],
            ['古希腊诗人有哪些代表作品？', '古希腊（爱琴海-色雷斯文明圈的黄金时期）']]
    cfg = get_cfg()
    bs = BertSim(cfg, cfg.ENTITYLINK.ENTITY_MODEL)
    import time
    s = time.time()
    result = bs.predict_sim(data)
    print(result, time.time() - s)
    # path = r'sim_data_link.txt'
    # read_data(path)
    # print('完成')
