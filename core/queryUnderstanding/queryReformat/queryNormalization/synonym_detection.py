from core.queryUnderstanding.queryReformat.queryNormalization.synonymMining import levenshtein, baike_crawler, semantic_network, word2vec_model
from core.queryUnderstanding.querySegmentation import Segmentation, Words
from core.tools import setup_logger
from core.tools.mongo import Mongo
import sys
import pandas as pd
from config import get_cfg

logger = setup_logger()


class SynonymDetection(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.seg = Segmentation(self.cfg)
        self.stopwords = Words(self.cfg).get_stopwords
        self.mongo = Mongo(self.cfg, self.cfg.INVERTEDINDEX.DB_NAME)

    def load_input_words(self):
        input_word_code_dict = dict()
        index = len(self.word2idx)
        with open(self.cfg.SYNONYM.INPUT_WORD,
                  "r",
                  encoding='utf-8',
                  errors='ignore') as f:
            line = f.readline()
            while line:
                word = line.strip()
                if self.word2idx is not None and word not in self.word2idx:
                    self.word2idx[word] = index
                    self.idx2word[index] = word
                    index += 1
                input_word_code_dict[word] = index
                line = f.readline()
        logger.info('totally {a} words .'.format(a=len(input_word_code_dict)))
        return input_word_code_dict

    def preprocess_file(self):
        logger.info('start preprocess......')
        self.word_id_file()
        self.word_list = [k for k, v in self.word2idx.items()]
        self.idx2word = {v: k for k, v in self.word2idx.items()}

        self.input_word_code_dict = self.load_input_words()

        self.input_word_id = [
            self.word2idx[w] for w in list(self.input_word_code_dict.keys())
        ]

        logger.info(' done!!!')

    def word_id_file(self):
        entity_set = set()
        data = pd.DataFrame(list(self.mongo.find(self.cfg.BASE.QA_COLLECTION, {}))).dropna()
        self.data = data['question_rough_cut'].values.tolist() + [
            x for ans in data['answer_rough_cut'] for x in ans
        ]
        for line in self.data:
            for word in line:
                entity_set.add(word)

        self.word2idx = {'<PAD>': 0}
        cnt = 1
        for e in entity_set:
            self.word2idx[e] = cnt
            cnt += 1
        logger.info(
            "file has created，totally {a} words .".format(a=len(entity_set)))

    def run(self):
        self.preprocess_file()
        
        if self.cfg.SYNONYM.USE_BAIKE_CRAWLER:
            logger.info("staring baike crawler...")
            word_code_list = []
            for k, v in self.input_word_code_dict.items():
                word_code_list.append((k, v))
            baike_crawler.baike_synonym_detect(self.cfg.QUERY_NORMALIZATION.SYNONYM_PATH,
                                               word_code_list)

        if self.cfg.SYNONYM.USE_SN_MODEL:
            semantic_network.synonym_detect(
                data=self.data,
                input_word_id=self.input_word_id,
                input_word_code_dict=self.input_word_code_dict,
                id2word=self.idx2word,
                word2id=self.word2idx,
                top_k=self.cfg.SYNONYM.TOPK,
                win_len=self.cfg.SYNONYM.WIN_LEN,
                process_number=self.cfg.SYNONYM.PROCESS_NUM,
                base_path=self.cfg.QUERY_NORMALIZATION.SYNONYM_PATH)

        if self.cfg.SYNONYM.USE_LEVEN_MODEL:
            l_model = levenshtein.Levenshtein_model(
                input_word=list(self.input_word_code_dict.keys()),
                candidate_word=self.word_list,
                process_number=self.cfg.SYNONYM.PROCESS_NUM,
                if_use_pinyin=self.cfg.SYNONYM.USE_PINYIN,
                pinyin_weight=self.cfg.SYNONYM.PINYIN_WEIGHT,
                top_k=self.cfg.SYNONYM.TOPK,
                base_path=self.cfg.QUERY_NORMALIZATION.SYNONYM_PATH)
            l_model.multipro_synonym_detect(self.input_word_code_dict)

        if self.cfg.SYNONYM.USE_W2V_MODEL:
            word2vec_model.synonym_detect(self.input_word_code_dict,
                                          self.cfg.SYNONYM.TOPK)


if __name__ == '__main__':
    cfg = get_cfg()
    sd = SynonymDetection(cfg)
    sd.run()