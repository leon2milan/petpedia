import time
import os
import numpy as np
import pandas as pd
from config import get_cfg
from qa.queryUnderstanding.querySegmentation import Words
from qa.queryUnderstanding.representation import Embedding, REPRESENTATION_REGISTRY
from qa.tools.logger import setup_logger
from qa.tools.utils import Singleton
from gensim.models import KeyedVectors, word2vec
from lmdb_embeddings.writer import LmdbEmbeddingsWriter
from lmdb_embeddings.reader import LruCachedLmdbEmbeddingsReader
from qa.queryUnderstanding.querySegmentation import Segmentation
from lmdb_embeddings.exceptions import MissingWordError
from tqdm import tqdm

tqdm.pandas(desc="word2vec")
logger = setup_logger()
__all__ = ['W2V']


@REPRESENTATION_REGISTRY.register()
@Singleton
class W2V(Embedding):
    __slot__ = [
        'cfg', 'stopwords', 'pretrained', 'model', 'word2idx', 'idx2embedding'
    ]

    def __init__(self, cfg, is_rough=False, finetune=False, gensim_way=True):
        Embedding.__init__(self, cfg)

        logger.info('Initializing word2vec instance ....')
        self.cfg = cfg

        if finetune:
            self.stopwords = Words(cfg).get_stopwords
            self.pretrained = KeyedVectors.load_word2vec_format(
                self.cfg.REPRESENTATION.WORD2VEC.PRETRAINED, binary=False)
        else:
            model_path = self.cfg.BASE.FINE_WORD2VEC if not is_rough else self.cfg.BASE.ROUGH_WORD2VEC
            if gensim_way:
                self.model = self.load(model_path, True)
            else:
                self.word2idx, self.idx2embedding = self.load(
                    model_path, False)

    def load(self, path, gensim_way=True):
        logger.info(
            "loading word2vec model gensim_way: {}...".format(gensim_way))
        if gensim_way:
            if self.cfg.REPRESENTATION.WORD2VEC.USE_LMDB:
                path = path.strip('.bin')
                return LruCachedLmdbEmbeddingsReader(path)
            else:
                return KeyedVectors.load_word2vec_format(path, binary=False)
        else:
            return self.getWordmap(path)

    def getWordmap(self, w2v_path):
        logger.info("loading word2vec model...")
        words = {}
        We = []
        f = open(w2v_path, 'r')
        lines = f.readlines()[1:]
        for (n, i) in enumerate(lines):
            i = i.split()
            j = 1
            v = []
            while j < len(i):
                v.append(float(i[j]))
                j += 1
            words[i[0]] = n
            We.append(v)
        return (words, np.array(We))

    def get_embedding_helper(self, s):
        return Embedding.wam(s, self.model,
                             self.cfg.REPRESENTATION.WORD2VEC.USE_LMDB,
                             self.cfg.REPRESENTATION.WORD2VEC.POOLING,
                             self.cfg.REPRESENTATION.WORD2VEC.EMBEDDING_SIZE)

    def save(self, model, path):
        model.wv.save_word2vec_format(path, binary=False)
        path = path.strip('.bin')
        if not os.path.exists(path):
            os.makedirs(path)
        writer = LmdbEmbeddingsWriter(
            W2V.iter_embeddings.__func__(model)).write(path)

    @staticmethod
    def iter_embeddings(gensim_model):
        for word in gensim_model.wv.vocab.keys():
            yield word, gensim_model[word]

    def load_data(self, data_path):
        data = pd.read_table(data_path, header=None,
                             names=['content']).dropna().drop_duplicates()
        sentences = data['content'].progress_apply(
            lambda x: [i for i in x.split()
                       if i not in self.stopwords]).values.tolist()
        return sentences

    def train(self, data_path, save_path):
        t1 = time.time()
        sentences = self.load_data(data_path)
        model_2 = word2vec.Word2Vec(
            size=self.cfg.REPRESENTATION.WORD2VEC.EMBEDDING_SIZE,
            window=self.cfg.REPRESENTATION.WORD2VEC.WINDOW,
            min_count=self.cfg.REPRESENTATION.WORD2VEC.MIN_COUNT,
            workers=6)
        model_2.build_vocab(sentences)
        total_examples = model_2.corpus_count
        model_2.build_vocab([list(self.pretrained.vocab.keys())], update=True)
        model_2.intersect_word2vec_format(
            self.cfg.REPRESENTATION.WORD2VEC.PRETRAINED, binary=False)
        model_2.train(sentences,
                      total_examples=total_examples,
                      epochs=model_2.iter)
        self.save(model_2, save_path)

        logger.info('-------------------------------------------')
        logger.info("Training word2vec model cost %.3f seconds...\n" %
                    (time.time() - t1))


if __name__ == '__main__':
    t1 = time.time()
    cfg = get_cfg()
    w2v = W2V(cfg, finetune=True)
    w2v.train(cfg.BASE.CHAR_FILE, cfg.BASE.CHAR_WORD2VEC)
    w2v.train(cfg.BASE.FINE_WORD_FILE, cfg.BASE.FINE_WORD2VEC)
    w2v.train(cfg.BASE.ROUGH_WORD_FILE, cfg.BASE.ROUGH_WORD2VEC)
