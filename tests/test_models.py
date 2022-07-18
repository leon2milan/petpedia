import os
import unittest
from pathlib import Path

from config import get_cfg


class ModelTest(unittest.TestCase):
    def setUp(self):
        cfg = get_cfg()
        self.basic_file = cfg.BASE.MODEL_PATH
        self.files = [
            'basic_structure', 'correction', 'intent', 'matching', 'retrieval',
            'representation', 'representation/embedding',
            'representation/ngram', 'representation/language_model'
        ]

    def test_predict(self):
        expect_output = {
            'models/basic_structure/entity_py_trie.pkl': 1077445,
            'models/basic_structure/word_trie.pkl': 37463396,
            'models/basic_structure/all_py_trie.pkl': 5772482,
            'models/correction/entity_py_word2py.json': 98892,
            'models/correction/all_py_py2word.json': 987520,
            'models/correction/bktree.pkl': 130590543,
            'models/correction/all_py_word2py.json': 1086406,
            'models/correction/entity_py_py2word.json': 97755,
            'models/intent/two_intent.bin': 101193394,
            'models/retrieval/W2V_fine_hnsw.bin': 362320662,
            'models/retrieval/W2V_rough_hnsw.bin': 362320662,
            'models/representation/simcse_unsup.pt': 409159223,
            'models/representation/embedding/fine_word2vec.bin': 171569932,
            'models/representation/embedding/char_word2vec.bin': 13085238,
            'models/representation/embedding/rough_word2vec.bin': 181478761,
            'models/representation/ngram/unigram.pkl': 932426,
            'models/representation/ngram/rebigram.pkl': 12193510,
            'models/representation/ngram/bigram.pkl': 12193510,
            'models/representation/language_model/pet_ngram.arpa': 411197813,
            'models/representation/language_model/pet_ngram.ngrams': 226393320,
            'models/representation/language_model/pet_ngram.klm': 279356099,
            'models/representation/language_model/pet_ngram.corpus': 342721474,
            'models/representation/language_model/pet_ngram.chars': 25071
        }
        for x in self.files:
            all_file = os.path.join(self.basic_file, x)
            for file in os.listdir(all_file):
                if not os.path.isdir(os.path.join(all_file, file)) and not os.path.join(all_file, file).endswith('txt'):
                    p = str(os.path.relpath(os.path.join(all_file, file)))
                    output = Path(os.path.join(all_file, file)).stat().st_size
                    self.assertEqual(output, expect_output[p])
