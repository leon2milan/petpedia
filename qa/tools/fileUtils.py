import codecs
import numpy as np
import json
from iopath.common.file_io import PathManager as PathManagerBase
PathManager = PathManagerBase()


def load_vec(file_voc):
    vector_file = codecs.open(file_voc, 'r', encoding='utf-8')
    line = vector_file.readline()
    voc_size, vec_dim = map(int, line.split(' '))
    embedding = dict()
    line = vector_file.readline()
    while line:
        items = line.split(' ')
        item = items[0]
        vec = np.array(items[1:], dtype='float32')
        embedding[item] = vec
        line = vector_file.readline()
    return embedding


def load_idf(file_idf):
    with codecs.open(file_idf, 'r', encoding='utf-8') as f:
        idf = json.loads(f.read())
    return idf

