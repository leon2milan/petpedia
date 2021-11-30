from collections import Counter
from tqdm import tqdm
import re, glob, os
from math import log10
import struct
import math
import kenlm
from collections import Counter
# 纠错
import pycorrector
from pypinyin import lazy_pinyin, Style
from config import get_cfg
from qa.queryUnderstanding.querySegmentation import Segmentation


class KenLM():
    def __init__(self, cfg):
        self.cfg = cfg

        self.seg = Segmentation(self.cfg)
        self.memory = self.cfg.REPRESENTATION.KENLM.MEMORY  # 运行预占用内存
        self.min_count = self.cfg.REPRESENTATION.KENLM.MIN_COUNT  # n-grams考虑的最低频率
        self.order = self.cfg.REPRESENTATION.KENLM.ORDER  # n-grams的数量
        self.kenlm_model_path = self.cfg.REPRESENTATION.KENLM.KENLM_MODEL
        project = self.cfg.REPRESENTATION.KENLM.PROJECT
        save_path = self.cfg.REPRESENTATION.KENLM.SAVE_PATH
        print('save_path', save_path)
        # kenlm模型路径 / 包括：count_ngrams/lmplz等kenlm模块的路径
        self.corpus_file = save_path + '/%s.corpus' % project  # 语料保存的文件名
        self.vocab_file = save_path + '/%s.chars' % project  # 字符集保存的文件名
        self.ngram_file = save_path + '/%s.ngrams' % project  # ngram集保存的文件名
        self.output_file = save_path + '/%s.vocab' % project  # 最后导出的词表文件名
        self.arpa_file = save_path + '/%s.arpa' % project  # 语言模型的文件名arpa
        self.klm_file = save_path + '/%s.klm' % project  # 语言模型的二进制文件名klm,也可以.bin
        self.skip_symbols = self.cfg.REPRESENTATION.KENLM.SKIP_SYMBOLS
        # lm_train训练时候，Treat <s>, </s>, and <unk> as whitespace instead of throwing an exception
        #这里的转移概率是人工总结的，总的来说，就是要降低长词的可能性。
        trans = {
            'bb': 1,
            'bc': 0.15,
            'cb': 1,
            'cd': 0.01,
            'db': 1,
            'de': 0.01,
            'eb': 1,
            'ee': 0.001
        }
        self.trans = {i: log10(j) for i, j in trans.items()}
        self.model = None

    def load_model(self, model_path):
        return kenlm.Model(model_path)

    # 语料生成器，并且初步预处理语料
    @staticmethod
    def text_generator(file_path, cut=False):
        '''
        输入:
            文本,list
        输出:
            ['你\n', '是\n', '谁\n']
        其中:
            cut,代表是否分词来判定
        
        '''
        for text in open(file_path).readlines():
            text = re.sub(u'[^\u4e00-\u9fa50-9a-zA-Z ]+', '\n', text)
            if cut:
                yield ' '.join(text.split()) + '\n'
            else:
                yield ' '.join(text) + '\n'

    @staticmethod
    def write_corpus(texts, filename):
        """将语料写到文件中，词与词(字与字)之间用空格隔开
        """
        with open(filename, 'w') as f:
            for s in texts:
                #s = ' '.join(s) + '\n'
                f.write(s)
        print('success writed')

    def count_ngrams(self):
        """
        通过os.system调用Kenlm的count_ngrams来统计频数
        
        # Counts n-grams from standard input.
        # corpus count:
        #   -h [ --help ]                     Show this help message
        #   -o [ --order ] arg                Order
        #   -T [ --temp_prefix ] arg (=/tmp/) Temporary file prefix
        #   -S [ --memory ] arg (=80%)        RAM
        #   --read_vocab_table arg            Vocabulary hash table to read.  This should
        #                                     be a probing hash table with size at the 
        #                                     beginning.
        #   --write_vocab_list arg            Vocabulary list to write as null-delimited 
        #                                     strings.
        """
        #corpus_file,vocab_file,ngram_file,memory = '50%',order = 4
        executive_code = self.kenlm_model_path + 'count_ngrams -S %s -o %s --write_vocab_list %s <%s >%s' % (
            self.memory, self.order, self.vocab_file, self.corpus_file,
            self.ngram_file)
        status = os.system(executive_code)
        if status == 0:
            return 'success,code is : %s , \n code is : %s ' % (status,
                                                                executive_code)
        else:
            return 'fail,code is : %s ,\n code is : %s ' % (status,
                                                            executive_code)

    def lm_train(self):
        '''
        # 训练数据格式一:保存成all.txt.parse 然后就可以直接训练了
        # 来源：https://github.com/DRUNK2013/lm-ken
        
        训练过程:
            输入 : self.corpus_path语料文件
            输出 : self.arpa_file语料文件
        
        报错：
        34304 , 需要增加样本量
        
        '''
        #corpus_file,arpa_file,memory = '50%',order = 4,skip_symbols = '"<unk>"'
        executive_code = self.kenlm_model_path + 'lmplz -S {} -o {} --skip_symbols {} < {} > {} --discount_fallback'.format(
            self.memory, self.order, self.skip_symbols, self.corpus_file,
            self.arpa_file)
        status = os.system(executive_code)
        if status == 0:
            return 'success,code is : %s , \n code is : %s ' % (status,
                                                                executive_code)
        else:
            return 'fail,code is : %s ,\n code is : %s ' % (status,
                                                            executive_code)

    def convert_format(self):
        '''
        
        ```
        ./kenlm/bin/build_binary weixin.arpa weixin.klm
        ```
        
        arpa是通用的语言模型格式，klm是kenlm定义的二进制格式，klm格式占用空间更少。
        
        报错：
        256 ： No such file or directory while opening output/test2.arpa
    
        '''
        #arpa_file,klm_file,memory = '50%'
        executive_code = self.kenlm_model_path + 'build_binary -S {} -s {} {}'.format(
            self.memory, self.arpa_file, self.klm_file)
        status = os.system(executive_code)
        if status == 0:
            return 'success,code is : %s , \n code is : %s ' % (status,
                                                                executive_code)
        else:
            return 'fail,code is : %s ,\n code is : %s ' % (status,
                                                            executive_code)

    '''
    分词模块
    '''

    def parse_text(self, text):
        return ' '.join(list(text))

    def viterbi(self, nodes):
        '''  # 分词系统
        #这里的转移概率是人工总结的，总的来说，就是要降低长词的可能性。
        #trans = {'bb':1, 'bc':0.15, 'cb':1, 'cd':0.01, 'db':1, 'de':0.01, 'eb':1, 'ee':0.001}
        #trans = {i:log10(j) for i,j in trans.items()}
        
        苏神的kenlm分词:
            b：单字词或者多字词的首字
            c：多字词的第二字
            d：多字词的第三字
            e：多字词的其余部分
        '''
        # py3的写法
        paths = nodes[0]
        for l in range(1, len(nodes)):
            paths_ = paths
            paths = {}
            for i in nodes[l]:
                nows = {}
                for j in paths_:
                    if j[-1] + i in self.trans:
                        nows[j +
                             i] = paths_[j] + nodes[l][i] + self.trans[j[-1] +
                                                                       i]
                #k = nows.values().index(max(nows.values()))
                k = max(nows, key=nows.get)
                #paths[nows.keys()[k]] = nows.values()[k]
                paths[k] = nows[k]
        #return paths.keys()[paths.values().index(max(paths.values()))]
        return max(paths, key=paths.get)

    def cp(self, s):
        if self.model == None:
            raise KeyError('please load model(.klm / .arpa).')
        return (
            self.model.score(' '.join(s), bos=False, eos=False) -
            self.model.score(' '.join(s[:-1]), bos=False, eos=False)) or -100.0

    def mycut(self, s):
        nodes = [{'b': self.cp(s[i]), 'c': self.cp(s[i - 1: i + 1]), 'd': self.cp(s[i - 2: i + 1]),\
                  'e': self.cp(s[i - 3: i + 1])} for i in range(len(s))]
        tags = self.viterbi(nodes)
        words = [s[0]]
        for i in range(1, len(s)):
            if tags[i] == 'b':
                words.append(s[i])
            else:
                words[-1] += s[i]
        return words

    '''
    kenlm n-grams训练模块 + 新词发现
    '''

    def unpack(self, t, s):
        return struct.unpack(t, s)[0]

    def read_ngrams(self):
        """读取思路参考https://github.com/kpu/kenlm/issues/201
        """
        # 数据读入
        f = open(self.vocab_file)
        chars = f.read()
        f.close()
        chars = chars.split('\x00')
        chars = [i for i in chars]  # .decode('utf-8')
        
        ngrams = [Counter({}) for _ in range(self.order)]
        total = 0
        size_per_item = self.order * 4 + 8
        f = open(self.ngram_file, 'rb')
        filedata = f.read()
        filesize = f.tell()
        f.close()
        for i in range(0, filesize, size_per_item):
            s = filedata[i:i + size_per_item]
            n = self.unpack('l', s[-8:])
            if n >= self.min_count:
                total += n
                c = [
                    self.unpack('i', s[j * 4:(j + 1) * 4])
                    for j in range(self.order)
                ]
                c = ''.join([chars[j] for j in c if j > 2])
                for j in range(self.order):  # len(c) -> self.order
                    ngrams[j][c[:j + 1]] = ngrams[j].get(c[:j + 1], 0) + n
        return ngrams, total

    def filter_ngrams(self, ngrams, total, min_pmi=1):
        """通过互信息过滤ngrams，只保留“结实”的ngram。
        """
        order = len(ngrams)
        if hasattr(min_pmi, '__iter__'):
            min_pmi = list(min_pmi)
        else:
            min_pmi = [min_pmi] * order
        #output_ngrams = set()
        output_ngrams = Counter()
        total = float(total)
        for i in range(order - 1, 0, -1):
            for w, v in ngrams[i].items():
                pmi = min([
                    total * v / (ngrams[j].get(w[:j + 1], total) *
                                 ngrams[i - j - 1].get(w[j + 1:], total))
                    for j in range(i)
                ])
                if math.log(pmi) >= min_pmi[i]:
                    #output_ngrams.add(w)
                    output_ngrams[w] = v
        return output_ngrams

    '''
    智能纠错模块
    '''

    def is_Chinese(self, word):
        for ch in word:
            if '\u4e00' <= ch <= '\u9fff':
                return True
        return False

    def word_match(self, text_a, text_b):
        '''
        筛选规则:
            # 字符数一致
            # 不为空
            # 拼音首字母一致
            
        输出:
            最佳是否相似,bool
        '''

        pinyin_n, match_w = 0, []
        text_a_pinyin = lazy_pinyin(text_a, style=Style.FIRST_LETTER)
        text_b_pinyin = lazy_pinyin(text_b, style=Style.FIRST_LETTER)
        #print(text_a_pinyin,text_b_pinyin)
        if len(text_a) > 0 and (len(text_b)
                                == len(text_a)) and self.is_Chinese(
                                    text_a) and self.is_Chinese(text_b):
            for n, w1 in enumerate(text_a):
                if text_b[n] == w1:
                    match_w.append(w1)
                if text_a_pinyin[n] == text_b_pinyin[n]:
                    pinyin_n += 1
            return True if len(match_w) > 0 and pinyin_n == len(
                text_a) else False
        else:
            return False

    def compare(self, text_a, text_b):
        '''
        通过kenlm对比两个文本的优劣:
            text_a - text_b > 0 , text_a 好
        '''
        return self.model.score(' '.join(text_a), bos=False,
                                eos=False) - self.model.score(
                                    ' '.join(text_b), bos=False, eos=False)

    def find_best_word(self, word, ngrams, freqs=10):
        '''
        通过kenlm找出比word更适合的词
        
        输入:
            word,str
            ngrams,dict,一个{word:freq}的词典
            
        输出:
            最佳替换word
        '''
        candidate = {
            bg: freq
            for bg, freq in ngrams.items()
            if self.word_match(word, bg) & (freq > freqs)
        }
        #if len(candidate) == 0:
        #    raise KeyError('zero candidate,large freqs')
        candidate_score = {
            k: self.compare(k, word)
            for k, v in candidate.items()
        }
        if len(candidate_score) > 0:
            return max(candidate_score, key=candidate_score.get)
        else:
            return word

    def word_discovery(
            self,
            ngrams_dict,
            good_pos=['n', 'v', 'ag', 'a', 'zg', 'd'],
            bad_words=['我', '你', '他', '也', '的', '是', '它', '再', '了', '让']):
        '''
        新词筛选
        筛选规则：
            - 分词分不出来
            - 词性也不包括以下几种
    
        jieba词性表：https://blog.csdn.net/orangefly0214/article/details/81391539
    
        坏词性：
            uj,ur,助词
            l,代词
    
        好词性：
            n,v,ag,a,zg,d(副词)
        '''
        new_words_2 = {}
        for nw, freq in tqdm(ngrams_dict.items()):
            lac_result = self.seg.cut(nw, mode='rank')
            if len(lac_result) != 3:
                continue
            words = lac_result[0]
            pos = lac_result[1]
            if (len(words) != 1)  and  \
               (len([gp for gp in good_pos if gp in ''.join(pos)]) > 0) \
               and (len([bw for bw in bad_words if bw in nw[0]]) == 0):
                new_words_2[nw] = freq
                #print(list(words))
        return new_words_2


if __name__ == '__main__':
    # 模型加载
    cfg = get_cfg()
    km = KenLM(cfg)

    km.write_corpus(
        km.text_generator(cfg.BASE.CHAR_FILE, cut=False),
        km.corpus_file)  # 将语料转存为文本

    # NLM模型训练
    status = km.lm_train()
    print(status)

    # NLM模型arpa文件转化
    km.convert_format()
    # '''
    # 新词发现
    # '''
    # # 模型n-grams生成
    km.count_ngrams()

    # # 模型读入与过滤
    # ngrams, total = km.read_ngrams()
    # ngrams_2 = km.filter_ngrams(ngrams, total, min_pmi=[0, 1, 3, 5])

    # # 新词发现
    # print(km.word_discovery(ngrams_2))
    # '''
    # 智能纠错
    # '''
    # # 加载模型
    # km.model = km.load_model(km.klm_file)

    # # n-grams读入
    # ngrams, total = km.read_ngrams()
    # ngrams_2 = km.filter_ngrams(ngrams, total, min_pmi=[0, 1, 3, 5])

    # sentence = '这瓶洗棉奶用着狠不错'
    # idx_errors = pycorrector.detect(sentence)

    # correct = []
    # for ide in idx_errors:
    #     right_word = km.find_best_word(ide[0], ngrams_2, freqs=0)
    #     if right_word != ide[0]:
    #         correct.append([right_word] + ide)

    # print('错误：', idx_errors)
    # print('pycorrector的结果：', pycorrector.correct(sentence))
    # print('kenlm的结果：', correct)