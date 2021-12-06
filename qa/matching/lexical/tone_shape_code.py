from config import get_cfg
from pypinyin import pinyin, lazy_pinyin, Style
import pypinyin
from qa.tools.mongo import Mongo
from qa.queryUnderstanding.querySegmentation import Words


class TSC:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.mongo = Mongo(cfg, cfg.INVERTEDINDEX.DB_NAME)
        self.get_map()

    def get_map(self):
        tsc = Words(self.cfg).get_tsc_words
        self.shengmuDict = tsc['shenmu_encode']
        self.yunmuDict = tsc['yunmu_encode']
        self.shapeDict = tsc['structure_map']
        self.strokesDict = tsc['stroke_map']
        self.strokesDictReverse = {v: int(k) for k, v in self.strokesDict.items()}
        self.hanziStructureDict = tsc['han_structure']
        self.hanziStrokesDict = tsc['stroke']
        self.hanziSSCDict = tsc['tsc_map']
        self.fourcorner_encode = tsc['fourcorner_encode']

    def getSoundCode(self, one_chi_word):
        res = []
        shengmuStr = pinyin(one_chi_word,
                            style=pypinyin.INITIALS,
                            heteronym=False,
                            strict=False)[0][0]
        if shengmuStr not in self.shengmuDict:
            shengmuStr = '0'

        yunmuStrFullStrict = pinyin(one_chi_word,
                                    style=pypinyin.FINALS_TONE3,
                                    heteronym=False,
                                    strict=True)[0][0]

        yindiao = '0'
        if yunmuStrFullStrict[-1] in ['1', '2', '3', '4']:
            yindiao = yunmuStrFullStrict[-1]
            yunmuStrFullStrict = yunmuStrFullStrict[:-1]

        if yunmuStrFullStrict in self.yunmuDict:
            #声母，韵母辅音补码，韵母，音调
            res.append(self.yunmuDict[yunmuStrFullStrict])
            res.append(self.shengmuDict[shengmuStr])
            res.append('0')
        elif len(yunmuStrFullStrict) > 1:
            res.append(self.yunmuDict[yunmuStrFullStrict[1:]])
            res.append(self.shengmuDict[shengmuStr])
            res.append(self.yunmuDict[yunmuStrFullStrict[0]])
        else:
            res.append('0')
            res.append(self.shengmuDict[shengmuStr])
            res.append('0')

        res.append(yindiao)
        return res

    def getSoundCodes(self, words):

        shengmuStrs = pinyin(words,
                             style=pypinyin.INITIALS,
                             heteronym=False,
                             strict=False)
        yunmuStrFullStricts = pinyin(words,
                                     style=pypinyin.FINALS_TONE3,
                                     heteronym=False,
                                     strict=True)
        soundCodes = []
        for shengmuStr0, yunmuStrFullStrict0 in zip(shengmuStrs,
                                                    yunmuStrFullStricts):
            res = []
            shengmuStr = shengmuStr0[0]
            yunmuStrFullStrict = yunmuStrFullStrict0[0]

            if shengmuStr not in self.shengmuDict:
                shengmuStr = '0'

            yindiao = '0'
            if yunmuStrFullStrict[-1] in ['1', '2', '3', '4']:
                yindiao = yunmuStrFullStrict[-1]
                yunmuStrFullStrict = yunmuStrFullStrict[:-1]

            if yunmuStrFullStrict in self.yunmuDict:
                #声母，韵母辅音补码，韵母，音调
                res.append(self.yunmuDict[yunmuStrFullStrict])
                res.append(self.shengmuDict[shengmuStr])
                res.append('0')
            elif len(yunmuStrFullStrict) > 1:
                res.append(self.yunmuDict[yunmuStrFullStrict[1:]])
                res.append(self.shengmuDict[shengmuStr])
                res.append(self.yunmuDict[yunmuStrFullStrict[0]])
            else:
                res.append('0')
                res.append(self.shengmuDict[shengmuStr])
                res.append('0')

            res.append(yindiao)
            soundCodes.append(res)

        return soundCodes

    def getShapeCode(self, one_chi_word):
        res = []
        structureShape = self.hanziStructureDict.get(one_chi_word, '0')  #形体结构
        res.append(self.shapeDict[structureShape])

        fourCornerCode = self.fourcorner_encode.get(one_chi_word,
                                                    None)  #四角号码（5位数字）
        if fourCornerCode is None:
            res.extend(['0', '0', '0', '0', '0'])
        else:
            res.extend(fourCornerCode[:])

        strokes = self.hanziStrokesDict.get(one_chi_word, '0')  #笔画数
        if int(strokes) > 35:
            res.append('Z')
        else:
            res.append(self.strokesDict.get(int(strokes), '0'))
        return res

    def getSSC(self, hanzi_sentence, encode_way='ALL'):
        hanzi_sentence_ssc_list = []
        for one_chi_word in hanzi_sentence:
            ssc = self.hanziSSCDict.get(one_chi_word, None)
            if ssc is None:
                soundCode = self.getSoundCode(one_chi_word)
                shapeCode = self.getShapeCode(one_chi_word)
                ssc = "".join(soundCode + shapeCode)
            if encode_way == "SOUND":
                ssc = ssc[:4]
            elif encode_way == "SHAPE":
                ssc = ssc[4:]
            else:
                pass
            hanzi_sentence_ssc_list.append(ssc)
        return hanzi_sentence_ssc_list

    def getSSC_sentence(self, hanzi_sentence, encode_way, analyzer):

        hanzi_sentence_ssc_list = []

        result_seg = analyzer.seg(hanzi_sentence)
        words = []
        for term in result_seg:
            words.append(term.word)

        soundCodes = self.getSoundCodes(words)

        for one_chi_word, soundCode in zip(hanzi_sentence, soundCodes):
            if encode_way == "SOUND":
                ssc = "".join(soundCode)
            elif encode_way == "SHAPE":
                shapeCode = self.getShapeCode(one_chi_word)
                ssc = "".join(shapeCode)
            elif encode_way == "ALL":
                shapeCode = self.getShapeCode(one_chi_word)
                ssc = "".join(soundCode + shapeCode)

            hanzi_sentence_ssc_list.append(ssc)

        return hanzi_sentence_ssc_list

    @staticmethod
    def computeSoundCodeSimilarity(
            soundCode1, soundCode2):  #soundCode=['2', '8', '5', '2']
        featureSize = len(soundCode1)
        wights = [0.4, 0.4, 0.1, 0.1]
        multiplier = []
        for i in range(featureSize):
            if soundCode1[i] == soundCode2[i]:
                multiplier.append(1)
            else:
                multiplier.append(0)
        soundSimilarity = 0
        for i in range(featureSize):
            soundSimilarity += wights[i] * multiplier[i]
        return soundSimilarity

    @staticmethod
    def computeShapeCodeSimilarity(
            shapeCode1, shapeCode2, strokesDictReverse
    ):  #shapeCode=['5', '6', '0', '1', '0', '3', '8']
        featureSize = len(shapeCode1)
        wights = [0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.25]
        multiplier = []
        for i in range(featureSize - 1):
            if shapeCode1[i] == shapeCode2[i]:
                multiplier.append(1)
            else:
                multiplier.append(0)
        multiplier.append(1 - abs(strokesDictReverse[shapeCode1[-1]] -
                                  strokesDictReverse[shapeCode2[-1]]) * 1.0 /
                          max(strokesDictReverse[shapeCode1[-1]],
                              strokesDictReverse[shapeCode2[-1]]))
        shapeSimilarity = 0
        for i in range(featureSize):
            shapeSimilarity += wights[i] * multiplier[i]
        return shapeSimilarity

    @staticmethod
    def computeSSCSimilaruty(ssc1,
                             ssc2,
                             ssc_encode_way,
                             strokesDictReverse,
                             soundWeight=0.5):
        #return 0.5*computeSoundCodeSimilarity(ssc1[:4], ssc2[:4])+0.5*computeShapeCodeSimilarity(ssc1[4:], ssc2[4:])
        if ssc_encode_way == "SOUND":
            return TSC.computeSoundCodeSimilarity(ssc1, ssc2)
        elif ssc_encode_way == "SHAPE":
            return TSC.computeShapeCodeSimilarity(ssc1, ssc2,
                                                  strokesDictReverse)
        else:
            soundSimi = TSC.computeSoundCodeSimilarity(ssc1[:4], ssc2[:4])
            shapeSimi = TSC.computeShapeCodeSimilarity(ssc1[4:], ssc2[4:],
                                                       strokesDictReverse)
            return soundWeight * soundSimi + (1 - soundWeight) * shapeSimi


class VatiantKMP(object):
    # 求模式串T的next函数（修正方法）值并存入next数组
    # nextVal = [-1]
    # startIdxRes = []#写在这里，多次使用kmp时startIdxRes不会被清空而是存放了上一次的数据，影响结果
    def __init__(self, threshold):
        self.threshold = threshold
        self.nextVal = [-1]
        self.startIdxRes = []

    def reset(self):
        self.nextVal = [-1]
        self.startIdxRes = []

    def indexKMP(self, haystack, needle, ssc_encode_way, strokesDictReverse):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        """
        try:
            return haystack.index(needle)
        except:
            return -1
        """
        #子串定位，即模式匹配，可采用BF算法  也可采用KMP算法，我采用KMP算法
        # 0<=pos<= len(strS) - len(strT)) + 1
        self.getNextVal(needle, ssc_encode_way, strokesDictReverse)
        i = 0
        while i < len(haystack):
            j = 0
            while i < len(haystack) and j < len(needle):
                #if j == -1 or haystack[i] == needle[j]:
                if j == -1 or TSC.computeSSCSimilaruty(
                        haystack[i], needle[j], ssc_encode_way,
                        strokesDictReverse) > self.threshold:
                    i += 1
                    j += 1
                else:
                    j = self.nextVal[j]
            if j == len(needle):
                self.startIdxRes.append(i - len(needle))

    def getNextVal(self, strT, ssc_encode_way, strokesDictReverse):
        i = 0
        j = -1
        while i < len(strT) - 1:
            #if j == -1 or strT[i] == strT[j]:
            if j == -1 or TSC.computeSSCSimilaruty(
                    strT[i], strT[j], ssc_encode_way,
                    strokesDictReverse) > self.threshold:
                i += 1
                j += 1
                #if i < len(strT) and strT[i] == strT[j]:
                if i < len(strT) and TSC.computeSSCSimilaruty(
                        strT[i], strT[j], ssc_encode_way,
                        strokesDictReverse) > self.threshold:
                    self.nextVal.append(self.nextVal[j])
                else:
                    self.nextVal.append(j)
            else:
                j = self.nextVal[j]


if __name__ == '__main__':
    cfg = get_cfg()
    tsc = TSC(cfg)
    import time
    s = time.time()
    # chi_word1 = '紫琅路'
    # chi_word2 = '国我爱你女生于无娃哇紫狼路爽晕约紫薇路又刘页列而紫粮路掩连哟罗'

    # chi_word1 = '呕吐'
    # chi_word2 = '呕土不止'

    # chi_word1 = '感冒'
    # chi_word2 = '我家猫猫感帽了'

    # chi_word1 = '咳嗽'
    # chi_word2 = '一直咳数'

    # chi_word1 = '没有精神'
    # chi_word2 = '我家猫猫精神没有'
    
    chi_word1 = '擦洗'
    chi_word2 = '我家狗拉啦稀了'

    # chi_word1 = '尿血'
    # chi_word2 = '狗狗偶尔尿xie怎么办'
    chi_word1_ssc = tsc.getSSC(chi_word1, 'ALL')
    print(chi_word1_ssc)
    print('encoding: {}'.format(time.time() - s))

    chi_word2_ssc = tsc.getSSC(chi_word2, 'ALL')
    print(chi_word2_ssc)
    print('encoding: {}'.format(time.time() - s))

    #应用串的模式匹配KMP算法，找变异词。效率比BF算法高
    kmp = VatiantKMP(0.5)
    kmp.indexKMP(chi_word2_ssc, chi_word1_ssc, 'ALL', tsc.strokesDictReverse)  #主串S、模式串T
    print(kmp.startIdxRes)
    print('kmp: {}'.format(time.time() - s))

    variabt_word = set()
    for i in kmp.startIdxRes:
        variabt_word.add(chi_word2[i:i + len(chi_word1)])
    print('变异词：', variabt_word)

    print('all: {}'.format(time.time() - s))