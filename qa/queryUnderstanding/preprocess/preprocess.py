from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pypinyin
import re
import six
from pypinyin import pinyin
import re
import cn2an
import emoji
import os
import opencc
from qa.queryUnderstanding.preprocess.cnvk import convert, Z_ASCII, H_ASCII

cur_dir = os.path.dirname(os.path.abspath("__file__"))

converter = opencc.OpenCC('t2s.json')


### 大小写转换 ###
def cap2lower(s):
    s = cn2an.transform(s, "an2cn").lower()
    return s


### 全半角转换 ###
def full2half(s):
    return convert(s, Z_ASCII, H_ASCII)


### 繁简体转换 ###
def trans2simple(str):
    return converter.convert(str)


### 无意义符号去除 ###
def demojize(s):
    return emoji.demojize(s)


### 分段 ###
def normal_cut_sentence(text):
    text = re.sub('([。！？\?])([^’”])', r'\1\n\2', text)  #普通断句符号且后面没有引号
    text = re.sub('(\.{6})([^’”])', r'\1\n\2', text)  #英文省略号且后面没有引号
    text = re.sub('(\…{2})([^’”])', r'\1\n\2', text)  #中文省略号且后面没有引号
    text = re.sub('([.。！？\?\.{6}\…{2}][’”])([^’”])', r'\1\n\2',
                  text)  #断句号+引号且后面没有引号
    return text.split("\n")


def cut_sentence_with_quotation_marks(text):
    p = re.compile("“.*?”")
    list = []
    index = 0
    length = len(text)
    for i in p.finditer(text):
        temp = ''
        start = i.start()
        end = i.end()
        for j in range(index, start):
            temp += text[j]
        if temp != '':
            temp_list = normal_cut_sentence(temp)
            list += temp_list
        temp = ''
        for k in range(start, end):
            temp += text[k]
        if temp != ' ':
            list.append(temp)
        index = end
    return list


def clean(s, is_tran=False, has_emogi=False, keep_zh=False):
    s = re.sub('[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+', '',
               s)
    s = cap2lower(s)
    s = full2half(s)
    if is_tran:
        s = trans2simple(s)
    if has_emogi:
        s = demojize(s)
    if keep_zh:
        s = re.sub('[^\u4e00-\u9fa5]+', '', 's')
    return s


def remove_parentheses(entity):
    keys = {'［', '(', '[', '（'}
    symbol = {'］': '［', ')': '(', ']': '[', '）': '（'}
    stack = []
    remove = []
    for index, s in enumerate(entity):
        if s in keys:
            stack.append((s, index))
        if s in symbol:
            if not stack: continue
            temp_v, temp_index = stack.pop()
            if entity[index - 1] == '\\':
                t = entity[temp_index - 1:index + 1]
                remove.append(t)
            else:
                remove.append(entity[temp_index:index + 1])

    for r in remove:
        entity = entity.replace(r, '')
    return entity


re_eng = re.compile('[same_stroke.txt-zA-Z0-9]', re.U)
# re_han = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&]+)", re.U)
re_han = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)", re.U)
re_poun = re.compile('\W+', re.U)


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if '\u4e00' <= uchar <= '\u9fa5':
        return True
    else:
        return False


def is_chinese_string(string):
    """判断是否全为汉字"""
    for c in string:
        if not is_chinese(c):
            return False
    return True


def is_number(uchar):
    """判断一个unicode是否是数字"""
    if u'u0030' <= uchar <= u'u0039':
        return True
    else:
        return False


def is_number_string(string):
    """判断是否全为数字"""
    for c in string:
        print(c)
        print(is_number(c))
        if not is_number(c):
            return False
    return True


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (u'u0041' <= uchar <= u'u005a') or (u'u0061' <= uchar <= u'u007a'):
        return True
    else:
        return False


def is_alphabet_string(string):
    """判断是否全部为英文字母"""
    for c in string:
        if c < 'a' or c > 'z':
            return False
    return True


def is_alp_diag_string(string):
    """判断是否只是英文字母和数字"""
    for c in string:
        if not ('a' <= c <= 'z') and not c.isdigit():
            return False
    return True


def is_other(uchar):
    """判断是否非汉字，非数字和非英文字符 , 错误"""
    if not is_chinese(uchar) and not is_number(uchar) and not is_alphabet(
            uchar):
        return True
    else:
        return False


def is_other_string(string):
    """判断是否非汉字，非数字和非英文字符"""
    if not is_chinese_string(string) and not is_number_string(
            string) and not is_alphabet_string(string):
        return True
    return False


def B2Q(uchar):
    """半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角字符就返回原来的字符
        return uchar
    if inside_code == 0x0020:  # 除了空格其他的全角半角的公式为:半角=全角-0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code)


def Q2B(uchar):
    """全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


def stringQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])


def uniform(ustring):
    """格式化字符串，完成全角转半角，大写转小写的工作"""
    return stringQ2B(ustring).lower()


def remove_punctuation(strs):
    """
    去除标点符号
    :param strs:
    :return:
    """
    return re.sub("[\s+\.\!\/<>“”,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",
                  strs.strip())


def get_homophones_by_char(input_char):
    """
    根据汉字取同音字
    :param input_char:
    :return:
    """
    result = []
    # CJK统一汉字区的范围是0x4E00-0x9FA5,也就是我们经常提到的20902个汉字
    for i in range(0x4e00, 0x9fa6):
        if pinyin([chr(i)], style=pypinyin.NORMAL)[0][0] == pinyin(
                input_char, style=pypinyin.NORMAL)[0][0]:
            result.append(chr(i))
    return result


def get_homophones_by_pinyin(input_pinyin):
    """
    根据拼音取同音字
    :param input_pinyin:
    :return:
    """
    result = []
    # CJK统一汉字区的范围是0x4E00-0x9FA5,也就是我们经常提到的20902个汉字
    for i in range(0x4e00, 0x9fa6):
        if pinyin([chr(i)], style=pypinyin.TONE2)[0][0] == input_pinyin:
            # TONE2: 中zho1ng
            result.append(chr(i))
    return result


def split_long_text(text, include_symbol=False):
    """
    长句切分为短句
    :param text: str
    :param include_symbol: bool
    :return: (sentence, idx)
    """
    result = []
    blocks = re_han.split(text)
    start_idx = 0
    for blk in blocks:
        if not blk:
            continue
        if include_symbol:
            result.append((blk, start_idx))
        else:
            if re_han.match(blk):
                result.append((blk, start_idx))
        start_idx += len(blk)
    return result