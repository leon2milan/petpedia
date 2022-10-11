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
    """
    It converts a string to lowercase, and then converts it to Chinese characters
    
    :param s: the string to be converted
    :return: A string with all the characters in lower case.
    """
    s = cn2an.transform(s, "an2cn").lower()
    return s


### 全半角转换 ###
def full2half(s):
    """
    It converts a string of full-width characters to half-width characters
    
    :param s: the string to be converted
    :return: the converted string.
    """
    return convert(s, Z_ASCII, H_ASCII)


### 繁简体转换 ###
def trans2simple(str):
    """
    It takes a string and returns a string
    
    :param str: The string to be converted
    :return: the converted string.
    """
    return converter.convert(str)

### 无意义符号去除 ###
def demojize(s):
    """
    It takes a string and returns a string
    
    :param s: The string to demojize
    """
    return emoji.demojize(s)


### emoji 判断 ###
def is_emoji(s):
    """
    It checks if all the characters in the string are emojis
    
    :param s: The string to check
    :return: A boolean value.
    """
    return all(emoji.is_emoji(i) for i in s)


### 分段 ###
def normal_cut_sentence(text):
    """
    1. Replace all the punctuation marks with a newline character, except for the punctuation marks that
    are followed by a quotation mark
    
    :param text: the text to be split
    :return: A list of sentences.
    """
    text = re.sub('([。！？\?])([^’”])', r'\1\n\2', text)  #普通断句符号且后面没有引号
    text = re.sub('(\.{6})([^’”])', r'\1\n\2', text)  #英文省略号且后面没有引号
    text = re.sub('(\…{2})([^’”])', r'\1\n\2', text)  #中文省略号且后面没有引号
    text = re.sub('([.。！？\?\.{6}\…{2}][’”])([^’”])', r'\1\n\2',
                  text)  #断句号+引号且后面没有引号
    return text.split("\n")


def cut_sentence_with_quotation_marks(text):
    """
    1. Find all the quotation marks in the text.
    2. Cut the text into sentences before the quotation marks.
    3. Add the quotation marks to the list of sentences.
    4. Cut the text into sentences after the quotation marks.
    5. Add the sentences after the quotation marks to the list of sentences
    
    :param text: the text to be cut
    :return: A list of sentences.
    """
    p = re.compile("“.*?”")
    list = []
    index = 0
    length = len(text)
    for i in p.finditer(text):
        start = i.start()
        end = i.end()
        temp = ''.join(text[j] for j in range(index, start))
        if temp != '':
            temp_list = normal_cut_sentence(temp)
            list += temp_list
        temp = ''.join(text[k] for k in range(start, end))
        if temp != ' ':
            list.append(temp)
        index = end
    return list


def clean(s, is_tran=False, has_emogi=False, keep_zh=False):
    """
    - Remove all punctuations
    - Convert all capital letters to lower case
    - Convert all full-width characters to half-width characters
    - Convert all traditional Chinese characters to simplified Chinese characters
    - Remove all emojis
    - Remove all non-Chinese characters
    
    :param s: the string to be cleaned
    :param is_tran: whether to convert traditional Chinese to simplified Chinese, defaults to False
    (optional)
    :param has_emogi: whether to remove emogi, defaults to False (optional)
    :param keep_zh: whether to keep Chinese characters, defaults to False (optional)
    :return: A string with all the punctuation removed.
    """
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
    """
    It removes all parentheses and their contents from a string
    
    :param entity: the entity to be processed
    :return: A list of tuples.
    """
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
    """
    If you're on Python 2, convert bytes to unicode. If you're on Python 3, convert bytes to str
    
    :param text: The text to be tokenized
    :return: the text in unicode format.
    """
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError(f"Unsupported string type: {type(text)}")
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError(f"Unsupported string type: {type(text)}")
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    return '\u4e00' <= uchar <= '\u9fa5'


def is_chinese_string(string):
    """判断是否全为汉字"""
    return all(is_chinese(c) for c in string)


def is_number(uchar):
    """判断一个unicode是否是数字"""
    return 'u0030' <= uchar <= 'u0039'


def is_number_string(string):
    """判断是否全为数字"""
    return all(is_number(c) for c in string)


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    return 'u0041' <= uchar <= 'u005a' or 'u0061' <= uchar <= 'u007a'


def is_alphabet_string(string):
    """判断是否全部为英文字母"""
    return not any(c < 'a' or c > 'z' for c in string)


def is_alp_diag_string(string):
    """判断是否只是英文字母和数字"""
    return not any(not 'a' <= c <= 'z' and not c.isdigit() for c in string)


def is_other(uchar):
    """判断是否非汉字，非数字和非英文字符 , 错误"""
    return not is_chinese(uchar) and not is_number(uchar) and not is_alphabet(uchar)


def is_other_string(string):
    """判断是否非汉字，非数字和非英文字符"""
    return not is_chinese_string(string) and not is_number_string(string) and not is_alphabet_string(string)


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
    return re.sub(
        "[\s+\.\!\/<>“”,$%^*(+\"\']+|[+——！，。？、_~@#￥%……&*（）()《》「」\{\};:：；]+", "",
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
        if not include_symbol and re_han.match(blk) or include_symbol:
            result.append((blk, start_idx))
        start_idx += len(blk)
    return result


def nonsence_detect(text):
    """
    1. Remove punctuation
    2. Check if the text is empty
    3. Check if the text is an emoji
    4. Check if the text is a string of alphabets or digits
    5. Check if the text is a string of Chinese characters that are all in the exclude list
    6. Check if the text is a string of characters that are not Chinese
    
    If any of the above conditions are met, the text is considered as nonsense
    
    :param text: the text to be checked
    """
    text = remove_punctuation(text.lower())
    if len(text) < 1:
        return True
    if is_emoji(text):
        return True
    if is_alp_diag_string(text):
        return True
    exclude = ['阿', '啊', '啦', '唉', '呢', '吧', '了', '哇', '呀', '吗', '哦', '哈', '哟', '么']

    if len(set(text)) == 1 and text[0] in exclude:
        return True
    if all(x in exclude for x in text):
        return True
    return all(not is_chinese(x) for x in text)


if __name__ == '__main__':
    print(nonsence_detect('sdgnosijt2_@IR!{G"AS<>G"LM{EGGEL'))