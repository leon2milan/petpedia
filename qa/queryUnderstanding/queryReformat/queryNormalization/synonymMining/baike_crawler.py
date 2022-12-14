import os
import re
import time
import traceback
import urllib
import string
from bs4 import BeautifulSoup
from qa.queryUnderstanding.preprocess import remove_parentheses
from qa.tools import setup_logger

logger = setup_logger()


def url_parse(url, word):
    word = urllib.parse.quote(word)
    url = url.format(a=word)
    return url


def get_text_from_tag(tag):
    return tag.get_text()


def get_info_box(soup):
    new_dict = dict()
    # base_info
    base_info = soup.find(attrs={"class": "basic-info J-basic-info cmn-clearfix"})
    if base_info is not None:
        all_name = base_info.find_all(attrs={"class": "basicInfo-item name"})
        all_value = base_info.find_all(attrs={"class": "basicInfo-item value"})
        if len(all_name) != len(all_value):
            logger.error('name and value not equal')
            raise Exception('name and value not equal')
        info_size = len(all_name)
        for i in range(info_size):
            name, value = all_name[i], all_value[i]
            name, value = name.get_text(strip=True).replace(
                u'\xa0', ''), value.get_text(strip=False)
            new_dict[name] = value
    return new_dict


def get_description(soup):
    new_dict = dict()
    desc_label = soup.select('meta[name="description"]')
    if not desc_label:
        new_dict['description'] = ''
    else:
        description = soup.select('meta[name="description"]')[0].get('content')
        new_dict['description'] = description
    return new_dict


def baike_synonym_detect(base_path, word_code_list):
    out_path = os.path.join(base_path, 'baike_synonym.txt')
    if os.path.exists(out_path):
        os.remove(out_path)
    for params in word_code_list:
        baike_search(base_path, params)


def baike_crawler(text):
    base_url = 'https://baike.baidu.com/item/{}'.format(text)
    url = urllib.parse.quote(base_url, safe=string.printable)
    response = urllib.request.urlopen(url)
    data = response.read()
    soup = BeautifulSoup(data)

    item_json = dict()

    des_dict = get_description(soup)
    item_json.update(des_dict)

    info_box_dict = get_info_box(soup)
    item_json.update(info_box_dict)

    synonym_list = get_synonym(item_json)
    return synonym_list


def baike_search(base_path, params):
    key_word, word_code = params
    key_word = remove_parentheses(key_word)
    file = open(os.path.join(base_path, 'baike_synonym.txt'),
                'a',
                encoding='utf8')
    try:
        synonym_list = baike_crawler(key_word)
        if len(synonym_list) > 0:
            write_line = str(word_code) + '\t' + key_word + '\t' + '|'.join(
                synonym_list) + '\n'
            file.write(write_line)

        logger.info(' input word = {a}, find {b} synonyms...'.format(
            a=key_word, b=len(synonym_list)))
        return synonym_list

    except Exception:
        logger.error(' input word = {a}, occur an error!'.format(a=key_word))
        traceback.print_exc()
    time.sleep(0.1)


def get_synonym(baike_json):
    info_key = ['??????', '????????????', '??????', '????????????', '????????????', '??????']
    pattern_list = ['??????', '??????', '??????']

    info_set = set()
    for key in info_key:
        if key in baike_json:
            value = baike_json[key]
            if value[-1] == '???':
                value = value[:-1]
            value = seg(value)
            info_set = info_set | set(value)

    description = baike_json['description']
    for p in pattern_list:
        pattern = r'' + p
        result = re_match(pattern, description)
        for r in result:
            value = seg(r)
            info_set = info_set | set(value)

    info_set = [
        s.strip().replace(u'\xa0',
                          '').replace('"', '').replace('???', '').replace(
                              '???', '').replace('???', '').replace('???', '')
        for s in info_set
    ]
    return info_set


def re_match(word, text):
    p_str = r'{a}(.+?)[?????????\s)???(???]'.format(a=word)
    pattern = re.compile(p_str)
    result = re.findall(pattern, text)
    return result


def seg(text):
    segment = [',', '???', '???', '???']
    current_seg = '&&'
    for seg in segment:
        if seg in text:
            current_seg = seg

    return text.split(current_seg)


if __name__ == '__main__':
    print(baike_crawler('??????'))
