'''
处理体格检查数据中多余的字符
'''

import re
from zhon.hanzi import punctuation
import datetime


class ProcessNumberformat():

    def __init__(self, settings=None):
        self.settings = settings

    def lm_find_unchinese(self, file, bcsexam):
        '''
        去除汉字、数字
        :param file:
        :return:
        '''
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        pattern1 = re.compile(r'[a-z]')
        pattern2 = re.compile(r'[A-Z]')

        # pattern3 = re.compile('[·’!"\#$%&\'()ｎａ＃！（）·*+。⊙⊙°の→℃,-/:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+')
        unchinese = re.sub(pattern, "", file)  # 排除汉字
        unchinese = re.sub(pattern1, "", unchinese)  # 排除小写字母
        unchinese = re.sub(pattern2, "", unchinese)  # 排除大写字母
        if bcsexam == 'weight':
            # print('w')
            pattern3 = re.compile('[()+*`·/-]')
        else:
            # print('o')
            pattern3 = re.compile(
                '[·’!"\#$%&\'()ｎａ＃！（）·*+。⊙⊙°の→℃,-/:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
            )
        unchinese = re.sub(pattern3, "", unchinese)  # 排除乱起八早的字符
        unchinese = re.sub('[{}]'.format(punctuation), "", unchinese)  # 排除中文符号
        # print("unchinese:",unchinese)
        return unchinese

    def processphybcs(self, temp, mean, threshold, exam):
        '''
        处理温度字符串 -针对bcs评分的
        :param temp:
        :return:
        '''
        temperature = 0
        # print(temp)

        temp = self.lm_find_unchinese(temp, exam)
        # print(temp)

        temperaturesm = temp.replace('..', '.').replace(' ',
                                                        '.').replace('\\', '')

        # temperaturesm = temp.replace('*', '').replace('·', '').replace('-', '').replace(',', '.').replace('<', '').replace('。', '.').replace('/', '').replace('+', '').replace('..', '.').replace('`', '').replace(')', '').replace('\';', '').replace(' ', '').replace('>', '')
        if len(temperaturesm) >= 1 and temperaturesm[-1] == '.':
            temperaturesm = temperaturesm[0:-1]
        if len(temperaturesm) >= 1 and temperaturesm[0] == '.':
            temperaturesm = temperaturesm[1:]
        if (temperaturesm == 'nan' or temperaturesm == ''
                or temperaturesm == '0' or temperaturesm == '未知'
                or temperaturesm == '无' or temperaturesm == 'ok'):
            temperature = 0.1
        if (temperaturesm == '正常'):
            temperature = mean
        if (len(temperaturesm.split('.')) > 2):
            temperaturesm = temperaturesm.split('.')[0]
        if (len(temperaturesm.replace(' ', '')) > 1):
            # print(temperaturesm)
            temperature = float(temperaturesm)
        else:
            temperature = 0.1

        # temperature=(temperature - mean) / std
        if temperature > threshold:
            temperature = temperature / 10
            if temperature > 100:
                temperature = 0.1

        return temperature

    def processphy(self, temp, avg, exam):
        '''
        处理温度字符串-针对疾病预测的
        :param temp:
        :return:
        '''
        temperature = 0
        # print(temp)

        temp = self.lm_find_unchinese(temp, exam)
        # print(temp)

        temperaturesm = temp.replace('..', '.').replace(' ',
                                                        '.').replace('\\', '')

        # temperaturesm = temp.replace('*', '').replace('·', '').replace('-', '').replace(',', '.').replace('<', '').replace('。', '.').replace('/', '').replace('+', '').replace('..', '.').replace('`', '').replace(')', '').replace('\';', '').replace(' ', '').replace('>', '')
        if len(temperaturesm) >= 1 and temperaturesm[-1] == '.':
            temperaturesm = temperaturesm[0:-1]
        if len(temperaturesm) >= 1 and temperaturesm[0] == '.':
            temperaturesm = temperaturesm[1:]
        if (temperaturesm == 'nan' or temperaturesm == ''
                or temperaturesm == '0' or temperaturesm == '未知'
                or temperaturesm == '无' or temperaturesm == 'ok'):
            temperature = 0.1
        if (temperaturesm == '正常'):
            temperature = avg
        if (len(temperaturesm.split('.')) > 2):
            temperaturesm = temperaturesm.split('.')[0]
        if (len(temperaturesm.replace(' ', '')) > 1):
            # print(temperaturesm)
            temperature = float(temperaturesm)
        else:
            temperature = 0.1
        return temperature

    def process_petageday(self, birthday, createtime):
        '''
        计算宠物的年龄
        :param birthday:
        :param petage:
        :param createtime:
        :return:
        '''
        birthdaynew = ''
        # print(birthday,createtime)
        if str(birthday) == 'nan' or str(birthday) == '':
            birthdaynew = createtime
            # print('yyyy')
        else:
            # print('xxxx')
            d1 = datetime.datetime.strptime(birthday, '%Y-%m-%d %H:%M:%S.%f')
            d2 = datetime.datetime.strptime(createtime, '%Y-%m-%d %H:%M:%S.%f')
            # 间隔天数
            day = (d2 - d1).days
            if day > 18000:  # 大于50年
                birthdaynew = createtime
            else:
                birthdaynew = birthday

        d1 = datetime.datetime.strptime(birthdaynew, '%Y-%m-%d %H:%M:%S.%f')
        d2 = datetime.datetime.strptime(createtime, '%Y-%m-%d %H:%M:%S.%f')
        age_day = (d2 - d1).days
        return age_day

    def getpetcent(self, x):
        '''
        得到百分比
        :param x:
        :return:
        '''
        return '%.2f%%' % (x * 100)

    def getdate(self, x):
        d1 = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
        return d1

    def getyearandmonth(self, x):
        xdate = x.split(' ')[0]
        xdates = xdate.split('-')
        return xdates[0] + '-' + xdates[1]


if __name__ == "__main__":
    x = 1
    y = 9
    rate = x / y
    r = '%.2f%%' % (rate * 100)
    print(r)

    x = '2020-07-27 09:21:29.0'
    x = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')
    print(x)

    file = '3.84/'
    profmat = ProcessNumberformat()
    xx = profmat.processphybcs(file, 0, 100, 'temper')
    print(xx)
    xx = profmat.processphybcs(file, 0, 100, 'weight')
    print(xx)
