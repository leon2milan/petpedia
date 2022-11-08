from dataclasses import dataclass, field
from config import get_cfg, CfgNode
import pandas as pd
import os
import json
import numpy as np
from healthyMangement.numberUtil import ProcessNumberformat


@dataclass
class DiseaseUtil:
    cfg: CfgNode = field(default_factory=get_cfg)

    def __call__(self):
        """
        The function takes in the dataframe and returns a dictionary with the disease rate, disease
        dictionary, disease count, deleted diseases, cat cat variety, cat max weight, cat min weight,
        dog dog variety, dog max weight, and dog min weight
        :return: A dictionary with the following keys:
            disease_rate
            disease_dict
            disease_count
            deldisease
            varcatcat
            variety_maxweight_cat
            varietyminweight_cat
            varcatdog
            variety_maxweight_dog
            varietyminweight_dog
        """
        disease_rate = self.rate_process()
        disease_dict, disease_count, deldisease = self.disease_process()
        varcatcat, variety_maxweight_cat, varietyminweight_cat, varcatdog, variety_maxweight_dog, varietyminweight_dog = self.weight_process(
        )
        return {
            'disease_rate': disease_rate,
            'disease_dict': disease_dict,
            'disease_count': disease_count,
            'deldisease': deldisease,
            'varcatcat': varcatcat,
            'variety_maxweight_cat': variety_maxweight_cat,
            'varietyminweight_cat': varietyminweight_cat,
            'varcatdog': varcatdog,
            'variety_maxweight_dog': variety_maxweight_dog,
            'varietyminweight_dog': varietyminweight_dog
        }

    def rate_process(self):
        """
        It reads in two csv files, appends them together, and then calculates the disease rate for each
        disease. 
        
        The disease rate is a dictionary of dictionaries. The first dictionary is keyed by disease name,
        and the second dictionary is keyed by variety name. The value of the second dictionary is the
        disease rate for that variety. 
        
        The disease rate is calculated by taking the average of the disease rate for each variety. 
        :return: A dictionary of disease rates.
        """
        if not os.path.exists(self.cfg.HEALTHY.DATA.DISEASE_RATE):
            cols = [
                'variety', 'disease', 'varietyavgrate', 'weightdiseaserate',
                'diseaseweightrate'
            ]
            df1 = pd.read_csv(self.cfg.HEALTHY.DATA.CAT_DISEASE,
                              header=0,
                              names=cols)
            df2 = pd.read_csv(self.cfg.HEALTHY.DATA.CAT_DISEASE,
                              header=0,
                              names=cols)
            df = df1.append(df2)
            diseases = set(df['disease'].values.tolist())
            disease_rate = {}
            for name in diseases:
                df_dis = df[df.disease == name]
                variety_rate = self.getvariety_rate(df_dis)
                disease_rate[name] = variety_rate
            json.dump(disease_rate,
                      open(self.cfg.HEALTHY.DATA.DISEASE_RATE, 'w'))
        else:
            disease_rate = json.load(open(self.cfg.HEALTHY.DATA.DISEASE_RATE))
        return disease_rate

    def disease_process(self):
        """
        > The function reads in the disease data, and then creates a dictionary of disease names and a
        dictionary of disease counts
        """
        if not os.path.exists(
                self.cfg.HEALTHY.DATA.DISEASE_COUNT) or not os.path.exists(
                    self.cfg.HEALTHY.DATA.DISEASE_DICT) or not os.path.exists(
                        self.cfg.HEALTHY.DATA.DEL_DISEASE):
            df2 = pd.read_excel(self.cfg.HEALTHY.DATA.DISEASE,
                                sheet_name='Sheet2',
                                header=0,
                                names=['name', 'relation', 'disease'])

            disnames = df2['disease'].values.tolist()
            df1 = pd.read_excel(self.cfg.HEALTHY.DATA.DISEASE,
                                sheet_name='Sheet3',
                                header=0,
                                names=['name', 'ins', 'count', 'del'])

            df3 = df1[df1['del'] == -1]
            df1 = df1[df1['ins'].notnull()]
            deldisease = set(df3['name'].values.tolist())
            disease_namedict = {}
            disease_countdict = {}
            for index, row in df1.iterrows():
                ins = row['ins']
                name = row['name']
                if ins in disnames:
                    if ins not in disease_namedict:
                        disease_namedict[ins] = name
                    else:
                        disease_namedict[ins] += ',' + name
                    count = int(row['count'])
                    if ins not in disease_countdict:
                        disease_countdict[ins] = count
                    else:
                        disease_countdict[ins] += count
            disease_countdictsort = sorted(disease_countdict.items(),
                                           key=lambda item: item[1],
                                           reverse=True)

            json.dump(disease_namedict,
                      open(self.cfg.HEALTHY.DATA.DISEASE_DICT, 'w'))
            json.dump(disease_countdictsort,
                      open(self.cfg.HEALTHY.DATA.DISEASE_COUNT, 'w'))

            json.dump(list(deldisease),
                      open(self.cfg.HEALTHY.DATA.DEL_DISEASE, 'w'))
        else:
            disease_namedict = json.load(
                open(self.cfg.HEALTHY.DATA.DISEASE_DICT))
            disease_countdictsort = json.load(
                open(self.cfg.HEALTHY.DATA.DISEASE_COUNT))
            deldisease = json.load(open(self.cfg.HEALTHY.DATA.DEL_DISEASE))
        return disease_namedict, disease_countdictsort, deldisease

    def getvariety_rate(self, df):
        '''
        获取品种和比率的字典
        :param df:
        :return:
        '''
        variety = df['variety'].values.tolist()
        rate = df['varietyavgrate'].values.tolist()
        return dict(zip(variety, rate))

    def weight_process(self):
        """
        If the files don't exist, create them. If they do exist, load them
        """
        if not os.path.exists(self.cfg.HEALTHY.DATA.CAT_WEIGHT) or \
            not os.path.exists(self.cfg.HEALTHY.DATA.DOG_WEIGHT) or \
            not os.path.exists(self.cfg.HEALTHY.DATA.CAT_MAX_WEIGHT) or \
            not os.path.exists(self.cfg.HEALTHY.DATA.DOG_MAX_WEIGHT) or \
            not os.path.exists(self.cfg.HEALTHY.DATA.CAT_MIN_WEIGHT) or \
            not os.path.exists(self.cfg.HEALTHY.DATA.DOG_MIN_WEIGHT) :
            varcatcat, variety_maxweight_cat, varietyminweight_cat = self.getvarietyweight(
                '猫', self.cfg.HEALTHY.DATA.CAT_INFO)
            varcatdog, variety_maxweight_dog, varietyminweight_dog = self.getvarietyweight(
                '犬', self.cfg.HEALTHY.DATA.CAT_INFO)
            json.dump(varcatcat, open(self.cfg.HEALTHY.DATA.CAT_WEIGHT, 'w'))
            json.dump(varcatdog, open(self.cfg.HEALTHY.DATA.DOG_WEIGHT, 'w'))
            json.dump(variety_maxweight_cat,
                      open(self.cfg.HEALTHY.DATA.CAT_MAX_WEIGHT, 'w'))
            json.dump(variety_maxweight_dog,
                      open(self.cfg.HEALTHY.DATA.DOG_MAX_WEIGHT, 'w'))
            json.dump(varietyminweight_cat,
                      open(self.cfg.HEALTHY.DATA.CAT_MIN_WEIGHT, 'w'))
            json.dump(
                varietyminweight_dog,
                open(self.cfg.HEALTHY.DATA.DOG_MIN_WEIGHT, 'w'),
            )
        else:
            varcatcat = json.load(open(self.cfg.HEALTHY.DATA.CAT_WEIGHT))
            varcatdog = json.load(open(self.cfg.HEALTHY.DATA.DOG_WEIGHT))
            variety_maxweight_cat = json.load(
                open(self.cfg.HEALTHY.DATA.CAT_MAX_WEIGHT))
            variety_maxweight_dog = json.load(
                open(self.cfg.HEALTHY.DATA.DOG_MAX_WEIGHT))
            varietyminweight_cat = json.load(
                open(self.cfg.HEALTHY.DATA.CAT_MIN_WEIGHT))
            varietyminweight_dog = json.load(
                open(self.cfg.HEALTHY.DATA.DOG_MIN_WEIGHT))
        return varcatcat, variety_maxweight_cat, varietyminweight_cat, varcatdog, variety_maxweight_dog, varietyminweight_dog

    def getvarietyweight(self, kindof, fi):
        '''
        计算每个品种的体重上限，返回kv
        :param kindof:
        :return:
        '''
        cols = [
            'name', 'chinese_alias', 'male_height', 'male_weight',
            'female_height', 'female_weight'
        ]

        if kindof == '犬':
            path = self.cfg.HEALTHY.DATA.DOG
        else:
            path = self.cfg.HEALTHY.DATA.CAT
        fr = open(path, 'r')
        file_data = fr.readlines()
        catordogset = []
        for x in file_data:
            x = x.replace(kindof, '').replace('\n', '') + kindof
            if x not in ['其他' + kindof, '其它' + kindof, '未知' + kindof, kindof]:
                catordogset.append(x)
        dfv = pd.read_csv(fi, header=0, names=cols)
        varcatdog = {}
        varietymaxweight = {}
        varietyminweight = {}
        for index, row in dfv.iterrows():
            name = row['name'].replace(kindof, '') + kindof
            chinese_alias = str(row['chinese_alias'])
            allname = name + '|' + chinese_alias
            male_weight = str(row['male_weight'])
            female_weight = str(row['female_weight'])
            allnames = set(allname.split('|'))
            intercat = set(catordogset).intersection(allnames)
            for cx in intercat:
                male_max = 0
                formale_max = 0
                male_min = 0
                formale_min = 0
                if '|' in male_weight:
                    male_max = male_weight.split('|')[1]
                    male_min = male_weight.split('|')[0]
                if '|' in female_weight:
                    formale_max = female_weight.split('|')[1]
                    formale_min = female_weight.split('|')[0]
                varcatdog[cx] = f'{male_weight}:{female_weight}'
                varietymaxweight[cx] = max(male_max, formale_max)
                varietyminweight[cx] = min(male_min, formale_min)
        return varcatdog, varietymaxweight, varietyminweight


# The above code is used to preprocess the data.
@dataclass
class TrainUtil:
    cfg: CfgNode = field(default_factory=get_cfg)

    def __call__(self, disease, disease_dict, disease_count,
                 variety_maxweight_cat, varietyminweight_cat,
                 variety_maxweight_dog, varietyminweight_dog):
        """
        This function takes in the disease, disease_dict, disease_count, variety_maxweight_cat,
        varietyminweight_cat, variety_maxweight_dog, varietyminweight_dog and returns the
        train_data_process function
        
        :param disease: the disease name
        :param disease_dict: a dictionary of diseases and their corresponding symptoms
        :param disease_count: the number of diseases in the dataset
        :param variety_maxweight_cat: the maximum weight of the cat
        :param varietyminweight_cat: The minimum weight of the cat
        :param variety_maxweight_dog: the maximum weight of the dog breed
        :param varietyminweight_dog: The minimum weight of the dog
        :return: The train_data_process function is being returned.
        """
        return self.train_data_process(disease, disease_dict, disease_count,
                                       variety_maxweight_cat,
                                       varietyminweight_cat,
                                       variety_maxweight_dog,
                                       varietyminweight_dog)

    def train_data_process(self, disease, disease_dict, disease_count,variety_maxweight_cat, varietyminweight_cat,
                           variety_maxweight_dog, varietyminweight_dog):
        """
        It reads the raw data, filters the data by disease name, and then preprocesses the data
        
        :param disease: the name of the disease
        :param disease_dict: a dictionary of diseases and their corresponding names
        :param disease_count: the number of diseases
        :param variety_maxweight_cat: the maximum weight of the cat
        :param varietyminweight_cat: the minimum weight of the cat
        :param variety_maxweight_dog: the maximum weight of the dog
        :param varietyminweight_dog: the minimum weight of the dog
        :return: df
        """
        if not os.path.exists(
                os.path.join(self.cfg.HEALTHY.DATA.TRAIN,
                             disease + '_processed.csv')):

            raw = pd.read_csv(
                self.cfg.HEALTHY.DATA.TRAIN_RAW,
                header=0,
                names=[
                    'weight', 'temperature', 'breathing_rate', 'heart_rate',
                    'id', 'dt', 'orgid', 'region', 'area', 'province', 'city',
                    'brand', 'brand_code', 'clinic_name', 'magnitude',
                    'org_scrm_id', 'cus_id', 'cus_name', 'cus_memberno',
                    'cus_gender', 'cus_birthday', 'cus_registration',
                    'cus_source', 'cus_curr_age', 'cus_cellphone',
                    'cus_card_type', 'cus_scrm_id', 'cus_is_nonu', 'pet_id',
                    'pet_name', 'pet_age', 'pet_gender', 'pet_birthday',
                    'pet_weight', 'pet_number', 'pet_kindof', 'pet_variety',
                    'pet_registdate', 'pet_ensure_code', 'pet_face_id',
                    'pet_scrm_id', 'pet_is_nonu', 'regid', 'vir_regid',
                    'record_type_reg', 'record_type_med', 'night_tag_reg',
                    'night_tag_med', 'reg_category', 'reservation_id',
                    'reservation_doctor', 'reservation_doctor_name',
                    'reservation_date', 'order_doctor', 'order_doctor_name',
                    'started_date', 'finished_date', 'process_state',
                    'item_id', 'item_name', 'item_amount', 'is_payed',
                    'reg_time', 'is_first_reg', 'p_payid', 'handle',
                    'source_id', 'clinic_reg_id', 'med_type', 'cem_type',
                    'record_type', 'record_number', 'department', 'cure_room',
                    'physician_id', 'physician_name', 'bar_code',
                    'chief_complaint', 'present_history', 'illness_situation',
                    'past_history', 'physical_level', 'physical_description',
                    'symptom_level1', 'symptom_level2', 'symptom_level3',
                    'symptom_code', 'symptom_category', 'treatment_opinion',
                    'doctor_advice', 'med_status', 'med_result', 'remark',
                    'reg_desc', 'start_time', 'operator_id', 'operator_name',
                    'end_time', 'create_time', 'update_time', 'med_level',
                    'night_tag', 'finish_type', 'departmentname',
                    'min_start_time', 'max_end_time'
                ])

            raw = raw[raw.city.notnull()]
            raw = raw[raw.pet_variety.notnull()]
            raw = raw[raw.pet_kindof.str.contains('犬|猫')]

            diseasenames = disease_dict[disease]

            namescontain = diseasenames.replace(',', '|')
            self.getfilterdata(raw, namescontain, disease)
            df = pd.read_csv(
                os.path.join(self.cfg.HEALTHY.DATA.TRAIN, disease + '.csv'),
                header=0,
                names=[
                    'y', 'weights', 'temperatures', 'breathing_rates',
                    'heart_rates', 'city', 'pet_age', 'pet_gender',
                    'pet_birthday', 'pet_weight', 'pet_kindof', 'pet_variety',
                    'record_type_reg', 'reg_time', 'chief_complaint',
                    'symptom_level3', 'doctor_advice'
                ])

            if (len(df) > 1):
                df = self.data_preprocess(df, variety_maxweight_cat,
                                          varietyminweight_cat,
                                          variety_maxweight_dog,
                                          varietyminweight_dog)
                df.to_csv(os.path.join(self.cfg.HEALTHY.DATA.TRAIN,
                                       disease + '_processed.csv'),
                          index=False)
        else:
            df = pd.read_csv(
                os.path.join(self.cfg.HEALTHY.DATA.TRAIN,
                             disease + '_processed.csv'))
        return df

    def data_preprocess(self, df, variety_maxweight_cat, varietyminweight_cat,
                        variety_maxweight_dog, varietyminweight_dog):
        """
        The function takes in a dataframe, and a few other parameters, and returns a dataframe with the
        following columns:
        
        - y: the target variable
        - weights: the weight of the pet
        - temperatures: the temperature of the pet
        - heart_rates: the heart rate of the pet
        - breathing_rates: the breathing rate of the pet
        - pet_age: the age of the pet
        - isadult: whether the pet is an adult
        - dummies_kindof: the kind of pet (cat or dog)
        - dummies_petgender: the gender of the pet
        - dummies_variety: the variety of the pet
        - dummies_city: the city of the pet
        - isbigcity: whether the city is a big city
        - isold: whether the pet is old
        - isoverweight: whether the pet is overweight
        - isshortageweight: whether the pet
        
        :param df: the dataframe to be processed
        :param variety_maxweight_cat: the maximum weight of each cat breed
        :param varietyminweight_cat: the minimum weight of the cat variety
        :param variety_maxweight_dog: the maximum weight of each dog breed
        :param varietyminweight_dog: the minimum weight of the dog
        """

        bigcity = ['北京', '上海', '广州', '深圳']

        secondcity = [
            '成都', '重庆', '杭州', '武汉', '西安', '天津', '苏州', '南京', '郑州', '长沙', '东莞',
            '沈阳', '青岛', '合肥', '佛山'
        ]

        df = df[df.pet_kindof.notnull()]
        df = df[df.pet_kindof.str.contains('犬|猫')]

        df['isadult'] = np.where(df.pet_age > 12, 1, 0)
        df['isold'] = np.where(df.pet_age > 96, 1, 0)

        df['isoverweight'] = df.apply(lambda x: self.process_varietyoverweight(
            x['weights'], x['pet_variety'], x['pet_kindof'],
            variety_maxweight_cat, variety_maxweight_dog),
                                      axis=1)
        df['isshortageweight'] = df.apply(
            lambda x: self.process_varietyshortageweight(
                x['weights'], x['pet_variety'], x['pet_kindof'],
                varietyminweight_cat, varietyminweight_dog),
            axis=1)
        df['isauditshortageweight'] = df.apply(
            lambda x: self.process_auditvarietyshortageweight(
                x['weights'], x['pet_variety'], x['isadult'], x['pet_kindof'],
                varietyminweight_cat, varietyminweight_dog),
            axis=1)
        df['isnotauditoverweight'] = df.apply(
            lambda x: self.process_auditvarietyoverweight(
                x['weights'], x['pet_variety'], x['isadult'], x['pet_kindof'],
                variety_maxweight_cat, variety_maxweight_dog),
            axis=1)
        df['isbigcity'] = df['city'].apply(
            lambda x: self.processcity(str(x), bigcity, secondcity))

        df['isother'] = df['symptom_level3'].apply(
            lambda x: self.processcomplication(str(x), ''))

        dummies_kindof = pd.get_dummies(df.pet_kindof, prefix='kindofs')
        dummies_petgender = pd.get_dummies(df.pet_gender, prefix='petgenders')
        dummies_variety = pd.get_dummies(df.pet_variety, prefix='varietys')
        dummies_city = pd.get_dummies(df.city, prefix='citys')
        df = pd.concat(
            [
                df.y,
                df.weights,
                df.temperatures,
                df.heart_rates,
                df.breathing_rates,
                df.pet_age,
                df.isadult,
                dummies_kindof,
                dummies_petgender,
                dummies_variety,
                dummies_city,
                df.isbigcity,
                df.isold
                # ,df.isother
                ,
                df.isoverweight,
                df.isshortageweight,
                df.isnotauditoverweight,
                df.isauditshortageweight
                # ,df.variety_rate
            ],
            axis=1)
        return df

    def getfilterdata(self, df, diseasenames, diseasename):
        '''
        过滤下时间和并且使用筛选的疾病字典中的疾病筛选
        :param diseasename:
        :param d2:
        :return:
        '''
        profromat = ProcessNumberformat()
        df = df.fillna('')
        df_gly = df[df.symptom_level3.str.contains(diseasenames)]
        df_gly['y'] = 1

        pet_ids = set(df_gly['pet_id'].values.tolist())

        # 用来筛选未得该病之前的病例数据
        df_notgly = df
        df_notgly = df_notgly.append(df_gly)
        df_notgly = df_notgly.drop_duplicates(subset=df.columns, keep=False)
        # df1.to_csv('outputpredict/11/t2.csv')
        df_notgly['isnotgly'] = df_notgly['pet_id'].apply(
            lambda x: self.getleavedata(x, pet_ids))
        df_notglyone = df_notgly[df_notgly.isnotgly == 1]
        df_notglyzero = df_notgly[df_notgly.isnotgly == 0]

        # df_notgly.to_csv('outputpredict/11/t4.csv')

        df_notglyone.drop(['isnotgly'], axis=1, inplace=True)
        df_gly['y'] = 1
        df_notglyone['y'] = 0
        # print(len(df_notglyone))
        # print(len(df_gly))
        count = max(len(df_notglyone), len(df_gly))

        df_new = df_notglyone.append(df_gly)
        df_new = df_new[[
            'y', 'weight', 'temperature', 'breathing_rate', 'heart_rate',
            'city', 'pet_age', 'pet_gender', 'pet_birthday', 'pet_weight',
            'pet_kindof', 'pet_variety', 'record_type_reg', 'reg_time',
            'chief_complaint', 'symptom_level3', 'doctor_advice'
        ]]

        # df_new.to_csv(self.outpath + 'predict/is_dataall' + diseasename +
        #               '.csv')

        if len(df_notglyone) < count:
            # 如果负例少，就从所有负例中找出健康的数据作为新的负例
            samnum = count - len(df_notglyone)
            df_notglyzero = df_notglyzero[
                df_notglyzero.symptom_level3.str.contains('体检|驱虫|免疫')]
            df_notglyzero['y'] = 0
            df_notglyzero = df_notglyzero.sample(n=samnum)
            dftrain = df_notglyone.append(df_gly)
            dftrain = dftrain.append(df_notglyzero)
        else:
            # 如果正例少的话，正例直接上采样
            df_gly = df_gly.sample(n=count, replace=True)
            dftrain = df_notglyone.append(df_gly)

        # df_notglyone=df_notglyone.sample(n=count,replace=True)
        # df_gly=df_gly.sample(n=count,replace=True)
        # dftrain=df_notglyone.append(df_gly)

        dftrain['weights'] = dftrain['weight'].apply(
            lambda x: profromat.processphy(str(x), 0, 'weight'))
        dftrain['temperatures'] = dftrain['temperature'].apply(
            lambda x: profromat.processphy(str(x), 38.5, 'temperature'))
        dftrain['heart_rates'] = dftrain['heart_rate'].apply(
            lambda x: profromat.processphy(str(x), 120, 'heart_rate'))
        dftrain['breathing_rates'] = dftrain['breathing_rate'].apply(
            lambda x: profromat.processphy(str(x), 30, 'breathing_rate'))
        dftrain['pet_age'] = dftrain['pet_age'].apply(
            lambda x: profromat.processphy(str(x), 0, 'pet_age'))  # 这里表示的是月份

        dftrain = dftrain[[
            'y', 'weights', 'temperatures', 'breathing_rates', 'heart_rates',
            'city', 'pet_age', 'pet_gender', 'pet_birthday', 'pet_weight',
            'pet_kindof', 'pet_variety', 'record_type_reg', 'reg_time',
            'chief_complaint', 'symptom_level3', 'doctor_advice'
        ]]
        dftrain.to_csv(os.path.join(self.cfg.HEALTHY.DATA.TRAIN,
                                    diseasename + '.csv'),
                       index=False)

    def process_auditvarietyshortageweight(self, xweight, xvariety, xaudit,
                                           xkindof, minweightcat,
                                           minweightdog):
        '''
        判断该品种的体重是否低于下线
        :param xweight:
        :param xvariety:
        :param minweights:
        :return:
        '''
        minweights = minweightdog if xkindof == '犬' else minweightcat
        if xaudit < 12:
            return 3
        if xvariety in minweights:
            return 1 if xweight < float(minweights[xvariety]) else 0
        else:
            return 2

    def process_varietyoverweight(self, xweight, xvariety, xkindof,
                                  maxweightscat, maxweightsdogs):
        '''
        判断该品种的体重是否超过上限
        :param xweight:
        :param xvariety:
        :param maxweights:
        :return:
        '''
        maxweights = maxweightsdogs if xkindof == '犬' else maxweightscat
        if xvariety in maxweights:
            maxweight = float(maxweights[xvariety])
            return 1 if xweight > maxweight else 0
        else:
            return 2

    def process_varietyshortageweight(self, xweight, xvariety, xkindof,
                                      minweightcat, minweightdog):
        '''
        判断该品种的体重是否低于下线
        :param xweight:
        :param xvariety:
        :param minweights:
        :return:
        '''
        minweights = minweightdog if xkindof == '犬' else minweightcat
        if xvariety in minweights:
            minweight = float(minweights[xvariety])
            return 1 if xweight < minweight else 0
        else:
            return 2

    def process_auditvarietyoverweight(self, xweight, xvariety, xaudit,
                                       xkindof, maxweightcat, maxweightdog):
        '''
        判断该品种的体重是否高于上线
        :param xweight:
        :param xvariety:
        :param minweights:
        :return:
        '''
        maxweights = maxweightdog if xkindof == '犬' else maxweightcat
        if xaudit >= 12:
            return 3
        if xvariety in maxweights:
            return 1 if xweight > float(maxweights[xvariety]) else 0
        else:
            return 2

    def processcity(self, x, bigcity, secondcity):
        x = x.replace('市', '')
        if x in bigcity:
            return 1
        elif x in secondcity:
            return 2
        else:
            return 0

    def processcomplication(self, x, name):
        '''
        是否有并发疾病 胰腺炎、肾衰
        :param x:
        :param name:
        :return:
        '''
        names = name.split('|')
        return next((1 for na in names if na in x), 0)

    def getleavedata(self, x, pet_ids):
        '''
        确定数据是否保留,完全匹配，只包括了11类疾病，不包括其他疾病
        :param x:
        :param diseasename:
        :return:
        '''
        return 1 if x in pet_ids else 0