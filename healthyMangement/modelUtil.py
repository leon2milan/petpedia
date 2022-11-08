'''
xgboost做分类模型
'''
import datetime
import os
import warnings
from dataclasses import dataclass, field
from os import path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config import get_cfg, CfgNode
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from healthyMangement.dataProcess import DiseaseUtil, TrainUtil

warnings.filterwarnings('ignore')
# 解决中文画图乱码的问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


class Xgboost():

    def __init__(self, cfg):
        self.cfg = cfg

    def trian(self, df, fomodel, modelname, classname):
        '''
        训练模型，并测试结果
        :param df: 数据
        :param fomodel: 模型输出的路径
        :param modelname: 模型的名字
        :param classname: 分类标签名字
        :return:
        '''
        print(fomodel)
        name = fomodel + modelname

        y = df['y'].values
        df.drop(['y'], axis=1, inplace=True)
        X = df.values
        print(df.columns.values.tolist())

        print('=============xgboost=============')
        X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.25,
                                                            random_state=0)

        params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',  # 多分类的问题
            'gamma': 0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
            'max_depth': 12,  # 构建树的深度，越大越容易过拟合
            'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
            'subsample': 0.7,  # 随机采样训练样本
            'colsample_bytree': 0.7,  # 生成树时进行的列采样
            'min_child_weight': 3,
            # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
            # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
            # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
            'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
            'eta': 0.001,  # 如同学习率
            'seed': 1000,
            'nthread': 7,  # cpu 线程数
            # 'scale_pos_weigh':0.26
            # 'eval_metric': 'auc'
        }

        xgb_train = xgb.DMatrix(X_train, label=Y_train)
        xgb_test = xgb.DMatrix(X_test, label=Y_test)

        plst = list(params.items())
        num_rounds = 30  # 迭代次数
        watchlist = [(xgb_train, 'train'), (xgb_test, 'val')]
        model = xgb.train(plst,
                          xgb_train,
                          num_rounds,
                          watchlist,
                          early_stopping_rounds=10)
        print(name)
        model.save_model(name + '.model')

        # modl = xgb_model
        #
        end0 = datetime.datetime.now()
        y_pred = model.predict(xgb_test)
        end1 = datetime.datetime.now()
        # y_pred = model.predict(X_test)
        print("================================", end1 - end0)
        # print(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(Y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        precision = precision_score(Y_test, predictions, average='macro')
        recall = recall_score(Y_test, predictions, average='macro')

        print("精确率: %.2f%%" % (precision * 100.0))
        print("召回率: %.2f%%" % (recall * 100.0))

        f1 = f1_score(Y_test, predictions, average='weighted')
        print("f1 %.2f%%" % (f1 * 100.0))

        # 混淆矩阵
        nn = confusion_matrix(Y_test, predictions)
        print("混淆矩阵", nn)

        # 分类报告：precision / recall / fi - score / 均值 / 分类个数
        # target_names = ['class 0', 'class 1','class 2']
        target_names = classname
        print(
            classification_report(Y_test,
                                  predictions,
                                  target_names=target_names))

        # # 把特征名该为自己的，而不是用默认的f1,f2之类的
        # def ceate_feature_map(features):
        #     outfile = open(name+'.fmap', 'w')
        #     i = 0
        #     for feat in features:
        #         # print(feat)
        #         # feat = feat.replace('ml','').replace('mg','').replace('g','').replace('cm','')
        #         # feat = feat.replace('：','').replace(' ', '').replace('#', '').replace('（','').replace('）','')
        #         outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        #         i = i + 1
        #     outfile.close()

        # print(df_train.columns)
        # print(len(df_train.columns))
        # ceate_feature_map(df.columns)
        # plot_tree(model, fmap=name+'.fmap')
        # plt.savefig(name+'leaf')

        # fig, ax = plt.subplots(figsize=(15, 15))
        # plot_importance(model,
        #                 height=0.5,
        #                 ax=ax,
        #                 title='特征重要性',
        #                 fmap=name+'.fmap',
        #                 max_num_features=64)
        # plt.savefig(name+'特征重要性')

        # graph = xgb.to_graphviz(model, fmap=name+'.fmap', feature_names=df.columns, num_trees=0,
        #                         **{'size': str(10)})
        # graph.render(filename=name+'.pdf')

        # dot_data = export_graphviz(model,
        #                            out_file=None,
        #                            feature_names=df.columns,
        #                            class_names=classname,
        #                            )
        #
        # graph = graphviz.Source(dot_data)
        # graph.render('outputpredict/11disease_' + numcalss + kindof)  # 把图保存到当前的目录下面
        return model, accuracy, precision, recall, f1

@dataclass
class XgboostUtil:
    cfg: CfgNode = field(default_factory=get_cfg)
    disease_util = DiseaseUtil()
    mapping: dict = field(default_factory=disease_util)
        
    def __call__(self):
        """
        A function that train the model.
        :return: A dictionary of models.
        """
        disease_py = [('猫下泌尿道综合征', 'mxmndzhz'), ('胰腺炎', 'yxy'),
                      ('肾功能衰竭', 'sgnsj'), ('心肌病', 'xjb'), ('关节炎', 'gjy'),
                      ('糖尿病', 'tnb'), ('椎间盘疾病，腰椎', 'zjpjb'), ('肾脏疾病', 'szjb'),
                      ('全身炎症反应综合征（SIRS）', 'qsyzfzzhz'), ('巴贝斯虫病', 'bbscb'),
                      ('肝平滑肌瘤', 'jphjl'), ('肾上腺皮质机能亢进', 'ssxpzjnkj')]
        self.disease_py = {x[0]: x[1] for x in disease_py}

        models = {}
        data = []
        print(self.mapping['disease_count'])
        for item in self.mapping['disease_count']:
            disease = item[0]
            count = item[1]

            if count > self.cfg.HEALTHY.MODEL.DATA_THRESHOLD:
                model_path = os.path.join(self.cfg.HEALTHY.MODEL.PATH,
                                          self.disease_py[disease] + '.model')
                model = Xgboost(self.cfg)

                from healthyMangement.numberUtil import ProcessNumberformat
                numberformat = ProcessNumberformat()
                if not os.path.exists(model_path):

                    train_util = TrainUtil(self.cfg)
                    df = train_util(disease, self.mapping['disease_dict'],
                                    self.mapping['disease_count'],
                                    self.mapping['variety_maxweight_cat'],
                                    self.mapping['varietyminweight_cat'],
                                    self.mapping['variety_maxweight_dog'],
                                    self.mapping['varietyminweight_dog'])

                    model, accuracy, precision, recall, f1 = model.trian(df, self.cfg.HEALTHY.MODEL.PATH, self.disease_py[disease],
                                           ['不是' + disease, '是' + disease])
                    data.append([
                        disease, count,
                        numberformat.getpetcent(accuracy),
                        numberformat.getpetcent(precision),
                        numberformat.getpetcent(recall),
                        numberformat.getpetcent(f1)
                    ])
                    dfnew = pd.DataFrame(
                        np.array(data),
                        columns=['name', 'count', 'accuracy', 'precision', 'recall', 'f1'])
                    dfnew.to_csv(self.cfg.HEALTHY.DATA.HEALTHY + 'output.csv')
                else:
                    model = self.load_model(model_path)
                models[disease] = model
        return models

    def load_model(self, model_path):
        """
        It loads the model from the path specified by the user
        
        :param model_path: The path to the saved model
        :return: The model is being returned.
        """
        clf = xgb.XGBClassifier()
        booster = xgb.Booster()
        booster.load_model(model_path)
        clf._Booster = booster
        return clf

if __name__ == '__main__':
    xgboost = XgboostUtil()
    xgboost()