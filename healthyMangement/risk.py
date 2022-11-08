from healthyMangement.modelUtil import XgboostUtil
from healthyMangement.dataProcess import DiseaseUtil, TrainUtil
from config import get_cfg, CfgNode
from dataclasses import dataclass, field
import pandas as pd
import os
import numpy as np


@dataclass
class Risk:
    cfg: CfgNode = field(default_factory=get_cfg)
    xgboost = XgboostUtil()
    disease_util = DiseaseUtil()
    train = TrainUtil()
    models: dict = field(default_factory=xgboost)
    mapping: dict = field(default_factory=disease_util)

    def __call__(self, data: dict) -> dict:
        """
        It takes in a dictionary of data, preprocesses it, and then returns a list of dictionaries
        containing the disease name, the probability of the disease, and the prediction of the disease.
        
        :param data: a dictionary of the data to be predicted
        :type data: dict
        :return: A list of dictionaries, each dictionary contains the name of the disease, the
        probability of the disease and the prediction of the disease.
        """
        result = []

        data = pd.DataFrame([data])
        data['y'] = 0
        data = self.train.data_preprocess(
            data, self.mapping['variety_maxweight_cat'],
            self.mapping['varietyminweight_cat'],
            self.mapping['variety_maxweight_dog'],
            self.mapping['varietyminweight_dog'])

        for disease, _ in self.xgboost.disease_py.items():
            cols = [
                x for x in pd.read_csv(
                    os.path.join(self.cfg.HEALTHY.DATA.TRAIN, disease +
                                 '_processed.csv')).columns if x not in ['y']
            ]

            for col in cols:
                if col not in data.columns and col != 'index':
                    data[col] = 0
            data = data[cols]

            tmp = {}
            tmp['name'] = disease
            tmp['proba'] = self.models[disease].predict_proba(data)
            tmp['prediction'] = np.argmax(tmp['proba'])
            result.append(tmp)
        return result


if __name__ == '__main__':
    risk = Risk()
    data = {
        'weights': 20.3,
        'temperatures': 0.1,
        'heart_rates': 0.1,
        'breathing_rates': 122.0,
        'pet_age': 1,
        'pet_gender': '公',
        'pet_kindof': '猫',
        'pet_variety': '中华田园猫',
        'city': '武汉市',
        'symptom_level3': ''
    }
    print(risk(data))