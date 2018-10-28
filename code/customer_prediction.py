import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import random
import csv
import os
import pandas as pd

subspace_num = 20
tree_number = 40


# 预测结果文件：src/step1/ground_truth/test_prediction.csv

def genera_particle(sample):
    particle_set = []
    for j in range(0, subspace_num):  # 每个set里面10个粒子
        particle = []
        for k in range(0, len(sample)):  # 每个粒子维度为样本数
            if (np.random.random()) > 0.6:
                particle.append(1);
            else:
                particle.append(0);
        particle_set.append(particle);
    particle_set = np.array(particle_set)
    return particle_set

def direct_ensemble(sample, particle_set=None, flag=False):  # sample是一个集合 10
    classifier_set = []
    weight_vector = np.array([])
    for k in range(0, subspace_num):
        if flag is False:
            tmp = np.array(sample[k])
        if flag is True:
            tmp = np.array(sample)
        if particle_set is None:
            sample_train_selected = tmp
        else:
            sample_train_selected = tmp[particle_set[k, :] == 1]
        # sample_test_selected = sample[best_set[k, :] == 0]  # 对未被选中的样本进行预测
        sample_test = tmp[particle_set[k, :] == 0]
        clf = RandomForestClassifier(n_estimators=tree_number)
        clf.fit(sample_train_selected[:, 0:-1], sample_train_selected[:, -1])
        pred_y = clf.predict_proba(sample_test[:, 0:-1])
        prob = [p[1] for p in pred_y]
        score = roc_auc_score(sample_test[:, -1], prob)
        weight_vector = np.append(weight_vector, score)
        classifier_set.append(clf)
    weight_vector = np.divide(weight_vector, np.sum(weight_vector))
    return weight_vector, classifier_set


def vote(testing_sign, weight_vector, classifier_set, sample):
    result = []
    for i in range(0, len(classifier_set)):
        subspace_idx = testing_sign[i]
        sample_sub = sample[:, subspace_idx]
        pre = classifier_set[i].predict_proba(sample_sub)
        prob = [p[1] * weight_vector[i] for p in pre]
        result.append(prob)
    result = np.array(result)
    result = np.sum(result, axis=0)

    return result


def getPrediction():
    # ********* Begin *********#
    train = pd.read_csv("src/step1/input/train.csv")
    test = pd.read_csv("src/step1/input/test.csv")
    #train = pd.read_csv("train.csv")
    #test = pd.read_csv("test.csv")
    train_sample = np.array(train.values)
    test_sample = np.array(test.values)
    x_id = test['ID']
    testing_sign = []
    for i in range(0, subspace_num):
        testing_sign.append([i for i in range(0, train_sample.shape[1] - 1)])
    testing_sign = np.array(testing_sign)
    particle_set = genera_particle(train_sample)
    weight_vector, classifier_set = direct_ensemble(
        train_sample, particle_set, flag=True)
    y_pre_array = vote(testing_sign, weight_vector, classifier_set,
                       test_sample)
  
    result=pd.DataFrame({'ID':np.array(x_id).reshape(300,),'TARGET':y_pre_array})
    result.to_csv("src/step1/ground_truth/test_prediction.csv",index=False,sep=',',encoding="utf-8")
    # ********* End *********#
