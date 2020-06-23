# coding: utf-8
import os
import sys
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from jieba import posseg as pseg
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
from sklearn import metrics
import time

root_dir = os.path.dirname(os.path.split(os.path.realpath(sys.argv[0]))[0])

# 建立数据处理函数,进行分词,过滤
jieba.load_userdict(os.path.join(root_dir, r'用户词典/泰迪杯地名词典ns.txt'))  # 加载自定义词典
stopwords = [w.strip() for w in open(os.path.join(root_dir, 'stopword.txt'), 'r', encoding='gbk')]  # 读取停用词
stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']  # 词性过滤列表


def cut_words(text):
    words = pseg.cut(text)
    result = []
    #     过滤
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    result_last = ' '.join(result)
    return result_last


def run_interpretability():
    print("开始计算答复的可解释性...")
    t = time.time()

    # 读取数据
    data = pd.read_excel(os.path.join(root_dir, 'T3/附件4_人工标注.xlsx'), dtype=str)

    # 提取数据
    X = data[['答复意见']]
    Y = data['可解释性']

    # 数据切词，去除停用词
    X['答复切词'] = X['答复意见'].apply(cut_words)

    # 拆分数据,一部分训练,一部分用于测试
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    # 提取特征,过滤不必要的词语,转换词频矩阵
    vect = CountVectorizer(max_df=0.8, min_df=3, token_pattern=u"(?u)\\b\\w+\\b", stop_words=frozenset(stopwords),
                           analyzer='word')
    # 统计出现次数
    matrix = vect.fit_transform(X_train['答复切词'])

    # 建立朴素贝叶斯分类器
    nb = MultinomialNB()

    # 建立一个导管,将分类器和向量化函数联系一起
    pipe = make_pipeline(vect, nb)

    # 拟合训练
    pipe.fit(X_train['答复切词'], Y_train)

    # 预测结果
    Y_pred = pipe.predict(X_test['答复切词'])

    # 查看分数
    print("解释分类模型的f1_score: %0.3f " % f1_score(Y_test, Y_pred, average='micro'))

    # 查看混淆矩阵
    metrics.confusion_matrix(Y_test, Y_pred)

    # 将结果保存到表中
    y_pred = pipe.predict(X['答复切词'])
    y_pred_transfrom = np.where(y_pred.astype(float) > 0, '优', '差')
    data_1 = pd.read_excel('result_相关性.xlsx', dtype=str)
    data_1.insert(8, '评价可解释性', y_pred_transfrom)
    path = os.path.join(root_dir, r"结果数据/答复意见评价.xlsx")
    data_1.to_excel(path, index=False, sheet_name='result')
    print("答复意见评价完成，用时%f，结果文件保存在%s" % ((time.time()-t), path))


if __name__ == '__main__':
    run_interpretability()
