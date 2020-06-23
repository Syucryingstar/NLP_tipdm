# coding: utf-8
import os
import time
from sklearn.svm import SVC
from sklearn.externals import joblib
import sys
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, decomposition
from sklearn.metrics import f1_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
import re

root_dir = os.path.dirname(os.path.split(os.path.realpath(sys.argv[0]))[0])
jieba.load_userdict(os.path.join(root_dir, r'用户词典/泰迪杯地名词典ns.txt'))  # 加载自定义词典


def rep(s):
    """
    去除字符中的特殊符号如空格等
    :param s:
    :return:
    """
    s = str(s)
    stop = ['\x22', '\x26', '\x3C', '\x3E', '\xA0', '\xA1', '\xA2', '\xA3', '\xA4', '\xA5', '\xA6', '\xA7', '\xA8',
            '\xA9', '\xAA', '\xAB', '\xAD', '\xAE', '\xAF', '\xB0', '\xB1', '\xB2', '\xB3', '\xB4', '\xB5', '\xB6',
            '\xB7', '\xB8', '\xB9', '\xBA', '\xBB', '\xBC', '\xBD', '\xBE', '\xBF', '\xD7', '\xF7', '\u0192', '\u02C6',
            '\u02DC', '\u2002', '\u2003', '\u2009', '\u200C', '\u200D', '\u200E', '\u200F', '\u2013', '\u2014', '\t',
            '\u2018', '\u2019', '\u201A', '\u201C', '\u201D', '\u201E', '\u2020', '\u2021', '\u2022', '\u2026', '\n',
            '\u2030', '\u2032', '\u2033', '\u2039', '\u203A', '\u203E', '\u2044', '\u20AC', '\u2111', '\u2113',
            '\u2116', '\u2118', '\u211C', '\u2122', '\u2135', '\u2190', '\u2191', '\u2193', '\u2194', '\u21B5',
            '\u21D0', '\u21D1', '\u21D2', '\u21D3', '\u21D4', '\u2200', '\u2202', '\u2203', '\u2205', '\u2207',
            '\u2208', '\u2209', '\u220B', '\u220F', '\u2211', '\u2212', '\u2217', '\u221A', '\u221D', '\u221E',
            '\u2220', '\u2227', '\u2228', '\u2229', '\u222A', '\u222B', '\u2234', '\u223C', '\u2245', '\u2248',
            '\u2260', '\u2261', '\u2264', '\u2265', '\u2282', '\u2283', '\u2284', '\u2286', '\u2287', '\u2295',
            '\u2297', '\u22A5', '\u22C5', '\u2308', '\u2309', '\u230A', '\u230B', '\u2329', '\u232A', '\u25CA',
            '\u2660', '\u2663', '\u2665', '\u2666', '\u200b', '\u3000', '\u2800', '\xa0', '\u0020', '\u00A0']
    ss = s
    for i in stop:
        ss = ss.replace(i, "")
    return ss.strip()


def process(s):  # 去除无用信息和去空函数
    s1 = rep(re.sub(r'[a-zA-Z0-9]', '', s))
    exclude = set('，。？！；：、-=+.（）《》')
    out = ''.join(ch for ch in s1 if ch not in exclude)
    return out


def run_category():  # 构建分类模型
    t = time.time()
    # 读取原始数据
    data = pd.read_excel(os.path.join(root_dir, r'全部数据/附件2.xlsx'))

    data['留言'] = (2 * data['留言主题'] + data['留言详情']).apply(lambda i: process(i))  # 创建新的属性并对属性值执行process函数

    data['text_split_list'] = data['留言'].apply(lambda i: jieba.lcut(i, cut_all=True))  # 使用全模式对文本进行切分

    data['text_split'] = [' '.join(i) for i in data['text_split_list']]

    data.head()  # 查看预处理后的数据

    lbl_enc = preprocessing.LabelEncoder()  # 将文本标签（Text Label）转化为数字(Integer)
    y = lbl_enc.fit_transform(data.一级标签.values)

    # 将DataFrame以9:1切分为训练集和验证集
    data_train, data_valid, ytrain, yvalid = train_test_split(data, y,
                                                              stratify=y,
                                                              random_state=42,
                                                              test_size=0.1, shuffle=True)

    xtrain = data_train.text_split.values  # 将训练集的text_split属性值保存在xtrain中
    xvalid = data_valid.text_split.values  # 将验证集的text_split属性值保存在ytrain中

    # 读取停用词文件，并将停用词保存到列表中
    stwlist = [line.strip() for line in
               open(os.path.join(root_dir, r'stopword.txt'), 'r', encoding='gbk').readlines()]

    # 使用TF-IDF算法将文本转化为词频矩阵
    tfv = TfidfVectorizer(min_df=3,
                          max_df=0.5,
                          max_features=None,
                          ngram_range=(1, 2),
                          use_idf=True,
                          smooth_idf=True,
                          stop_words=stwlist)

    # 使用TF-IDF来fit训练集和测试集
    tfv.fit(list(xtrain) + list(xvalid))
    xtrain_tfv = tfv.transform(xtrain)
    xvalid_tfv = tfv.transform(xvalid)

    # 使用SVD进行降维，components设为150，对于SVM来说，SVD的components的合适调整区间一般为120~200
    svd = decomposition.TruncatedSVD(n_components=150, random_state=42)
    svd.fit(xtrain_tfv)
    xtrain_svd = svd.transform(xtrain_tfv)
    xvalid_svd = svd.transform(xvalid_tfv)

    # 对从SVD获得的数据进行缩放
    scl = preprocessing.StandardScaler()
    scl.fit(xtrain_svd)
    xtrain_svd_scl = scl.transform(xtrain_svd)
    xvalid_svd_scl = scl.transform(xvalid_svd)

    # #使用网格搜索法寻找支持向量机的最佳超参数
    # from sklearn.model_selection import GridSearchCV
    # # 把要调整的参数以及其候选值 列出来；
    # param_grid = {"gamma": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    #               "C": [0.001, 0.01, 0.1, 1, 10, 100]}
    # print("Parameters:{}".format(param_grid))
    #
    # clf = SVC(random_state=11)  # 实例化一个SVC类
    #
    # grid_search = GridSearchCV(clf, param_grid, cv=10, scoring='f1_micro')  # 实例化一个GridSearchCV类
    # grid_search.fit(xtrain_svd_scl, ytrain)  # 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
    # print("Test set score:{:.2f}".format(grid_search.score(xvalid_svd_scl, yvalid)))
    # print("Best parameters:{}".format(grid_search.best_params_))
    # print("Best score on train set:{:.2f}".format(grid_search.best_score_))

    # 调用SVM模型
    clf = SVC(C=10, gamma=0.001, random_state=151)
    clf.fit(xtrain_svd_scl, ytrain)  # 输入训练集数据
    predictions = clf.predict(xvalid_svd_scl)  # 输入需要被分类的验证集数据
    print("模型的f1_score: %0.3f " % f1_score(yvalid, predictions, average='micro'))
    model_path = os.path.join(root_dir, "结果数据/留言分类模型.m")
    joblib.dump(clf, model_path)

    # # 读取模型
    # clf = joblib.load("my_model.m")

    # 保存验证集的分类结果
    result = data_valid.copy()
    result.drop(['留言', 'text_split_list', 'text_split'], axis=1, inplace=True)
    result['模型结果'] = lbl_enc.inverse_transform(predictions)

    result_path = os.path.join(root_dir, r'结果数据/分类结果.xlsx')
    result.to_excel(result_path, index=False)
    print(result.head(5))
    print("训练结束，用时：%f" % (time.time() - t))
    print("第一题分类模型构建完毕, 模型保存在%s, 分类结果保存在%s" % (model_path, result_path))


if __name__ == '__main__':
    run_category()
