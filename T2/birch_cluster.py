# -*- coding: utf-8 -*-
# @Time : 2020/5/4 23:28

import json
import pickle
from sklearn.metrics import calinski_harabaz_score, silhouette_score, davies_bouldin_score
import pandas as pd
from pyhanlp import HanLP
from dateutil.parser import parse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import metrics
import sys
import time
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from sklearn.cluster import Birch
import datetime

root_dir = os.path.dirname(os.path.split(os.path.realpath(sys.argv[0]))[0])
stop_words = [line.strip() for line in open(os.path.join(root_dir, r'stopword.txt'), 'r', encoding='gbk').readlines()]


# 封装日志
class Logger(object):

    def __init__(self, name='', log_name='log.txt', log_dir="./"):
        # 生成一个日志对象,()内为日志对象的名字,可以不带,名字不给定就是root
        self.logger = logging.getLogger(name)  # 一般给定名字,否则会把其他的日志输出也会打印到你的文件里。

        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        log_file = os.path.join(log_dir, log_name)
        # print(log_file)

        # 生成一个滚动的handler（处理器），按天滚动
        file_handler = TimedRotatingFileHandler(filename=log_file, when="D", interval=7, backupCount=1,
                                                encoding='utf-8')
        console_handler = logging.StreamHandler()

        # formatter 下面代码指定日志的输出格式
        fmt = '%(asctime)s - %(message)s'
        formatter = logging.Formatter(fmt)  # 实例化formatter

        file_handler.setFormatter(formatter)  # 为handler添加formatter
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        log_level = logging.DEBUG
        self.logger.addHandler(console_handler)
        self.logger.setLevel(log_level)  # 设置日志输出信息的级别

    def get_logger(self):
        return self.logger


log_path = os.path.join(root_dir, r"T2")
log_name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) + ".txt"
log = Logger(name='debug', log_dir=os.path.join(log_path, "log"), log_name=log_name).get_logger()  # 实例化log对象


def process(s):
    """
    预处理，分词后去停用词
    :param s: str
    :return: eg: '今天 天气 很好'
    """
    s = s.replace('\t', '').replace('\n', '').replace("\u200b", "").strip()
    segment = HanLP.newSegment() \
        .enablePlaceRecognize(True) \
        .enableCustomDictionary(True) \
        .enableOrganizationRecognize(True) \
        .enableNameRecognize(True)
    hanlp_result = segment.seg(s)
    word_list = [i.word for i in hanlp_result]
    nature_list = [i.nature for i in hanlp_result]

    sss = [word_list[i] + "/" + str(nature_list[i]) for i in range(len(word_list)) if word_list[i] not in stop_words]
    res = " ".join(sss)
    if not res.strip():
        res = "none/none"
    return res


def get_tfidf(corpus):
    """
    获取tfidf矩阵, 参数酌情自调
    :param corpus: 语料
    :return:
    """
    t = time.time()
    vectorizer = TfidfVectorizer(max_df=0.8, max_features=5000, min_df=3, ngram_range=(1, 2))
    weights = vectorizer.fit_transform(corpus.values.astype('U')).toarray()

    log.info("get_tfidf()获取tfidf向量 维度数：%s  用时 %f" % (str(weights.shape), time.time() - t))
    return weights


def get_tfidf_weighted(corpus, max_df=0.8, max_features=4000, min_df=4, ngram_range=(1, 2)):
    """
    对语料进行tfidf处理后，再根据语料的词性权重设置
    :param corpus: 语料
    :param max_df:
    :param max_features:
    :param min_df:
    :param ngram_range:
    :return:
    """
    t = time.time()

    word_after_cut = []
    part_of_speech = []

    for i in range(len(corpus)):
        b = str(corpus[i]).split(" ")
        word_after_cut = [j.split("/")[0] for j in b]
        part_of_speech = [j.split("/")[1] for j in b]

    pos_word_dict = {word_after_cut[i]: part_of_speech[i] for i in range(len(word_after_cut))}  # 构建词库
    # print(pos_word_dict)

    vectorizer = TfidfVectorizer(max_df=max_df, max_features=max_features, min_df=min_df, ngram_range=ngram_range)
    tf_idf = vectorizer.fit_transform(corpus.values.astype('U'))

    weight = tf_idf.toarray()  # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    log.info(
        "tfidf向量获取完毕，参数：max_df=%f, min_df=%d, max_features=%d, ngram_range=%s  维度数：%s  用时 %f" % (
            max_df, min_df, max_features, ngram_range, str(weight.shape), time.time() - t))

    # 基于词性的新权重
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    word_weight = [1 for i in range(len(word))]

    organization = ['ni', 'nic', 'nis', 'nit', 'nts', 'ntu', 'nto', 'nth', 'ntch', 'ntcf', 'ntcb', 'ntc', 'nrf', 'nr',
                    'nnt', 'nnd', 'nn', 'nz']  # 机构。银行医院学校公司机构等 还有人名
    place = ['ns', 'nsf']
    '''
    hanlp词性解释：https://blog.csdn.net/u014258362/article/details/81044286
    jieba词性解释：https://blog.csdn.net/huludan/article/details/52727298
    '''
    for i in range(len(word)):  # 权重调整根据实际情况进行更改
        if word[i] not in pos_word_dict.keys():  # 忽略词袋模型中没有的词
            continue
        if pos_word_dict[word[i]] in organization:  # 机构团体\人名加权
            word_weight[i] = 1.2
        elif pos_word_dict[word[i]] in place:  # 地名加权
            word_weight[i] = 2
        elif pos_word_dict[word[i]] == 'n':  # 名词加权
            word_weight[i] = 1.2
        elif pos_word_dict[word[i]] == 'm':  # 数词加权
            word_weight[i] = 1.2
        else:
            continue
    word_weight = np.array(word_weight)
    new_weight = weight.copy()
    for i in range(len(weight)):
        for j in range(len(word)):
            new_weight[i][j] = weight[i][j] * word_weight[j]  # 为每个语料样本的每个特征加权
    # print(new_weight)
    log.info("get_tfidf_weighted() 基于词性加权, 维度数：%s 用时 %f" % (str(weight.shape), time.time() - t))

    return new_weight


def pca(weights, n_components='auto'):
    """
    对数据进行PCA降维, 数据量小的话不需要
    :param mode: 选择降维模式, 默认手动设置n_components。若mode="auto"，则降维到解释度为99%
    :param weights: tfidf处理后的向量
    :param n_components: <1 则为保留下来的特征维度数； >1 则为保留原来维数的百分之多少
    :return:
    """
    t = time.time()
    if n_components == 'auto':
        n_components = 0.99
    pca = PCA(n_components=n_components)
    pca_weights = pca.fit_transform(weights)
    # pickle.dump(pca_weights, open("pca_weights.pickle", "wb"))  # 保存模型
    log.info("降维后的词向量保存在：pca_weights.pickle")
    log.info(
        "pca降维, 维度：%s 可解释度：%f 用时 %f" % (pca_weights.shape, pca.explained_variance_ratio_.sum(), time.time() - t))
    return pca_weights


def process_result(result):
    """
    处理聚类结果文件，处理出留言时间范围、最新帖子时间、每条留言(点赞数-反对数)的和
    :param result:
    :param path:
    :return:
    """
    tim = time.time()
    path = os.path.join(root_dir, r"T2/cluster_result.json")
    cluster_result = result
    cluster_all = []
    for i in cluster_result:
        cluster = {}  # 每一个簇的信息
        cluster['留言数'] = len(i['留言主题'])  # 此簇中样本个数
        all_text = ''
        for j in range(len(i['留言主题'])):
            all_text += (i['留言主题'][j] + "。")
        cluster['留言主题'] = all_text  # 此簇中的每个留言主题用中文句号相连形成一个段落-->今天天气好。我心情不错。我很高兴。

        t = [parse(e) for e in i['留言时间']]  # 格式化时间字段
        t_copy = t.copy()
        t.sort()  # 对此簇中的留言时间排序
        message_id_list = []  # 留言编号列表
        flag = len(t_copy)
        for r in t:  # 根据留言时间升序排列留言编号
            for j in range(flag):
                a = r.strftime("%Y/%m/%d %H:%M:%S")
                b = t_copy[j].strftime("%Y/%m/%d %H:%M:%S")
                if a == b:
                    message_id_list.append(i['留言编号'][j])
                    del t_copy[j]
                    del i['留言编号'][j]
                    flag -= 1
                    break

        cluster['留言编号'] = message_id_list
        cluster['时间范围'] = str(t[0]) + " 至 " + str(t[-1])  # 留言时间范围

        # 最新和最久远的时间转化成秒数
        date_time_new = datetime.datetime.strptime(str(t[-1]), '%Y-%m-%d %H:%M:%S')
        s_time_new = time.mktime(date_time_new.timetuple())
        cluster['最新时间戳'] = s_time_new

        cluster['点赞数-反对数'] = sum(i['点赞反对差'])

        cluster_all.append(cluster)
    # print(cluster_all)
    with open(path, 'w', encoding='utf-8') as json_file:
        json.dump(cluster_all, json_file, ensure_ascii=False)
    log.info("聚类结果保存在%s, 用时：%f" % (path, (time.time() - tim)))
    return cluster_all


def drow_data(i_, branching_factor, calinskiharabaz_score, _silhouette_score, dbscore):
    """
    绘制评价指数的变化
    :param i_: 聚类标签
    :param branching_factor:
    :param calinskiharabaz_score:
    :param _silhouette_score:
    :param dbscore:
    :return:
    """
    plt.figure()
    plt.plot(i_, calinskiharabaz_score)
    plt.xlabel("birch-threshold")
    plt.ylabel("calinski_harabaz_score")
    plt.title("branching_factor=%d" % branching_factor)
    # plt.savefig("calinski_harabaz_score.png")
    plt.show()

    plt.figure()
    plt.plot(i_, _silhouette_score)
    plt.xlabel("birch-threshold")
    plt.ylabel("_silhouette_score")
    plt.title("branching_factor=%d" % branching_factor)
    # plt.savefig("_silhouette_score.png")
    plt.show()

    plt.figure()
    plt.plot(i_, dbscore)
    plt.xlabel("birch-threshold")
    plt.ylabel("daviesbouldinscore")
    plt.title("branching_factor=%d" % branching_factor)
    # plt.savefig("daviesbouldinscore.png")
    plt.show()


def birch(w, data, branching_factor=10, threshold=0.01, fig=""):
    """
    进行BIRCH聚类，经过多轮调参后发现branching_factor=160，threshold=0.82时的聚类效果最佳
    :param w: 词向量
    :param data: 分词附件3.xlsx的pd对象
    :param branching_factor: 调参
    :param threshold: 调参
    :param fig: 是否画图。参数为"2D"/"3D"，默认为不画图
    :return:
    """
    for s in range(160, 161, 50):
        i_ = []
        calinskiharabaz_score = []
        _silhouette_score = []
        dbscore = []
        for j in range(82, 83, 1):
            t = time.time()
            clf = Birch(n_clusters=None, branching_factor=s, threshold=j / 100).fit(w)

            log.info("聚类结束, threshold: %f  branching_factor:%d 用时 %f" % ((j / 100), s, time.time() - t))

            cluster_labels = list(clf.labels_)  # 聚类后的标签，-1为未分类的噪点
            # print(cluster_labels)

            data['cluster'] = cluster_labels

            cluster_list = list(set(cluster_labels))  # 标签列表 eg: [-1, 0, 1, 2]
            noise = 0  # 初始化噪点数（未分类的样本个数）
            if -1 in cluster_list:
                noise = cluster_labels.count(-1)  # 统计有多少个-1标签的样本，即noise
                cluster_list.remove(-1)  # 去除-1标签（未分类）
            log.info("已分类%s个类簇:" % len(cluster_list))

            result = []  # 分类结果
            for i in cluster_list:
                row = {}  # 每个簇的信息
                row['留言编号'] = list(data[data['cluster'] == i]['留言编号'])
                row['留言主题'] = list(data[data['cluster'] == i]['主题分词'].apply(lambda s: s.replace(" ", '')))
                row['留言时间'] = list(data[data['cluster'] == i]['留言时间'])
                row['点赞反对差'] = list(data[data['cluster'] == i]['点赞反对差'])
                result.append(row)
            # log.info("result: {}".format(result))

            # 把每组标签对应的留言数据归为一类打印输出
            res = []
            for i in result:
                res.append(i['留言主题'])

            res_sorted = sorted(res, key=lambda i: len(i), reverse=True)  # 倒序
            for i in range(len(res_sorted)):
                if i < 30:  # 只打印前20个簇
                    log.info("%d%s" % (len(res_sorted[i]), res_sorted[i]))
            # [log.info("%d%s" % (len(i), i)) for i in res_sorted]  # 打印所有簇

            log.info("未分类(噪点)%d个: " % noise)
            # log.info(list(data[data['cluster'] == -1]['留言主题']))

            # 保存聚类结果到json文件中
            process_result(result)

            # 画图
            if fig == "2D":
                weights = pca(w, 2)  # 向量降为2维
                plt.scatter(weights[:, 0], weights[:, 1], c=cluster_labels)
                plt.show()
            elif fig == "3D":
                weights = pca(w, 3)  # 向量降为3维
                figs = plt.figure()
                axes3d = Axes3D(figs)
                axes3d.scatter(weights[:, 0], weights[:, 1], weights[:, 2], c=cluster_labels)
                plt.show()

            try:
                calinskiharabazscore = calinski_harabaz_score(w, clf.labels_)
                silhouettescore = silhouette_score(w, clf.labels_, metric='euclidean')
                daviesbouldinscore = davies_bouldin_score(w, clf.labels_)
                i_.append(j / 100)
                calinskiharabaz_score.append(calinskiharabazscore)
                _silhouette_score.append(silhouettescore)
                dbscore.append(daviesbouldinscore)
            except:
                log.info("birch-threshold %f 计算出错" % (j / 100))
                continue

        log.info("calinski_harabaz_score {}".format(calinskiharabaz_score))
        log.info("silhouette_score  {}".format(_silhouette_score))
        log.info("davies_bouldin_score  {}".format(dbscore))
        # drow_data(i_, s, calinskiharabaz_score, _silhouette_score, dbscore)  # 绘制评价指数的变化


def prepare_data(path):
    """
    准备数据
    :return: pd.DataFrame格式数据
    """
    t = time.time()
    if path.split(".")[1] in ['xlsx', 'xls']:  # 自动判断文件类型
        data = pd.read_excel(path)
    elif path.split(".")[1] == 'csv':
        data = pd.read_csv(path)
    else:
        data = pd.read_excel(path)

    corpus_name = "主题分词_词性"
    data["语料"] = data[corpus_name]
    log.info("加载语料结束，语料路径：%s  使用语料：%s 用时 %f" % (path, corpus_name, time.time() - t))

    return data


def run_cluster():
    t0 = time.time()
    # 设置停用词
    data = prepare_data(os.path.join(root_dir, r"T2/分词附件3.xlsx"))  # 加载数据

    # 保存词向量
    # if os.path.exists("pca_weights.pickle"):
    #     w = pickle.load(open("pca_weights.pickle", "rb"))
    #     log.info("成功读取pca_weights本地词向量")
    # else:
    #     # 获取特征向量
    #     w = pca(get_tfidf_weighted(data['语料'], max_df=0.8, max_features=5000, min_df=3, ngram_range=(1, 2)), 0.999)

    # 获取词向量后进行pca(SVD)降维
    w = pca(get_tfidf_weighted(data['语料'], max_df=0.8, max_features=5000, min_df=3, ngram_range=(1, 2)), 0.999)

    birch(w, data, branching_factor=160, threshold=0.82)  # 文本聚类

    log.info("总共用时：%f" % (time.time() - t0))


if __name__ == '__main__':
    run_cluster()
