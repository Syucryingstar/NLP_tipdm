# coding: utf-8
import os
import sys
import jieba
import time
import pandas as pd
from gensim import corpora, models, similarities
from jieba import posseg as pseg

root_dir = os.path.dirname(os.path.split(os.path.realpath(sys.argv[0]))[0])
jieba.load_userdict(os.path.join(root_dir, r'用户词典/泰迪杯地名词典ns.txt'))  # 加载自定义词典

# 导入停用词
# 设置词性列表，分词后根据词性去掉
stopwords = [w.strip() for w in open(os.path.join(root_dir, "stopword.txt"), 'r', encoding='gbk')]
stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']


# 分词函数
def cut_word(text):
    words = pseg.cut(text)
    result = []
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result


def pro(s):
    return list(s)


def run_process():  # 预处理函数
    print("开始预处理第三题数据")
    t = time.time()
    data = pd.read_excel(os.path.join(root_dir, '全部数据/附件4.xlsx'), dtype=str)
    data['留言详情'] = data['留言详情'].apply(lambda i: cut_word(i))
    data['答复意见'] = data['答复意见'].apply(lambda i: cut_word(i))
    data['留言主题'] = data['留言主题'].apply(lambda i: cut_word(i))
    data.to_excel(os.path.join(root_dir, 'T3/分词附件4.xlsx'), index=None)
    print("第三题预处理完毕，用时：%f" % (time.time() - t))


def run_relativity():  # 计算答复的相关性
    print("开始计算答复的相关性")

    t = time.time()
    if not os.path.exists(os.path.join(root_dir, 'T3/分词附件4.xlsx')):
        run_process()
    data = pd.read_excel(os.path.join(root_dir, 'T3/分词附件4.xlsx'))
    data['留言详情'] = data['留言详情'].apply(lambda i: list(i))
    data['答复意见'] = data['答复意见'].apply(lambda i: list(i))
    data['留言主题'] = data['留言主题'].apply(lambda i: list(i))

    appraise = []
    corpus = data['留言详情'].tolist()

    # 创建词袋语料库，记录文本的词频
    dictionary = corpora.Dictionary(corpus)  # 建立词典
    doc_vectors = [dictionary.doc2bow(text) for text in corpus]  # 实现词袋模型

    # 从语料库中生成tfidf模型
    tfidf = models.TfidfModel(doc_vectors)  # 将文档由词频表示转换为tf-idf格式
    tfidf_vectors = tfidf[doc_vectors]

    # 创建相似度模型
    index = similarities.Similarity('./', tfidf_vectors, len(dictionary))
    # 测试
    for i in range(0, len(data)):
        # 构建query文本
        query = data.iloc[i]['答复意见']
        query_bow = dictionary.doc2bow(query)  # 转换为bows格式
        # 下一步处理,引用tf-idf模型,计算相关性
        query_tfidf = tfidf[query_bow]
        sims_tfidf = index[query_tfidf]  # 计算相似度
        # 根据相关性评价回复
        if sims_tfidf[i] < 0.1:
            appraise.append('差')
        else:
            appraise.append('良')

    # 保存结果
    data_1 = pd.read_excel(os.path.join(root_dir, '全部数据/附件4.xlsx'), dtype=str)
    data_1.insert(7, '相关性评价', appraise)
    data_1.to_excel(os.path.join(root_dir, 'T3/result_相关性.xlsx'), index=False, sheet_name='result')
    print("相关性评价完毕，用时：%f" % (time.time() - t))


if __name__ == '__main__':
    run_relativity()
