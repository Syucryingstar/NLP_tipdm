# -*- coding: utf-8 -*-
# @Time : 2020/4/9 20:03

import time
import pandas as pd
import jieba
import jieba.posseg as pseg
from pyhanlp import *
root_dir = os.path.dirname(os.path.split(os.path.realpath(sys.argv[0]))[0])
# 设置停用词
stop_words = [line.strip() for line in open(os.path.join(root_dir, 'stopword.txt'), 'r', encoding='gbk').readlines()]


def summary(document):
    """
    提取文本段落的摘要(排名前两句)
    :param document: A市保利麓谷林语小区，是A市的大型小区，居住人口有一万多人。地下一层墙体严重开裂，并且地下水和泥沙大量溢出，
    特别是逢下雨天 严重，小区还算比较新，才建几年，楼高33楼，我们让专业的人员看过，不仅负一楼停车场经常被“淹”，并且大量泥沙漫出，
    而且严重影响威胁居民的生命生产安全。我们像保利物业和保利地产投诉和反应过很多次，未果。而且投诉时长可以用年来计算，保利的拖延
    理由是“等天晴出太阳再说”这个太阳一等就不知道是多久，最近保利更是出息了， 视居民的安全不顾，选择将裂缝“盲视”，在楼体裂缝处的表
    面全部“镀”上了钢板来装饰。（划重点）小区扶手全部锈掉去年发生了重大事故小孩靠着扶手休息就坠楼死亡，路灯被风一吹就，就把路过的老
    人砸倒，诸多问题多次引起了西地省台的新闻报道，但是小区毫无半点整改，和保利总部多次反应也未果，地方各部门也并且到处踢皮球。小区
    想成立业委会，A市保利物业多翻阻拦，简直成了纵横的地霸。
    :return: [A市保利麓谷林语小区, 我们像保利物业和保利地产投诉和反应过很多次]
    """
    sentence_list = HanLP.extractSummary(document, 2)
    print("提取两句摘要：%s" % str(sentence_list))
    return " ".join(sentence_list)


def pro_special(_data):
    """
    填充无意义的留言主题，将异常的留言主题设为其留言详情中提取的摘要
    eg: 留言主题="（"   # 提取摘要前的留言主题
        留言主题="A市保利麓谷林语小区 我们像保利物业和保利地产投诉和反应过很多次"  # 提取摘要后的留言主题
    :param _data: 附件3的pandas对象
    :return:
    """
    _data["主题分词"] = _data["留言主题"].apply(lambda s: process_jieba(s))
    for i in range(len(_data)):
        if _data["主题分词"][i] == "none":
            _data["留言主题"][i] = summary(_data["留言详情"][i])
    del _data["主题分词"]
    return _data


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


def process_jieba_nature(s):
    """
    预处理，分词后去停用词
    :param s: str
    :return: eg: '今天 天气 很好'
    """
    # s = rep(s)
    w = []
    for i in pseg.cut(s):
        if i.word == " ":
            continue
        w.append(i.word + "/" + str(i.flag))

    res = " ".join(w)
    if not res.strip():
        res = "none/none"
    return res


def process_jieba(s):
    """
    预处理，分词后去停用词
    :param s: str
    :return: eg: '今天 天气 很好'
    """
    s = rep(s)
    word_list = []
    for i in pseg.cut(s):
        word_list.append(i.word)
    sss = [word_list[i] for i in range(len(word_list)) if word_list[i] not in stop_words]
    res = " ".join(sss)
    if not res.strip():
        res = "none"
    return res


def process_hanlp_nature(s):
    """
    预处理，分词后去停用词
    :param s: str
    :return: eg: '今天 天气 很好'
    """
    s = rep(s)
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


def process_hanlp(s):
    """
    预处理，分词后去停用词
    :param s: str
    :return: eg: '今天 天气 很好'
    """
    s = rep(s)
    segment = HanLP.newSegment() \
        .enablePlaceRecognize(True) \
        .enableCustomDictionary(True) \
        .enableOrganizationRecognize(True) \
        .enableNameRecognize(True)
    hanlp_result = segment.seg(s)
    word_list = [i.word for i in hanlp_result]
    sss = [word_list[i] for i in range(len(word_list)) if word_list[i] not in stop_words]
    res = " ".join(sss)
    if not res.strip():
        res = "none"

    return res


def run_process():
    jieba.load_userdict(os.path.join(root_dir, r'用户词典/泰迪杯地名词典ns.txt'))  # 加载自定义词典
    print("开始预处理语料")
    data = pd.read_excel(os.path.join(root_dir, r'全部数据/附件3.xlsx'))
    data.drop_duplicates(subset=["留言编号", "留言用户", "留言主题", "留言时间", "留言详情", "点赞数", "反对数"]
                         , keep=False, inplace=True)  # 去重
    data = pro_special(data)  # 填充无意义的留言主题
    data["留言时间"] = data["留言时间"].apply(lambda s: str(s).replace("-", "/"))  # 格式化时间
    data["主题分词"] = data["留言主题"].apply(lambda s: process_jieba(s))  # 将留言主题分词
    data["主题分词_词性"] = data["主题分词"].apply(lambda s: process_jieba_nature(s))  # 将留言主题分词(带词性)
    data["点赞反对差"] = data["点赞数"] - data["反对数"]

    data.to_excel(os.path.join(root_dir, r'T2/分词附件3.xlsx'), index=None)  # 保存文件


if __name__ == '__main__':
    t = time.time()
    run_process()  # 开始预处理
    print("预处理结束，用时%f" % (time.time() - t))
