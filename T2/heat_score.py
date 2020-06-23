# -*- coding: utf-8 -*-
# @Time : 2020/4/14 18:41

import math
import time
from pyhanlp import *
import json
import pandas as pd
import jieba

root_dir = os.path.dirname(os.path.split(os.path.realpath(sys.argv[0]))[0])
segment = HanLP.newSegment() \
    .enablePlaceRecognize(True) \
    .enableCustomDictionary(True) \
    .enableOrganizationRecognize(True) \
    .enableNameRecognize(True)  # hanlp分词词性标注器
# 设置停用词
stop_words = [line.strip() for line in open(os.path.join(root_dir, r'stopword.txt'), 'r', encoding='gbk').readlines()]
organization = ['ni', 'nic', 'nis', 'nit', 'nts', 'ntu', 'nto', 'nth', 'ntch', 'ntcf', 'ntcb', 'ntc', 'nrf', 'nr',
                'nnt', 'nnd', 'nn', 'nt', 'ng', 'nz']  # 机构。银行医院学校公司机构等 还有人名、其他专名
place = ['ns', 'nsf']  # 地名和音译地名
entity = ['entity']  # 实体词性
# 行政地名词典
place_list = [line.strip() for line in
              open(os.path.join(root_dir, r'用户词典/泰迪杯行政地名词典.txt'), 'r', encoding='utf-8').readlines()]
jieba.load_userdict(os.path.join(root_dir, r'用户词典/泰迪杯地名词典ns.txt'))  # 加载自定义词典


def process(s):
    """
    预处理，分词后去停用词
    :param s: str
    :return: eg: '今天天气很好'
    """
    s = s.replace('\t', '').replace('\n', '').strip()
    hanlp_result = segment.seg(s)
    ss = [i.word for i in hanlp_result]
    sss = [i for i in ss if i not in stop_words]
    return "".join(sss)


def place_extraction(s):
    """
    从此簇所有留言主题中提取完整的行政地区名
    :param s: "A市强制学生实习。A3区强制学生实习。某省强制实习"
    :return: “某省A市A3区”
    """
    text_place = []  # 文本中的地名列表
    cut_word = segment.seg(s)  # 分词
    for i in cut_word:  # 把地名属性的词提取出来
        if str(i.nature) == "ns":
            text_place.append(i.word)
    text_place = set(text_place)  # 地名去重
    sheng = ''  # 省
    shi = ''  # 市
    xian = ''  # 县
    qu = ''  # 区
    zhen = ''  # 镇
    bus = ''  # 公交车
    subway = ''  # 地铁
    for i in text_place:  # 提取各级地名
        if i in place_list:
            if "省" in i:
                sheng = i
            elif "市" in i:
                shi = i
            elif "县" in i:
                xian = i
            elif "区" in i:
                qu = i
            elif "镇" in i:
                zhen = i
            elif "公交车" in i:
                bus = i
            elif "地铁" in i:
                subway = i
    place = sheng + shi + xian + qu + zhen + bus + subway  # 完整的地区名
    # print(place)
    return place


def completion_palce(s, place):
    """
    补全句子中的行政地名
    :param s: 句子
    :return:
    """
    # print("提取行政地名：%s" % place)
    sentence = s
    word_list = [i.word for i in segment.seg(sentence)]  # 对摘要进行分词，方便替换地名
    place_flag = False  # 判断有无行政地名
    for i in word_list:
        if i in place_list:
            place_flag = True

    if place_flag is False:  # 如果摘要没有行政地名则给它加上
        sentence = place + "".join(word_list)
    else:
        # "A市学生强制实习" ==> "西地省A市5区学生强制实习"
        flag = False
        len_word_list = len(word_list)
        count = -1
        for i in range(len_word_list):
            count += 1
            if count <= len_word_list:
                if word_list[count] in place_list:  # 如果第一个地名出现，则把前面的place赋值给它，之后再出现的地名则全部删除
                    if flag is True:
                        del word_list[count]
                        len_word_list -= 1
                        count -= 1
                        continue
                    word_list[count] = place
                    flag = True

        sentence = "".join(word_list)
    if sentence == "":
        sentence = s

    return sentence


def summary_extraction(s):
    """
    生成文本摘要(含完整行政地区名)
    :param s:
    :return:
    """
    place = place_extraction(s)  # 提取完整的行政地名

    if s.split("。")[1] == "" and len(s.split("。")) == 2:  # 如果只有1个句子就跳过摘要
        summary = s.replace("。", "")
    else:
        summary = HanLP.extractSummary(s, 1)[0].replace("。", "")

    summary = completion_palce(summary, place)
    # print("摘要："+summary)
    return summary


def entity_flag(s):
    """
    这波操作是对词法分析后词性不准确的修正：
    例: 经过词法分析后的分词词性： ['A市楚府东路/nx', '路面/n', '状况/n', '特别/d', '差/a', '修整/v']
    其中"A市楚府东路"被标注为'nx'字母专名，这显然是不对，所以对"A市楚府东路"进行hanlp分词词性标注->['A市/ns', '楚府/tag', '东路/ns']
    再遍历词性，若词性中含有如organization、place中的词性，则把实体块'A市楚府东路'标注为"entity"词性。
    :param s: 'A市楚府东路'
    :return: bool 此函数返回的是 是否为"entity"词性
    """
    flag = False
    for j in segment.seg(s):
        if str(j.nature) in organization + place:
            flag = True
    return flag


def entity_extraction(s, mode="CRF"):
    """
    分词后根据词性提取出关键词
    hanlp词性解释：https://blog.csdn.net/u014258362/article/details/81044286
    :param mode: 模型选择，有“Perceptron”和“CRF”两种模型
    :param s:
    :return:
    """
    # biological = ['nb', 'nba', 'nbc', 'nbp', 'nf']  # 生物类和食品
    # medicine = ['nh', 'nhd', 'nhm']  # 医疗药物类

    part_of_speech = []  # 词性列表
    word_after_cut = []  # 词列表
    PerceptronLexicalAnalyzer = JClass('com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer')  # 感知机词法分析
    CRFLexicalAnalyzer = JClass("com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer")  # CRF词法分析
    if mode == "CRF":  # 模型选择
        analyzer = CRFLexicalAnalyzer()
    else:
        analyzer = PerceptronLexicalAnalyzer()

    for i in analyzer.analyze(s):
        if entity_flag(str(i.getValue())):
            part_of_speech.append("entity")
        else:
            part_of_speech.append(str(i.getLabel()))
        word_after_cut.append(str(i.getValue()))

    # print(word_after_cut + part_of_speech)

    # 下面根据词性筛选出实体成分
    word_pos_dict = {word_after_cut[i]: part_of_speech[i] for i in range(len(word_after_cut))}  # 把分词和对应的词性存入字典方便读取
    entity_list = []  # 实体成分列表
    for i in range(len(word_after_cut)):
        if word_pos_dict[word_after_cut[i]] in organization + place + entity:  # 机构团体\人名 + 地名和音译地名
            entity_list.append(i)

    addr_person = ""  # 根据索引把实体成分(词)连在一起构成实体 --> 地点/人群
    for i in entity_list:
        addr_person += word_after_cut[i]

    if addr_person == "":  # 如果分析不出来则返回原来值
        addr_person = s
        return addr_person

    '''
    若addr_person中只含有行政地名，则把此句中所有名词也算进实体中，如 "A市人才新政落户申请购房补贴成功" 提取的实体只有"A市"，
    则把"人才/n"、"新政/n"加进实体 -> "A市人才新政"
    '''
    check = segment.seg(addr_person)
    check_flag = 0
    for i in check:
        if i.word in place_list:
            check_flag += 1
    if check_flag == len(check):  # 判断行政地名数量是否等于整个句子分词的数量，若等于则说明此实体只含有行政地名，需要加名词
        for i in range(len(word_after_cut)):
            a = word_pos_dict[word_after_cut[i]]
            if a in ['n']:  # 把名词属性的词加进去
                entity_list.append(i)
        addr_person = ""  # 根据索引把实体成分(词)连在一起构成实体 --> 地点/人群
        for i in entity_list:
            addr_person += word_after_cut[i]

    return addr_person


def get_entity(s):
    """
    选择字符串最长的实体(包含信息更多)
    :param s:
    :return:
    """
    place = place_extraction(s)  # 提取完整的行政地名
    _crf = entity_extraction(s, mode="CRF")
    _nlp = entity_extraction(s, mode="NLP")
    if len(_crf) < len(_nlp):
        _entity = _nlp
    else:
        _entity = _crf
    _entity = completion_palce(_entity, place)
    # print("实体：" + _entity)
    # print()
    return _entity


def make_score(x, t, s):
    """
    参考了reddit的热度公式
    :param s: 留言数
    :param x: 点赞数-反对数
    :param t: 发帖时间戳（秒）
    :return:
    """
    t1 = t - 1356364800  # 以2000/1/1 00:00:00的时间戳为参照
    if x >= 1:
        z = x
    else:  # x < 0
        z = 1
    score = (s * t1 / 86400) * (1 + (0.1 * math.log(z, 2)))
    return round(score, 3)


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


def save_data(path=os.path.join(root_dir, r"T2/cluster_result.json")):
    """
    计算最终热度指数并按题目规定格式保存文件
    :param path: 聚类结果文件，json格式
    :return:
    """
    t = time.time()
    # result = process_result("cluster_result.json")
    print("开始读取文件...")
    with open(path, 'r', encoding='utf-8') as json_file:
        heat_score_data = json.load(json_file)
    score_list = []
    for i in heat_score_data:  # 遍历每个簇的数据,计算这个簇的热度指数
        i['score'] = make_score(i['点赞数-反对数'], i['最新时间戳'], i['留言数'])
        score_list.append(i['score'])
    score_list.sort(reverse=True)  # 根据热度指数降序排序
    print("热度指数计算完成")

    t2 = time.time()
    table1 = []  # 输出热点问题表
    rank_id = []  # 热度排名id列表，按热度排序
    id_list = []  # 留言编号列表，按簇内升序后热度排序
    print("开始提取摘要和命名实体...")
    for i in range(0, 5):  # 取排名前5的热点问题
        for j in heat_score_data:
            if j['score'] == score_list[i]:
                j['describe'] = summary_extraction(j['留言主题'])
                j['addr_person'] = get_entity(j['describe'])  # 提取问题描述的实体，抽取问题描述
                table1.append([i + 1, i + 1, j['score'], j['时间范围'], j['addr_person'], j['describe']])
                rank_id += [i + 1 for d in range(j['留言数'])]
                id_list += j['留言编号']
    print("提取完毕，用时: %f，正在输出文件..." % (time.time() - t2))
    t3 = time.time()
    table1 = pd.DataFrame(table1)
    table1.columns = ['热度排名', '问题ID', '热度指数', '时间范围', '地点/人群', '问题描述']
    table1.to_excel(os.path.join(root_dir, "结果数据/热点问题表.xlsx"), index=None, encoding='utf-8')

    print("热点问题表输出完毕， 用时：%f" % (time.time() - t3))

    print("开始处理表2...")
    t4 = time.time()
    table2 = []  # 输出表2
    data = pd.read_excel(os.path.join(root_dir, r'全部数据/附件3.xlsx'))
    for i in id_list:
        table2.append(data[data['留言编号'] == i].values[0])
    table2 = pd.DataFrame(table2)
    table2.columns = ["留言编号", "留言用户", "留言主题", "留言时间", "留言详情", "点赞数", "反对数"]
    table2.insert(0, "问题ID", rank_id)
    table2['留言时间'] = table2['留言时间'].apply(lambda s: str(s).replace("-", "/"))  # 格式化时间
    table2['留言详情'] = table2['留言详情'].apply(lambda s: rep(s))
    table2.to_excel(os.path.join(root_dir, "结果数据/热点问题留言明细表.xlsx"), index=None, encoding='utf-8')
    print("热点问题留言明细表输出完毕，用时: %f" % (time.time() - t4))
    print("第二题处理结束，总共用时：%f" % (time.time() - t))


if __name__ == '__main__':
    save_data(os.path.join(root_dir, r"T2/cluster_result.json"))
