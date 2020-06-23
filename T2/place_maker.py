# -*- coding: utf-8 -*-
# @Time : 2020/4/14 14:45

import os
import sys
import pandas as pd
root_dir = os.path.dirname(os.path.split(os.path.realpath(sys.argv[0]))[0])


def find_sheng(path):
    """
    从材料中提取出可能的省名
    :return:
    """
    # 搜索文档中可能存在的省名
    data = pd.read_excel(path)
    data["主题详情"] = data['留言主题'] + data['留言详情']
    sheng = []
    for i in data['主题详情']:
        a = list(enumerate(i))
        for j in a:
            if j[1] == "省":
                sheng.append(i[j[0] - 4:j[0] + 3])
    sheng = set(sheng)
    print(sheng)


def transportation():
    """
    生成交通线路名称：地铁一号线、2路公交车
    :return:
    """
    c_num = ["一", "二", "三", "四", "五", "六" "七", "八", "九", "十", "十一", "十二", "十三", "十四", "十五"]
    num = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
    subway = []
    for i in c_num:
        subway.append("地铁" + i + "号线\n")
    for i in num:
        subway.append("地铁" + i + "号线\n")
    bus = []
    for i in range(0, 1000):
        bus.append(str(i) + "路公交车\n")

    return subway + bus


def place_make():
    """
    生成泰迪杯比赛用地名，如省市县区镇名、道路名，也可自己添加
    :return:
    """
    sheng = ["西地省\n"]
    diy = ["小区\n"]  # 自定义地区

    # 生成市县区镇名
    dict_zimu = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    shi = []
    for i in dict_zimu:
        shi.append(i + "市\n")
    for j in dict_zimu:
        for i in range(21):
            shi.append(j + str(i) + "市\n")
    print(shi)

    xian = []
    for i in dict_zimu:
        xian.append(i + "县\n")
    for j in dict_zimu:
        for i in range(21):
            xian.append(j + str(i) + "县\n")
    print(xian)

    qu = []
    for i in dict_zimu:
        qu.append(i + "区\n")
    for j in dict_zimu:
        for i in range(21):
            qu.append(j + str(i) + "区\n")
    print(qu)

    zhen = []
    for i in dict_zimu:
        zhen.append(i + "镇\n")
    for j in dict_zimu:
        for i in range(21):
            zhen.append(j + str(i) + "镇\n")
    print(zhen)

    # 生成道路名
    first = ["解放", "中山", "建设", "人民", "和平", "新华", "劳动", "工业", "文化", "光明", "复兴", "朝阳", "胜利", "自强",
             "健康", "幸福", "仁爱", "太平", "永兴", "兴业", "平安", "创业", "阳光"]
    middle = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "东", "南", "西", "北", "中"]
    special = ["一段", "二段", "三段", "四段", "五段", "中段"]
    r1 = []  # 劳动路
    for i in first:
        r1.append(i + "路\n")
    print(r1)

    r2 = []
    for i in first:
        for j in middle:
            r2.append(i + j + '路\n')
    print(r2)

    r3 = []
    for i in first:
        for j in special:
            r3.append(i + "路" + j + "\n")
    print(r3)

    transportations = transportation()  # 生成交通线路名
    print(transportations)

    f = open(os.path.join(root_dir, r'用户词典\泰迪杯行政地名词典.txt'), 'w', encoding='utf-8')
    f.writelines(sheng + shi + xian + qu + zhen)
    f.close()

    f = open(os.path.join(root_dir, r'用户词典\泰迪杯地名词典.txt'), 'w', encoding='utf-8')
    f.writelines(diy + sheng + shi + xian + qu + zhen + r1 + r2 + r3 + transportations)
    f.close()


def add_ns():
    """
    给词典添加ns词性
    :return:
    """
    f1 = open(os.path.join(root_dir, r'用户词典\泰迪杯地名词典.txt'), 'r', encoding='utf-8')
    f2 = open(os.path.join(root_dir, r'用户词典\泰迪杯地名词典ns.txt'), 'w', encoding='utf-8')
    t1 = f1.readlines()
    for i in t1:
        f2.write(i.replace("\n", "") + " ns\n")
    f1.close()
    f2.close()


if __name__ == '__main__':
    # find_sheng()
    place_make()
    add_ns()
