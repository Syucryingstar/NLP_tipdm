# -*- coding: utf-8 -*-
# @Time : 2020/5/8 11:12


# 运行时长大概2-3分钟，请耐心等待。

# 生成地名词典
from T2.place_maker import *

place_make()
add_ns()

# 预处理
from T2.process import *

run_process()

# 文本聚类
from T2.birch_cluster import *

run_cluster()

# 计算热度指数，文本摘要。命名实体识别并输出结果
from T2.heat_score import *


def run_dig():
    save_data()


if __name__ == '__main__':
    run_dig()
