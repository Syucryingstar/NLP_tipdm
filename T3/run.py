# -*- coding: utf-8 -*-
# @Time : 2020/5/8 14:05
from T3.relativity import *
from T3.interpretability import *


def run_opinion():
    # 计算答复的相关性
    run_relativity()

    # 计算答复的可解释性
    run_interpretability()


if __name__ == '__main__':
    run_opinion()
