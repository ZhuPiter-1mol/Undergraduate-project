#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :utils.py
@说明        :工具脚本,存放自定义函数和类
@时间        :2022/03/01 11:03:41
@作者        :钱彬
@版本        :1.0
'''

class AverageMeter(object):
    '''
    平均器类,用于计算平均值、总和
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count