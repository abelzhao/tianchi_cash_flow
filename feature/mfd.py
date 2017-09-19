#!/usr/bin/env python
# -*- encoding:utf-8 -*-
#
#        Author: ZhaoFei - zhaofei@calfdata.com
#        Create: 2017-09-19 17:05:54
# Last Modified: 2017-09-19 17:05:54
#      Filename: mfd.py
#   Description: ---
# Copyright (c) 2016 Chengdu Lanjing Data&Information Co.


import utils
import pandas as pd

print "---bank shibor---"
bank_shibor = pd.read_csv(utils.DATA_DIR+"mfd_bank_shibor.csv")
print bank_shibor.shape
print bank_shibor.columns



print "---day share interest---"
day_share_interest = pd.read_csv(utils.DATA_DIR+"mfd_day_share_interest.csv")
print day_share_interest.shape
print day_share_interest.columns

