#!/usr/bin/env python
# -*- encoding:utf-8 -*-
#
#        Author: ZhaoFei - zhaofei@calfdata.com
#        Create: 2017-09-19 16:48:36
# Last Modified: 2017-09-19 16:48:36
#      Filename: user_profile.py
#   Description: ---
# Copyright (c) 2016 Chengdu Lanjing Data&Information Co.


import utils
import pandas as pd


print "---user profile---"
user_profile = pd.read_csv(utils.DATA_DIR + "user_profile_table.csv")
print user_profile.shape
print user_profile.columns

print "---user balance---"
user_balance = pd.read_csv(utils.DATA_DIR + "user_balance_table.csv")
print user_balance.shape
print user_balance.columns







