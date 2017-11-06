#!/usr/bin/env python
# -*- encoding:utf-8 -*-

#        Author: ZhaoFei - zhaofei@calfdata.com
#        Create: 2017-10-25 11:56:47
# Last Modified: 2017-10-25 11:56:47
#      Filename: 2-lr-multiclassify.py
#   Description: ---
# Copyright (c) 2016 Chengdu Lanjing Data&Information Co.


import gc
import time
import numpy as np
import pandas as pd
from pandas.core.series import Series
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.random_projection import SparseRandomProjection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# dir
DATA_DIR = "./data/"
RET_DIR = DATA_DIR + "predict/"

# STEP 0.
print "0. construct several ml models"
rf = RandomForestClassifier(n_estimators=300,
                            criterion="gini",
                            max_features="auto",
                            max_depth=None,
                            min_samples_split=5,
                            min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0,
                            max_leaf_nodes=None,
                            bootstrap=True,
                            oob_score=False,
                            n_jobs=4,
                            random_state=13,
                            verbose=1,
                            warm_start=True)
et = ExtraTreesClassifier(n_estimators=300,
                          n_jobs=4)
bg = BaggingClassifier(n_estimators=300,
                       max_samples=0.8,
                       max_features=0.8,
                       n_jobs=4,
                       random_state=1,
                       verbose=1)

# STEP 1.
print "1. load data"
shop_infos = pd.read_csv(DATA_DIR + "训练数据-ccf_first_round_shop_info.csv")
user_shop_behavior = pd.read_csv(DATA_DIR + "训练数据-ccf_first_round_user_shop_behavior.csv")
datasets_predict = pd.read_csv(DATA_DIR + "AB榜测试集-evaluation_public.csv")

# STEP 2.
print "2. get mall_id list"
mall_id_list = list(shop_infos.mall_id.unique())

# STEP 3.
print "3. get datasets sample in and out"
datasets_t = pd.merge(user_shop_behavior, shop_infos[['shop_id', 'mall_id']], on=['shop_id'])
datasets_p = pd.read_csv(DATA_DIR + "AB榜测试集-evaluation_public.csv")

# STEP 4.
print "4. loop mall_id to construct model"
accuracy_list = []
for m_id in mall_id_list:
    print "4.0 predict for mall :", m_id

    print "4.1 separate datasets"
    shop_infos_m = shop_infos[shop_infos.mall_id == m_id]
    shop_infos_m.index = range(shop_infos_m.shape[0])
    datasets_t_m = datasets_t[datasets_t.mall_id == m_id].copy()
    datasets_t_m.insert(0, 'row_id', np.nan)
    datasets_t_m = datasets_t_m.loc[:, ['row_id', 'user_id', 'mall_id', 'time_stamp',
                                        'longitude', 'latitude', 'wifi_infos', 'shop_id']]
    datasets_p_m = datasets_p[datasets_p.mall_id == m_id].copy()
    datasets_p_m.loc[:, 'shop_id'] = np.nan

    datasets_m = pd.concat([datasets_t_m, datasets_p_m], axis=0, ignore_index=True)
    datasets_m.insert(0, 'sample_id', range(datasets_m.shape[0]))

    shop_le = LabelEncoder()
    shop_le.fit([s for s in datasets_m.shop_id.values if isinstance(s, str)])

    del datasets_t_m
    del datasets_p_m
    gc.collect()

    print "4.2 add features"
    # add location feature
    for k in range(shop_infos_m.shape[0]):
        long_diff_name = 'lg_diff_' + shop_infos_m.loc[k, 'shop_id']
        long_diff = 1000 * (datasets_m.loc[:, 'longitude'] - shop_infos_m.loc[k, 'longitude'])
        datasets_m.loc[:, long_diff_name] = MinMaxScaler().fit_transform(long_diff.values.reshape(-1, 1))
        lati_diff_name = 'lt_diff_' + shop_infos_m.loc[k, 'shop_id']
        lati_diff = 1000 * (datasets_m.loc[:, 'latitude'] - shop_infos_m.loc[k, 'latitude'])
        datasets_m.loc[:, lati_diff_name] = MinMaxScaler().fit_transform(lati_diff.values.reshape(-1, 1))

    # add time feature
    datasets_m.loc[:, 'wday'] = datasets_m.time_stamp.apply(lambda t: time.strptime(t, '%Y-%m-%d %H:%M').tm_wday)
    datasets_m.loc[:, 'hour'] = datasets_m.time_stamp.apply(lambda t: time.strptime(t, '%Y-%m-%d %H:%M').tm_hour)
    datasets_m.loc[:, 'min'] = datasets_m.time_stamp.apply(lambda t: time.strptime(t, '%Y-%m-%d %H:%M').tm_min)
    datasets_m.loc[:, 'time_str'] = map(lambda *xlist: "-".join([str(x) for x in xlist]),
                                        datasets_m.loc[:, 'wday'],
                                        datasets_m.loc[:, 'hour'],
                                        datasets_m.loc[:, 'min'])

    dd = pd.get_dummies(datasets_m.time_str)
    dd.columns = ['time_str_' + str(d) for d in dd.columns]

    datasets_m = pd.concat([datasets_m, dd], axis=1, ignore_index=False)
    del datasets_m['wday']
    del datasets_m['hour']
    del datasets_m['min']
    del datasets_m['time_str']

    # add wifi feature
    sample_id_list = []
    wifi_id_list = []
    signal_power_list = []
    signal_flag_list = []
    sample_size = datasets_m.shape[0]
    for i in range(sample_size):
        wifi_info = datasets_m.wifi_infos[i].split(';')
        sample_id = i
        for w in wifi_info:
            w_values = w.split('|')
            wifi_id_list.append(w_values[0])
            signal_power_list.append(float(w_values[1]))
            signal_flag_list.append(w_values[2])
            sample_id_list.append(sample_id)

    wifi_feat = pd.DataFrame({'sample_id': sample_id_list,
                              'signal_power': signal_power_list,
                              'signal_flag': signal_flag_list,
                              'wifi_id': wifi_id_list},
                             columns=['sample_id', 'wifi_id', 'signal_power', 'signal_flag'])
    wifi_feat['signal_flag'] = wifi_feat.signal_flag.apply(lambda x: 1 if x == 'true' else 0)
    wifi_feat['signal_power'] = MinMaxScaler().fit_transform(wifi_feat['signal_power'].values.reshape(-1, 1))

    wifi_flag = pd.pivot_table(wifi_feat, index='sample_id', columns='wifi_id', values='signal_flag').fillna(0)
    wifi_power = pd.pivot_table(wifi_feat, index='sample_id', columns='wifi_id', values='signal_power').fillna(0)
    del wifi_feat

    datasets_m = pd.concat([datasets_m, wifi_power, wifi_flag], axis=1, ignore_index=False)
    del wifi_power
    del wifi_flag
    gc.collect()

    print "4.3 project features"
    rp = SparseRandomProjection(n_components=500)
    projected_data = pd.DataFrame(data=rp.fit_transform(datasets_m.iloc[:, 9:]))
    datasets_pj = pd.concat([datasets_m.iloc[:, 0:9], projected_data], axis=1)
    del rp
    del projected_data
    del datasets_m
    gc.collect()

    datasets_sample_in = datasets_pj[datasets_pj.shop_id.notnull()].copy()
    datasets_sample_in.loc[:, 'shop_id'] = shop_le.transform(datasets_sample_in.shop_id.values)

    X_train, X_validate, y_train, y_validate = train_test_split(datasets_sample_in.iloc[:, 9:],
                                                                datasets_sample_in.iloc[:, 8],
                                                                test_size=0.05)

    datasets_sample_out = datasets_pj[datasets_pj.row_id.notnull()].copy().reset_index(drop=True)
    datasets_sample_out.loc[:, 'row_id'] = datasets_sample_out.loc[:, 'row_id'].astype(int).astype(str)
    del datasets_pj
    gc.collect()

    print "4.4 train voting models"
    vtc = VotingClassifier(estimators=[('rf', rf), ('et', et), ('bg', bg)],
                           voting='soft',
                           weights=[1, 1, 1],
                           flatten_transform=True)
    vtc.fit(X_train, y_train)

    print "shop predict: "
    acc = [accuracy_score(vtc.predict(X_train), y_train), accuracy_score(vtc.predict(X_validate), y_validate)]
    accuracy_list.append(acc)
    print acc

    p_sample_out = Series(shop_le.inverse_transform(vtc.predict(datasets_sample_out.ix[:, 9:])))
    predict_pd = pd.concat([datasets_sample_out.loc[:, 'row_id'], p_sample_out], axis=1, ignore_index=True)
    predict_pd.columns = ['row_id', 'shop_id']
    print "p_result: "
    print predict_pd.head()
    print "p_result shape: ", predict_pd.shape
    predict_pd.to_csv(RET_DIR + "/{}.csv".format(m_id), index=False)

# STEP 5
print "5. train for every mall finished."
tra_acc_list = [ac[0] for ac in accuracy_list]
val_acc_list = [ac[1] for ac in accuracy_list]
print "train average acc: ", sum(tra_acc_list) * 1.0 / len(tra_acc_list)
print "validate average acc: ", sum(val_acc_list) * 1.0 / len(val_acc_list)



