# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:12:01 2022

@author: gaozhongzhanshi
"""
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
data1 = pd.read_csv('C:\pocedure\python\lib\DataSet1.csv')
# 标记标签量：没有消费返回0；收到优惠券15天内消费，返回1；收到优惠券超过15天消费返回-1
data1.label.replace(-1, 0, inplace=True)
data2 = pd.read_csv('C:\pocedure\python\lib\DataSet2.csv')
data2.label.replace(-1, 0, inplace=True)
data3 = pd.read_csv('C:\pocedure\python\lib\DataSet3.csv')
# 删除重复行数据
data1.drop_duplicates(inplace=True)
data2.drop_duplicates(inplace=True)
data12 = pd.concat([data1, data2], axis=0)
data12_yq = data12.label
data12_xp = data12.drop(['user_id', 'label', 'day_gap_before', 'coupon_id', 'day_gap_after'], axis=1)
data3.drop_duplicates(inplace=True)
data3_pre = data3[['user_id', 'coupon_id', 'date_received']]
data3_xr = data3.drop(['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after'], axis=1)
data_train = xgb.DMatrix(data12_xp, label=data12_yq)
dataTest = xgb.DMatrix(data3_xr)
# 训练模型xgboost
params = {'booster': 'gbtree',
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'silent': 1,
          'eta': 0.01,
          'max_depth': 5,
          'min_child_weight': 1,
          'gamma': 0,
          'lambda': 1,
          'colsample_bylevel': 0.7,
          'colsample_bytree': 0.7,
          'subsample': 0.9,
          'scale_pos_weight': 1}
wl = [(data_train, 'train')]
# eval：获取返回值
model_xgb = xgb.train(params, data_train, num_boost_round=3633, evals=wl, early_stopping_rounds=50)  
model_xgb.save_model(r'C:\pocedure\python\lib\xgbmodel')
# 测试集预测
model_xgb = xgb.Booster()
model_xgb.load_model(r'C:\pocedure\python\lib\xgbmodel')
# predict
data3_pre = data3_pre
data3_pre['label'] = model_xgb.predict(dataTest)
# 标签归一化在[0，1]
data3_pre.label = MinMaxScaler(copy=True, feature_range=(0, 1)).fit_transform(data3_pre.label.values.reshape(-1, 1))
data3_pre.sort_values(by=['coupon_id', 'label'], inplace=True)
data3_pre.to_csv(r"C:\pocedure\python\lib\preds.csv", index=None, header=None)










