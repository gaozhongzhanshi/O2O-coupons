# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:11:34 2022

@author: gaozhongzhanshi
"""

import pandas as pd
import numpy as np
from datetime import date
import datetime as dt
import os

# 源数据路径
Data = 'C:\pocedure\python\lib'
# 预处理后数据存放路径
FeatureP = 'C:\pocedure\python\lib'
# 缺失值以字符串形式存储‘null’
off_train = pd.read_csv(os.path.join(Data, 'ccf_offline_stage1_train.csv'),header=0, keep_default_na=False)
off_train.columns = ['user_id', 'merchant_id', 'coupon_id','discount_rate', 'distance', 'date_received', 'date']
off_test = pd.read_csv(os.path.join(Data, 'ccf_offline_stage1_test_revised.csv'),header=0, keep_default_na=False)
off_test.columns = ['user_id', 'merchant_id', 'coupon_id','discount_rate', 'distance', 'date_received']
# 交叉训练集一：收到券的日期大于4月14日和小于5月14日(对这个时间段收到优惠券的人进行预测；没有领过的为无意义样本)
dataset1 = off_train[(off_train.date_received >= '20160414')& (off_train.date_received <= '20160514')]
# 交叉训练集一特征：线下数据中领券和用券日期大于1月1日和小于4月13日（data数据必须在这个时间段里；data可能有具体时间，也可能是null，所以就有了中间的或；如果data是null，那就让在在这个时间段领过优惠券的人作为数据吧）
# 要不然在这个时间段有消费；要不然在这个时间段领优惠券
feature1 = off_train[(off_train.date >= '20160101') & (off_train.date <= '20160413')
                     | ((off_train.date == 'null') & (off_train.date_received >= '20160101')
                        & (off_train.date_received <= '20160413'))]
# 交叉训练集二：收到券的日期大于5月15日和小于6月15日
dataset2 = off_train[(off_train.date_received >= '20160515')& (off_train.date_received <= '20160615')]
# 交叉训练集二特征：线下数据中领券和用券日期大于2月1日和小于5月14日
feature2 = off_train[(off_train.date >= '20160201') & (off_train.date <= '20160514')
                     | ((off_train.date == 'null') & (off_train.date_received >= '20160201')
                        & (off_train.date_received <= '20160514'))]
# 测试集
dataset3 = off_test
# 测试集特征 :线下数据中领券和用券日期大于3月15日和小于6月30日的
feature3 = off_train[((off_train.date >= '20160315') & (off_train.date <= '20160630'))
                     | ((off_train.date == 'null') & (off_train.date_received >= '20160315')
                        & (off_train.date_received <= '20160630'))]

def g_datereceived_gap(wjh):
    wjh = wjh.split(':')
    # ().days:按天数返回
    return (date(int(wjh[0][0:4]), int(wjh[0][4:6]), int(wjh[0][6:8])) - date(int(wjh[1][0:4]), int(wjh[1][4:6]),int(wjh[1][6:8]))).days

def g_discnt_rate(count):
    count = str(count)
    count = count.split(':')
    if len(count) == 1:
        return float(count[0])
    else:
        return 1.0-float(count[1])/float(count[0])

def g_cnt_p(people):
    people = str(people)
    people = people.split(':')
    if len(people) == 1:
        return 'null'
    else:
        return int(people[0])
    
def is_mj(ww):
    ww = str(ww)
    ww = ww.split(':')
    if len(ww) == 1:
        return 0
    else:
        return 1

def g_discnt_ml(money):
    money = str(money)
    money = money.split(':')
    if len(money) == 1:
        return 'null'
    else:
        return int(money[1])

def first_last(y):
    if y == 0:
        return 1
    elif y > 0:
        return 0
    else:
        return -1

def g_dgapb(wjh):
    # 同一优惠券之前收到的最小间隔，之前没有收到返回-1
    date_received, dates = wjh.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        # 将时间差转化为天数
        this_gap = (dt.date(int(date_received[0:4]), int(date_received[4:6]), int(
            date_received[6:8]))-dt.date(int(d[0:4]), int(d[4:6]), int(d[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)

def g_dgapa(wjh):
    # 同一优惠券之后收到的最小间隔，之后没有收到返回-1
    date_received, dates = wjh.split('-')
    dates = dates.split(':')
    gaps = []
    for d in dates:
        this_gap = (dt.datetime(int(d[0:4]), int(d[4:6]), int(d[6:8]))-dt.datetime(
            int(date_received[0:4]), int(date_received[4:6]), int(date_received[6:8]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return -1
    else:
        return min(gaps)

def g_umf(feature):
    # feature中user_id和merchant_id是不会缺失的
    all_um = feature[['user_id', 'merchant_id']].copy()
    all_um.drop_duplicates(inplace=True)

    # 一个客户在一个商家一共买的次数
    w = feature[['user_id', 'merchant_id', 'date']].copy()
    w = w[w.date != 'null'][['user_id', 'merchant_id']]
    w['user_merchant_buy_total'] = 1
    w = w.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    # 去重没有任何意义，因为agg(sum)的时候已经相当于去重了，reset_index之后也不会变回agg前的行数
    w.drop_duplicates(inplace=True)

    # 一个客户在一个商家一共收到的优惠券
    w1 = feature[['user_id', 'merchant_id', 'coupon_id']]
    w1 = w1[w1.coupon_id != 'null'][['user_id', 'merchant_id']]
    w1['user_merchant_received'] = 1
    w1 = w1.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    w1.drop_duplicates(inplace=True)

    # 一个客户在一个商家使用优惠券购买的次数
    w2 = feature[['user_id', 'merchant_id', 'date', 'date_received']]
    w2 = w2[(w2.date != 'null') & (w2.date_received != 'null')][['user_id', 'merchant_id']]
    w2['user_use_coupon'] = 1
    w2 = w2.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    w2.drop_duplicates(inplace=True)

    # 一个客户在一个商家浏览的次数（领过优惠券或者买过商品）
    w3 = feature[['user_id', 'merchant_id']]
    w3['user_merchant_any'] = 1
    w3 = w3.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    w3.drop_duplicates(inplace=True)

    # 一个客户在一个商家没有使用优惠券购买的次数
    w4 = feature[['user_id', 'merchant_id', 'date', 'coupon_id']]
    w4 = w4[(w4.date != 'null') & (w4.coupon_id == 'null')][['user_id', 'merchant_id']]
    w4['user_merchant_buy_common'] = 1
    w4 = w4.groupby(['user_id', 'merchant_id']).agg('sum').reset_index()
    w4.drop_duplicates(inplace=True)

    u_m = pd.merge(all_um, w, on=['user_id', 'merchant_id'], how='left')
    u_m = pd.merge(u_m, w1, on=['user_id', 'merchant_id'], how='left')
    u_m = pd.merge(u_m, w2, on=['user_id', 'merchant_id'], how='left')
    u_m = pd.merge(u_m, w3, on=['user_id', 'merchant_id'], how='left')
    u_m = pd.merge(u_m, w4, on=['user_id', 'merchant_id'], how='left')
    u_m.user_use_coupon = u_m.user_use_coupon.replace(np.nan, 0)
    u_m.user_merchant_buy_common = u_m.user_merchant_buy_common.replace(np.nan, 0)
    u_m['user_merchant_coupon_transfer_rate'] = u_m.user_use_coupon.astype(
        'float') / u_m.user_merchant_received.astype('float')
    u_m['user_merchant_coupon_buy_rate'] = u_m.user_use_coupon.astype(
        'float') / u_m.user_merchant_buy_total.astype('float')
    u_m['user_merchant_rate'] = u_m.user_merchant_buy_total.astype(
        'float') / u_m.user_merchant_any.astype('float')
    u_m['user_merchant_common_buy_rate'] = u_m.user_merchant_buy_common.astype(
        'float') / u_m.user_merchant_buy_total.astype('float')
    return u_m

def g_ufg(feature):
    # for dataset3
    user = feature[['user_id', 'merchant_id', 'coupon_id','discount_rate', 'distance', 'date_received', 'date']].copy()
    w = user[['user_id']].copy()
    w.drop_duplicates(inplace=True)

    # 客户一共买的商品
    w1 = user[user.date != 'null'][['user_id', 'merchant_id']].copy()
    w1.drop_duplicates(inplace=True)
    w1.merchant_id = 1
    w1 = w1.groupby('user_id').agg('sum').reset_index()
    w1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)

    # 客户使用优惠券线下购买距离商店的最小距离
    w2 = user[(user.date != 'null') & (user.coupon_id != 'null')][['user_id', 'distance']]
    w2.replace('null', -1, inplace=True)
    w2.distance = w2.distance.astype('int')
    w2.replace(-1, np.nan, inplace=True)
    w3 = w2.groupby('user_id').agg('min').reset_index()
    w3.rename(columns={'distance': 'user_min_distance'}, inplace=True)

    # 客户使用优惠券线下购买距离商店的最大距离
    w4 = w2.groupby('user_id').agg('max').reset_index()
    w4.rename(columns={'distance': 'user_max_distance'}, inplace=True)

    # 客户使用优惠券线下购买距离商店的平均距离
    w5 = w2.groupby('user_id').agg('mean').reset_index()
    w5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)

    # 客户使用优惠券线下购买距离商店的中间距离
    w6 = w2.groupby('user_id').agg('median').reset_index()
    w6.rename(columns={'distance': 'user_median_distance'}, inplace=True)

    # 客户使用优惠券购买的次数
    w7 = user[(user.date != 'null') & (user.coupon_id != 'null')][['user_id']]
    w7['buy_use_coupon'] = 1
    w7 = w7.groupby('user_id').agg('sum').reset_index()

    # 客户购买任意商品的总次数
    w8 = user[user.date != 'null'][['user_id']]
    w8['buy_total'] = 1
    w8 = w8.groupby('user_id').agg('sum').reset_index()

    # 客户收到优惠券的总数
    w9 = user[user.coupon_id != 'null'][['user_id']]
    w9['coupon_received'] = 1
    w9 = w9.groupby('user_id').agg('sum').reset_index()

    # 客户从收优惠券到消费的时间间隔
    w10 = user[(user.date_received != 'null') & (user.date != 'null')][['user_id', 'date_received', 'date']]
    w10['user_date_datereceived_gap'] = w10.date + ':' + w10.date_received
    w10.user_date_datereceived_gap = w10.user_date_datereceived_gap.apply(g_datereceived_gap)
    w10 = w10[['user_id', 'user_date_datereceived_gap']]

    # 客户从收优惠券到消费的平均时间间隔
    w11 = w10.groupby('user_id').agg('mean').reset_index()
    w11.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'}, inplace=True)
    # 客户从收优惠券到消费的最小时间间隔
    w12 = w10.groupby('user_id').agg('min').reset_index()
    w12.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)
    # 客户从收优惠券到消费的最大时间间隔
    w13 = w10.groupby('user_id').agg('max').reset_index()
    w13.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)

    u_f = pd.merge(w, w1, on='user_id', how='left')
    u_f = pd.merge(u_f, w3, on='user_id', how='left')
    u_f = pd.merge(u_f, w4, on='user_id', how='left')
    u_f = pd.merge(u_f, w5, on='user_id', how='left')
    u_f = pd.merge(u_f, w6, on='user_id', how='left')
    u_f = pd.merge(u_f, w7, on='user_id', how='left')
    u_f = pd.merge(u_f, w8, on='user_id', how='left')
    u_f = pd.merge(u_f, w9, on='user_id', how='left')
    u_f = pd.merge(u_f, w11, on='user_id', how='left')
    u_f = pd.merge(u_f, w12, on='user_id', how='left')
    u_f = pd.merge(u_f, w13, on='user_id', how='left')
    u_f.count_merchant = u_f.count_merchant.replace(np.nan, 0)
    u_f.buy_use_coupon = u_f.buy_use_coupon.replace(np.nan, 0)
    u_f['buy_use_coupon_rate'] = u_f.buy_use_coupon.astype('float') / u_f.buy_total.astype('float')
    u_f['user_coupon_transfer_rate'] = u_f.buy_use_coupon.astype('float') / u_f.coupon_received.astype('float')
    # 先除，再将缺失值转换为0，防止除的时候0作为分母发生错误；np.nan在做除法的时候直接跳过，结果也为np.nan
    u_f.buy_total = u_f.buy_total.replace(np.nan, 0)
    u_f.coupon_received = u_f.coupon_received.replace(np.nan, 0)
    return u_f

def g_cfg(datasetw, feature):
    # 为了求得每个feature中date最大的日期，其会被用在求days_distance字段
    # t:feature中最大消费时间
    t = feature[feature['date'] != 'null']['date'].unique()
    t = max(t)

    # weekday返回一周的第几天
    datasetw['day_of_week'] = datasetw.date_received.astype('str').apply(
        lambda x: date(int(x[0:4]), int(x[4:6]), int(x[6:8])).weekday()+1)
    # 显示时间是几月
    datasetw['day_of_month'] = datasetw.date_received.astype('str').apply(lambda x: int(x[6:8]))
    # 显示时期和截止日之间的天数
    datasetw['days_distance'] = datasetw.date_received.astype('str').apply(lambda x: (date(
        int(x[0:4]), int(x[4:6]), int(x[6:8]))-date(int(t[0:4]), int(t[4:6]), int(t[6:8]))).days)
    # 显示满了多少钱后开始减
    datasetw['discount_man'] = datasetw.discount_rate.apply(g_cnt_p)
    # 显示满减的减少的钱
    datasetw['discount_jian'] = datasetw.discount_rate.apply(g_discnt_ml)
    # 返回优惠券是否是满减券
    datasetw['is_man_jian'] = datasetw.discount_rate.apply(is_mj)
    # 显示打折力度
    datasetw['discount_rate'] = datasetw.discount_rate.apply(g_discnt_rate)
    d = datasetw[['coupon_id']]
    d['coupon_count'] = 1
    # 显示每一种优惠券的数量
    d = d.groupby('coupon_id').agg('sum').reset_index()
    datasetw = pd.merge(datasetw, d, on='coupon_id', how='left')
    return datasetw

def g_mfg(feature):
    # merchant_id和user_id不会是‘null’
    merchant = feature[['merchant_id', 'coupon_id','distance', 'date_received', 'date']].copy()
    w = merchant[['merchant_id']].copy()
    # 删除重复行数据
    w.drop_duplicates(inplace=True)

    # 卖出的商品
    w1 = merchant[merchant.date != 'null'][['merchant_id']].copy()
    w1['total_sales'] = 1
    # 每个商品的销售数量
    w1 = w1.groupby('merchant_id').agg('sum').reset_index()

    # 使用了优惠券消费的商品，正样本
    w2 = merchant[(merchant.date != 'null') & (merchant.coupon_id != 'null')][['merchant_id']].copy()
    w2['sales_use_coupon'] = 1
    w2 = w2.groupby('merchant_id').agg('sum').reset_index()

    # 商品的优惠券的总数量
    w3 = merchant[merchant.coupon_id != 'null'][['merchant_id']].copy()
    w3['total_coupon'] = 1
    w3 = w3.groupby('merchant_id').agg('sum').reset_index()

    # 商品销量和距离的关系
    w4 = merchant[(merchant.date != 'null') & (merchant.coupon_id != 'null')][['merchant_id', 'distance']].copy()
    # 下面三行代码的主要作用就是为了将distance字段的数据类型转化为int
    # 把数据中的null值全部替换为-1
    w4.replace('null', -1, inplace=True)
    w4.distance = w4.distance.astype('int')
    # 再把数据中的-1全部替换为NaN
    # np.nan是float的子类
    w4.replace(-1, np.nan, inplace=True)

    # 返回所有使用优惠券购买该商品的用户中离商品的距离最小值
    w5 = w4.groupby('merchant_id').agg('min').reset_index()
    w5.rename(columns={'distance': 'merchant_min_distance'}, inplace=True)

    # 返回用户离商品的距离最大值
    w6 = w4.groupby('merchant_id').agg('max').reset_index()
    w6.rename(columns={'distance': 'merchant_max_distance'}, inplace=True)
    # 返回距离的平均值
    w7 = w4.groupby('merchant_id').agg('mean').reset_index()
    w7.rename(columns={'distance': 'merchant_mean_distance'}, inplace=True)
    # 返回距离的中位值
    w8 = w4.groupby('merchant_id').agg('median').reset_index()
    w8.rename(columns={'distance': 'merchant_median_distance'}, inplace=True)

    m_r = pd.merge(w, w1, on='merchant_id', how='left')
    m_r = pd.merge(m_r, w2, on='merchant_id', how='left')
    m_r = pd.merge(m_r, w3, on='merchant_id', how='left')
    m_r = pd.merge(m_r, w5, on='merchant_id', how='left')
    m_r = pd.merge(m_r, w6, on='merchant_id', how='left')
    m_r = pd.merge(m_r, w7, on='merchant_id', how='left')
    m_r = pd.merge(m_r, w8, on='merchant_id', how='left')

    # 将数据中的NaN用0来替换
    m_r.sales_use_coupon = m_r.sales_use_coupon.replace(np.nan, 0)
    # 优惠券的使用率
    m_r['merchant_coupon_transfer_rate'] = m_r.sales_use_coupon.astype('float') / m_r.total_coupon
    # 即卖出商品中使用优惠券的占比
    m_r['coupon_rate'] = m_r.sales_use_coupon.astype('float') / m_r.total_sales
    # 将数据中的NaN用0来替换
    m_r.total_coupon = m_r.total_coupon.replace(np.nan, 0)

    return m_r

def g_of(dataset):
    # t:每个用户收到优惠券的数量(因为此数据集中每个人都是收到优惠券的人)
    w = dataset[['user_id']].copy()
    w['this_month_user_receive_all_coupon_count'] = 1
    w = w.groupby('user_id').agg('sum').reset_index()
    # t1：用户领取指定优惠券的数量
    w1 = dataset[['user_id', 'coupon_id']].copy()
    w1['this_month_user_receive_same_coupn_count'] = 1
    w1 = w1.groupby(['user_id', 'coupon_id']).agg('sum').reset_index()
    # t2:用户领取特定优惠券的最大时间和最小时间
    w2 = dataset[['user_id', 'coupon_id', 'date_received']].copy()
    # astype:全部转换为str型
    w2.date_received = w2.date_received.astype('str')
    # 如果出现相同的用户接收相同的优惠券在接收时间上用‘：’连接上第n次接受优惠券的时间
    # t2处理后为3列
    w2 = w2.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    # 将接收时间的一组按着':'分开，这样就可以计算接受了优惠券的数量,apply是合并
    # apply:对t2.date_received每一个元素应用函数
    w2['receive_number'] = w2.date_received.apply(lambda s: len(s.split(':')))
    w2 = w2[w2.receive_number > 1]
    # 最大接受的日期
    w2['max_date_received'] = w2.date_received.apply(lambda x: max([int(d) for d in x.split(':')]))
    # 最小的接收日期
    w2['min_date_received'] = w2.date_received.apply(lambda x: min([int(d) for d in x.split(':')]))
    w2 = w2[['user_id', 'coupon_id', 'max_date_received', 'min_date_received']]

    w3 = dataset[['user_id', 'coupon_id', 'date_received']]
    # 将两表融合只保留左表数据,这样得到的表，相当于保留了最近接收时间和最远接受时间
    # 缺失值用nan填充
    w3 = pd.merge(w3, w2, on=['user_id', 'coupon_id'], how='left')
    # 这个优惠券最近接受时间
    # 对于max_date_received为nan的，this_month_user_receive_same_coupon_lastone也为nan
    w3['this_month_user_receive_same_coupon_lastone'] = w3.max_date_received - \
        w3.date_received.astype(int)
    # 这个优惠券最远接受时间
    w3['this_month_user_receive_same_coupon_firstone'] = w3.date_received.astype(
        int)-w3.min_date_received

    w3.this_month_user_receive_same_coupon_lastone = w3.this_month_user_receive_same_coupon_lastone.apply(first_last)
    w3.this_month_user_receive_same_coupon_firstone = w3.this_month_user_receive_same_coupon_lastone.apply(first_last)
    # this_month_user_receive_same_coupon_lastone中nan（某一用户对于某一优惠券只领过一次）为-1，是为1，不是为0
    w3 = w3[['user_id', 'coupon_id', 'date_received', 'this_month_user_receive_same_coupon_lastone','this_month_user_receive_same_coupon_firstone']]

    # 提取第四个特征,一个用户当天所接收到的所有优惠券的数量
    w4 = dataset[['user_id', 'date_received']].copy()
    w4['this_day_receive_all_coupon_count'] = 1
    w4 = w4.groupby(['user_id', 'date_received']).agg('sum').reset_index()

    # 提取第五个特征,一个用户当天所接收到相同优惠券的数量
    w5 = dataset[['user_id', 'coupon_id', 'date_received']].copy()
    w5['this_day_user_receive_same_coupon_count'] = 1
    w5 = w5.groupby(['user_id', 'coupon_id', 'date_received']).agg('sum').reset_index()
    # 一个用户不同优惠券 的接受时间
    # 某一用户对同一优惠券的所有领取时间
    w6 = dataset[['user_id', 'coupon_id', 'date_received']].copy()
    w6.date_received = w6.date_received.astype('str')
    w6 = w6.groupby(['user_id', 'coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    w6.rename(columns={'date_received': 'dates'}, inplace=True)

    w7 = dataset[['user_id', 'coupon_id', 'date_received']]
    w7 = pd.merge(w7, w6, on=['user_id', 'coupon_id'], how='left')
    w7['date_received_date'] = w7.date_received.astype('str')+'-'+w7.dates
    w7['day_gap_before'] = w7.date_received_date.apply(g_dgapb)
    w7['day_gap_after'] = w7.date_received_date.apply(g_dgapa)
    w7 = w7[['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after']]
    of = pd.merge(w1, w, on='user_id')
    of = pd.merge(of, w3, on=['user_id', 'coupon_id'])
    of = pd.merge(of, w4, on=['user_id', 'date_received'])
    of = pd.merge(of, w5, on=['user_id', 'coupon_id', 'date_received'])
    of = pd.merge(of, w7, on=['user_id', 'coupon_id', 'date_received'])
    return of

def g_label(k):
    k = k.split(':')
    if k[0] == 'null':
        return 0
    elif (date(int(k[0][0:4]), int(k[0][4:6]), int(k[0][6:8])) - date(int(k[1][0:4]), int(k[1][4:6]),
                                                                      int(k[1][6:8]))).days <= 15:
        return 1
    else:
        return -1

def DataProcess(dataset, feature, TrainFlag):
    other_feature = g_of(dataset)
    merchant = g_mfg(feature)
    user = g_ufg(feature)
    user_merchant = g_umf(feature)
    coupon = g_cfg(dataset, feature)
    dataset = pd.merge(coupon, merchant, on='merchant_id', how='left')
    # 相当于把coupon中的user_ID扩展为user
    dataset = pd.merge(dataset, user, on='user_id', how='left')
    # 对于用户-商品属性，user_id和merchant_id就是它的index，所以以其为标准merge
    dataset = pd.merge(dataset, user_merchant, on=['user_id', 'merchant_id'], how='left')
    dataset = pd.merge(dataset, other_feature, on=['user_id', 'coupon_id', 'date_received'], how='left')
    dataset.drop_duplicates(inplace=True)
    dataset.user_merchant_buy_total = dataset.user_merchant_buy_total.replace(np.nan, 0)
    dataset.user_merchant_any = dataset.user_merchant_any.replace(np.nan, 0)
    dataset.user_merchant_received = dataset.user_merchant_received.replace(np.nan, 0)
    dataset['is_weekend'] = dataset.day_of_week.apply(lambda x: 1 if x in (6, 7) else 0)
    weekday_dummies = pd.get_dummies(dataset.day_of_week)
    weekday_dummies.columns = ['weekday' + str(i + 1) for i in range(weekday_dummies.shape[1])]
    # 合并
    dataset = pd.concat([dataset, weekday_dummies], axis=1)
    if TrainFlag:
        dataset['date'] = dataset['date'].fillna('null')
        dataset['label'] = dataset.date.astype(
            'str') + ':' + dataset.date_received.astype('str')
        # 标记标签量：没有消费返回0；收到优惠券15天内消费，返回1；收到优惠券超过15天消费返回-1
        dataset.label = dataset.label.apply(g_label)
        # axis = 1：按列删除
        dataset.drop(['merchant_id', 'day_of_week', 'date','date_received', 'coupon_count'], axis=1, inplace=True)
    else:
        dataset.drop(['merchant_id', 'day_of_week','coupon_count'], axis=1, inplace=True)
    dataset = dataset.replace('null', np.nan)
    return dataset
DataSet1 = DataProcess(dataset1,feature1,True)
DataSet1.to_csv(os.path.join(Data,'DataSet1.csv'),index=None)
DataSet2 = DataProcess(dataset2,feature2,True)
DataSet2.to_csv(os.path.join(Data,'DataSet2.csv'),index=None)
# 3是测试集，所以不标记
DataSet3 = DataProcess(dataset3,feature3,False)
DataSet3.to_csv(os.path.join(Data,'DataSet3.csv'),index=None)

