# -*- coding:utf-8 -*-

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import Geohash
import matplotlib.pyplot as plt
import statsmodels.api as sm

# date = pd.date_range('20170822', periods=7)
# print date
#
# df = pd.DataFrame(data=np.random.randn(7, 4), index=date, columns=['A', 'B', 'C', 'D'])
# print df

# s = Series([100, 'python', 'ray', 'gao'])
# print s
# print type(s.values)
# print type(s.index)
#
# s2 = Series(data=[100, 'python', 'ray', 'gao'], index=[2, 'gg', 4, 'ghad'])
# print s2

#  orderid  userid  bikeid  biketype  starttime  geohashed_start_loc  geohashed_end_loc
#
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# locs = [loc[0:3] for loc in train['geohashed_start_loc'].values]
# for loc in set(locs):
#   print '%s: %d: %.4f, %.4f' % (loc, locs.count(loc), round(Geohash.decode_exactly(loc)[0], 4), round(Geohash.decode_exactly(loc)[1], 4))
#
# for loc in train['geohashed_start_loc'].values:
#   if 'wsk' == loc[0:3]:
#     print '%s: %.4f, %.4f' % (loc, Geohash.decode_exactly(loc)[0], Geohash.decode_exactly(loc)[1])

# print len(set([loc[0:3] for loc in train['geohashed_start_loc'].values]))
# print train.sort_values(by='geohashed_start_loc')
# print set([loc[0:3] for loc in train['geohashed_end_loc'].values])
# print set([loc[0:3] for loc in test['geohashed_start_loc'].values])
# all = pd.concat([train['geohashed_start_loc'], train['geohashed_end_loc']])

# print len(set(all))
# print len(set(train.userid))
# print len(set(train.bikeid))
# print len(set(train.geohashed_start_loc))
# print len(set(train.geohashed_end_loc))
#
# print len(set(test.orderid))
# print len(set(test.userid))
# print len(set(test.bikeid))
# print len(set(test.geohashed_start_loc))

# test.loc[:, u'userid'] = np.nan
# print test

# print len(set(pd.concat([test['geohashed_start_loc'], all])))# - len(set(all))

# print total
# print len(set(total['geohashed_start_loc']))
# print total.sort_values(by=u'bikeid')

# print train.sort_values(by=u'userid')
# print test.sort_values(by=u'userid')

# df = DataFrame({'key1':['a','a','b','b','a', 'a', 'c'],
#                    'key2':['one','two','one','two','one', 'three', 'one']})
#
# print df.rename(columns={'key1': 'key2', 'key2':'key1'}, inplace=True)
#
# a = df.groupby(['key1','key2'], as_index=False)['key2'].agg({'a': 'count'})
#
# a.sort_values('a', inplace=True)
# print a
# a = a.groupby('key1').tail(2)
# print a

# train = pd.read_csv('./data/train.csv')
# # print type(train['starttime'].values)
# # train['starttime'] = pd.to_datetime(train['starttime']).dt.date
# # print train
#
# test = train.groupby(['starttime'], as_index=False)['starttime'].agg({'count':'count'})
# train1 = train[(train['starttime'] < '2017-05-23 00:00:00')]
# train2 = train[(train['starttime'] >= '2017-05-23 00:00:00')]
#
user_count = train.groupby('userid', as_index=False)['geohashed_start_loc'].agg({'user_count': 'count'})
print train[(train['userid'] == 2730)].groupby('geohashed_start_loc', as_index=False)['geohashed_start_loc'].agg({'number': 'count'})
# test_csv = train[(train['userid'] == 2730)].groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False)['geohashed_end_loc'].agg({'number': 'count'})
# test_csv.to_csv('test.csv')
# count = user_count.groupby('user_count', as_index=False)['user_count'].agg({'count': 'count'})
# insertRow = pd.DataFrame([[0, 0]], columns=['user_count', 'count'])
# count = pd.concat([insertRow, count], ignore_index=True)
# sumAll = count['count'].sum()
#
# xaxis = list(count['user_count'])
# yaxis = list(count['count']/sumAll)

# sm.distributions.ECDF()
#
# plt.plot(xaxis, yaxis)
# plt.show()

# sloc_eloc_count = train.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False)[
#       'geohashed_end_loc'].agg({'sloc_eloc_count': 'count'}) # 统计各个起始-终止区域个数
# sloc_eloc_count.sort_values('sloc_eloc_count', inplace=True)
# print sloc_eloc_count
# sloc_eloc_count = sloc_eloc_count.groupby('geohashed_start_loc').tail(3) # 统计各个起始点去的最多的3个终点
# result = pd.merge(train2[['orderid', 'geohashed_start_loc']], sloc_eloc_count, on='geohashed_start_loc', how='left')
# result = result[['orderid', 'geohashed_end_loc']]
#
# test_temp = train2.copy()
# test_temp.rename(columns={'geohashed_end_loc': 'label'}, inplace=True)
# result = pd.merge(result, test_temp, on='orderid', how='left')
# result['label'] = (result['label'] == result['geohashed_end_loc']).astype(int)
# print result['label'].sum()
# print result.shape[0]
# print result
print Geohash.decode_exactly('wx4gn2s')
# 湖南岳阳 山东烟台 湖北鄂州 江西南昌 山西临汾 山东济南 天津 广西南宁 陕西西安 河北保定 宁夏银川 广东惠州 广东广州 广东揭阳 福建揭阳 河北唐山 四川绵阳 广东湛江 江苏南京 江苏无锡 河南郑州 上海 河北邢台
# 安徽合肥 云南红河 云南玉溪 浙江杭州 北京 湖南株洲 四川成都 四川资阳 福建福州 福建宁德
# for loc in locs:
#   print [round(i, 4) for i in Geohash.decode_exactly(loc)[0:2]]
