# -*- coding:utf-8 -*-

from pandas import Series, DataFrame
import pandas as pd
import numpy as np

# s = Series([100, 'python', 'ray', 'gao'])
# print s
# print type(s.values)
# print type(s.index)
#
# s2 = Series(data=[100, 'python', 'ray', 'gao'], index=[2, 'gg', 4, 'ghad'])
# print s2

#  orderid  userid  bikeid  biketype  starttime  geohashed_start_loc  geohashed_end_loc

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
# print len(set(train.orderid))
# print len(set(train.userid))
# print len(set(train.bikeid))
# print len(set(train.geohashed_start_loc))
# print len(set(train.geohashed_end_loc))
#
# print len(set(test.orderid))
# print len(set(test.userid))
# print len(set(test.bikeid))
# print len(set(test.geohashed_start_loc))

test.loc[:, u'userid'] = np.nan
print test

# total = pd.concat([train, test], axis=0)

# print total
# print len(set(total.geohashed_start_loc))
# print total.sort_values(by=u'bikeid')

# print train.sort_values(by=u'userid')
# print test.sort_values(by=u'userid')