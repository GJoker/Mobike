# -*- coding:utf-8 -*-
import os
import gc
import time
import pickle
import Geohash
import numpy as np
import pandas as pd

train_path = 'data/train.csv'
test_path = 'data/test_csv'
cache_path = 'mobike_cache_1/'

def get_sample(train, test):
  result_path = cache_path + 'sample_%d.hdf' % (train.shape[0] * test.shape[0])



def make_train_set(train, test):
  result = get_sample(train, test)

if __name__ == '__main__':
  t0 = time.time()
  train = pd.read_csv(train_path)
  test = pd.read_csv(test_path)
  train1 = train[(train['starttime'] < '2017-05-23 00:00:00')]
  train2 = train[(train['starttime'] >= '2017-05-23 00:00:00')]
  train2.loc[:, 'geohashed_start_loc'] = np.nan
  test.loc[:, 'geohashed_start_loc'] = np.nan

  train_feat = make_train_set(train1, train2)

