# -*- coding:utf-8 -*-
import os
import gc
import time
import pickle
import Geohash
import numpy as np
import pandas as pd

train_path = 'data/train.csv'
test_path = 'data/test.csv'
cache_path = 'cache_1/'

def get_sample(train):
  result_path = cache_path + 'sample_%d.hdf' % (train.shape[0] * test.shape[0])
  if os.path.exists(result_path):
    result = pd.read_hdf(result_path, 'w')
  else:
    user_eloc = train[['userid', 'geohashed_end_loc']].drop_duplicates()
    user_eloc.rename(columns={'geohashed_end_loc': 'label'}, inplace=True)
    result = pd.merge(train, user_eloc, on='userid', how='left')
    sloc_to_eloc = train[['geohashed_start_loc', 'geohashed_end_loc']].drop_duplicates()
    sloc_to_eloc.rename(columns={'geohashed_end_loc': 'label'}, inplace=True)
    result = pd.merge(result, sloc_to_eloc, on='geohashed_start_loc', how='left')
  return result


def make_train_set(train):
  result = get_sample(train)
  return result

if __name__ == '__main__':
  t0 = time.time()
  train = pd.read_csv(train_path).drop('biketype', axis=1)
  test = pd.read_csv(test_path).drop('biketype', axis=1)

  train_set = make_train_set(train)
  print train_set.shape()

                          