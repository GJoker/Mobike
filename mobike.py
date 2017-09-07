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


def get_sloc_to_eloc(train, test):
  result_path = cache_path + 'sloc_to_eloc.hdf'
  if os.path.exists(result_path):
    result = pd.read_hdf(result_path, 'r')
  else:
    sloc_to_eloc = train[['geohashed_start_loc', 'geohashed_end_loc']].sort_values('geohashed_start_loc')
    sloc_to_eloc = train.groupby(['geohashed_start_loc', 'geohashed_end_loc'], as_index=False)[
      'geohashed_start_loc'].agg({'sloc_eloc_count': 'count'})
    sloc_to_eloc.sort_values('sloc_eloc_count', inplace=True)
    result = sloc_to_eloc.groupby('geohashed_start_loc').tail(3)
    result.rename(columns={'geohashed_end_loc': 'label'}, inplace=True)
    result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
  return result


def get_user_to_loc(train, test):
  result_path = cache_path + 'user_to_loc.hdf'
  if os.path.exists(result_path):
    result = pd.read_hdf(result_path, 'r')
  else:
    user_to_sloc = train[['userid', 'geohashed_start_loc']].drop_duplicates()
    user_to_sloc.rename(columns={'geohashed_start_loc': 'label'}, inplace=True)
    user_to_eloc = train[['userid', 'geohashed_end_loc']].drop_duplicates()
    user_to_eloc.rename(columns={'geohashed_end_loc': 'label'}, inplace=True)
    result = pd.concat([user_to_sloc, user_to_eloc]).drop_duplicates()
    result = pd.merge(test, result, how='left', on='userid')
    result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
  return result


def get_sample(train, test):
  result_path = cache_path + 'sample.hdf'
  if os.path.exists(result_path):
    result = pd.read_hdf(result_path, 'r')
  else:
    user_to_loc = get_user_to_loc(train, test)
    sloc_to_eloc = get_sloc_to_eloc(train, test)
    result = pd.merge(train, user_to_loc, on='userid', how='left')
    result = pd.merge(result, sloc_to_eloc, on='geohashed_start_loc', how='left').drop_duplicates()
    result.to_hdf(result_path, 'w', complib='blosc', complevel=5)
  return result


def make_train_set(train, test):
  result = get_sample(train, test)
  return result


if __name__ == '__main__':
  t0 = time.time()
  train = pd.read_csv(train_path).drop('biketype', axis=1)
  test = pd.read_csv(test_path).drop('biketype', axis=1)

  train1 = train[(train['starttime'] < '2017-05-21 00:00:00')]
  train2 = train[(train['starttime'] >= '2017-05-21 00:00:00')]

  train_set = make_train_set(train1, train2)

  # train1 = train[()]
  # train_set = make_train_set(train)
