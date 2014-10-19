#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ismel
# @Date:   2014-10-19 14:32:30
# @Last Modified by:   ismelbr@gmail.com
# @Last Modified time: 2014-10-19 19:37:15

"""
    The following code is intended to search the best parameters of SVR algorithm 
    for predicting bike sharing demand.
    For further information visit: https://www.kaggle.com/c/bike-sharing-demand and
    https://www.kaggle.com/c/bike-sharing-demand/data
"""

print __doc__

import datetime
import pandas as pd
import numpy as np
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn import svm, feature_extraction, ensemble, grid_search, preprocessing
from sklearn.metrics import precision_score

# defines the arguments parser
parser = argparse.ArgumentParser(description='Bike sharing demand prediction')

# includes 'year' as feature on the prediction study
parser.add_argument('--year', dest='year', action='store_true', help="extract the year from the timestamp of the date and include it on the prediction study")
parser.set_defaults(year=False)

# includes 'month' as feature on the prediction study
parser.add_argument('--month', dest='month', action='store_true', help="extract the month from the timestamp of the date and include it on the prediction study")
parser.set_defaults(month=False)

# includes 'day' as feature on the prediction study
parser.add_argument('--day', dest='day', action='store_true', help="extract the day from the timestamp of the date and include it on the prediction study")
parser.set_defaults(day=False)

# includes 'hour' as feature on the prediction study
parser.add_argument('--hour', dest='hour', action='store_true', help="extract the hour from the timestamp of the date and include it on the prediction study")
parser.set_defaults(month=False)

# adds argument to determine whether scale or not scale the data, that is the question
parser.add_argument('--scale', dest='scale', action='store_true', help="apply standard scale to the data")
parser.set_defaults(scale=False)

args = parser.parse_args()

# imports training and test data from files
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# chooses count feature for prediction
labels = train['count'].values

# removes 'casual', 'registered' and 'count' columns from train 
# this first approach discards prediting casual and registered features before predicting the target variable 'count'
train.drop(['count'], axis=1, inplace=True)
train.drop(['casual', 'registered'], axis=1, inplace=True)

# creates arrays for train and test
X, Y = np.array(train), np.array(test)

# gets the features names
features = np.array(train.columns)

# if asked, extracts 'year', 'month', 'day' and 'hour' data from 'datatime' and creates the new columns

# adds feature 'year'
if args.year:
	train_year = np.zeros((train.shape[0], 1))
	test_year = np.zeros((test.shape[0], 1))
	train_year[:, 0] = [(datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S").year) for t in train['datetime']]
	test_year[:, 0] = [(datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S").year) for t in test['datetime']]
	X = np.append(X, train_year, 1)
	Y = np.append(Y, test_year, 1)
	features = np.append(np.array(features), np.array("year"))

# adds feature 'month'
if args.month:
	train_month = np.zeros((train.shape[0], 1))
	test_month = np.zeros((test.shape[0], 1))
	train_month[:, 0] = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S").month for t in train['datetime']]
	test_month[:, 0] = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S").month for t in test['datetime']]
	X = np.append(X, train_month, 1)
	Y = np.append(Y, test_month, 1)
	features = np.append(np.array(features), np.array("month"))

# adds feature 'day'
if args.day:
	train_day = np.zeros((train.shape[0], 1))
	test_day = np.zeros((test.shape[0], 1))
	train_day[:, 0] = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S").day for t in train['datetime']]
	test_day[:, 0] = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S").day for t in test['datetime']]
	X = np.append(X, train_day, 1)
	Y = np.append(Y, test_day, 1)
	features = np.append(np.array(features), np.array("day"))

# adds feature 'hour'
if args.hour:
	train_hour = np.zeros((train.shape[0], 1))
	test_hour = np.zeros((test.shape[0], 1))
	train_hour[:, 0] = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S").hour for t in train['datetime']]
	test_hour[:, 0] = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S").hour for t in test['datetime']]
	X = np.append(X, train_hour, 1)
	Y = np.append(Y, test_hour, 1)
	features = np.append(np.array(features), np.array("hour"))

# scales the data for SVR training if args.scale is set
if args.scale:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = scaler.fit_transform(Y)

# sets the range of RFR parameters 
forest_params={
     'n_estimators': range(50, 400, 50),
     'max_features': ['auto','sqrt','log2'],
     'min_samples_split': range(2, 11),
     'min_samples_leaf': range(1, 11)
}

# exports predictions to output file in order to be submitted
sample = pd.read_csv('results/sampleSubmission.csv')
sample_array = np.array(sample)

for datetime in Y[:, 0]:
	# creates the grid and sets the number of cv to 5
	clf = grid_search.GridSearchCV(ensemble.RandomForestRegressor(), param_grid=forest_params, cv = 5, verbose=1)
 	filtered_train = X[X[:, 0] < datetime, 1:]
 	
 	filtered_labels = np.take(np.array(labels), range(filtered_train.shape[0]))
	clf.fit(filtered_train, filtered_labels)

	prediction_params = Y[Y[:, 0] == datetime, 1:]
	prediction = np.round(clf.predict(prediction_params).astype(float), 0)

	print str(datetime) + " " + str(prediction)

	sample_array[sample_array[:, 0] == datetime, sample_array.shape[1] - 1] = prediction

np.savetxt('results/africa_rfr_gridsearch.csv', sample_array, delimiter=",", fmt="%s", header='datetime,count', comments='')