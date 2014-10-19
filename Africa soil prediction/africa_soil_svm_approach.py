#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ismel
# @Date:   2014-10-19 13:13:22
# @Last Modified by:   maco
# @Last Modified time: 2014-10-19 13:22:44

"""
    The following code is intended to search the best parameters of SVR algorithm 
    for predicting physical and chemical properties of Africa soils.
    For further information visit: https://www.kaggle.com/c/afsis-soil-properties and
    https://www.kaggle.com/c/afsis-soil-properties/data
"""

print __doc__

import numpy as np
import pylab as pl
import pandas as pd
import argparse

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn import feature_extraction
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV

from sklearn.feature_selection import SelectKBest, f_regression

# defines the arguments parser
parser = argparse.ArgumentParser(description='Africa soil property predictions')

# includes depth of the soil sample on the prediction study
parser.add_argument('--depthSampleFeature', dest='depthSampleFeature', action='store_true', help="include depth of the soil sample on the prediction study")
parser.set_defaults(depthSampleFeature=False)

# adds argument to determine whether scale or not scale the data, that is the question
parser.add_argument('--scale', dest='scale', action='store_true', help="apply standard scale to the data")
parser.set_defaults(scale=False)

# adds argument for permitting to set, from the command line, the number of kbest features argument in the SelectKBest filter
parser.add_argument("-k", "--kBest", type=int, default = 0,
                    help="argument k for the SelectBestK filter")

args = parser.parse_args()

# imports training and test data from files
train = pd.read_csv('data/training.csv')
test = pd.read_csv('data/sorted_test.csv')

# chooses target variables for predictions
labels = train[['Ca','P','pH','SOC','Sand']].values

# removes id and target variables from training and test matrix 
train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)

# drops CO2 parameters as they suggested 
train.drop(['m2379.76','m2377.83','m2375.9','m2373.97','m2372.04','m2370.11','m2368.18','m2366.26','m2364.33','m2362.4','m2360.47','m2358.54','m2356.61','m2354.68','m2352.76'], axis=1, inplace=True)
test.drop(['m2379.76','m2377.83','m2375.9','m2373.97','m2372.04','m2370.11','m2368.18','m2366.26','m2364.33','m2362.4','m2360.47','m2358.54','m2356.61','m2354.68','m2352.76'], axis=1, inplace=True)

if args.depthSampleFeature:
    X, Y = np.array(train)[:,:3579], np.array(test)[:,:3579]
    
    # converts (2 categories: "Topsoil", "Subsoil") into 0-1 values
    X[:, 3578] = [1 if t == 'Topsoil' else 0 for t in X[:, 3578]]
    Y[:, 3578] = [1 if t == 'Topsoil' else 0 for t in Y[:, 3578]]
else:
    X, Y = np.array(train)[:,:3578], np.array(test)[:,:3578]

# scales the data for SVR training if args.scale is set
if args.scale:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Y = scaler.fit_transform(Y)

# sets the range of SVR parameters 
kernel_range = ['rbf', 'lineal']
#C_range = 10. ** np.arange(-2, 9)
#gamma_range = 10. ** np.arange(-5, -4)
C_range = 10. ** np.arange(3, 5)
gamma_range = [0, 1]
param_grid = dict(kernel=kernel_range, gamma=gamma_range, C=C_range)

# creates the grid and sets the number of cv to 5
grid = GridSearchCV(SVR(), param_grid=param_grid, cv=5, verbose = 2)

# defines and initialize the array of predictions
preds = np.zeros((Y.shape[0], 5))

# for every property to predict
for i in range(5):

	# Selects the best K variables for the prediction of that property
    selector = SelectKBest(f_regression, args.kBest if args.kBest > 0 else 'all')
    selector.fit(X, labels[:, i])

    print selector.get_support().shape

    # and filters the input data accordingly
    new_xtrain = X[:, selector.get_support()]
    new_xtest = Y[:, selector.get_support()]

    # trains the grid
    grid.fit(new_xtrain, labels[:,i])

    # print the best the predictor encountered
    print("The best predictor is: ", grid.best_estimator_)

    # uses then the predictor above to predict the test set
    preds[:,i] = grid.predict(new_xtest).astype(float)

# exports predictions to output file in order to be submitted
sample = pd.read_csv('results/sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]
sample.to_csv('results/africa_svr_gridsearch.csv', index = False)