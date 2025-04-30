#!/usr/bin/env python3

import pandas as pd
import sklearn

filename = "laptop-data.csv"
train_filename = "laptop-data-train.csv"
test_filename = "laptop-data-test.csv"
data = pd.read_csv(filename, encoding='latin1')
seed = 100
ratio = 0.2
data_train, data_test = \
    sklearn.model_selection.train_test_split(data, test_size=ratio, random_state=seed)

data_train.to_csv(train_filename)
data_test.to_csv(test_filename)
