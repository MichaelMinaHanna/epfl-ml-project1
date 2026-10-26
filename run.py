from helpers import load_csv_data 
import numpy as np
x_train, x_test, y_train, train_ids, test_ids = load_csv_data("dataset")
print(x_train.shape, x_test.shape)
print(np.unique(y_train, return_counts=True))
print(np.isnan(x_train).mean(axis=0)) 
