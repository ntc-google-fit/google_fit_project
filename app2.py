import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import scipy
import itertools

from sklearn import set_config
from sklearn import metrics
from sklearn import model_selection  # train_test_split
from sklearn import compose
from sklearn import impute
from sklearn.preprocessing import RobustScaler
from sklearn import preprocessing  # OrdinalEncoder, LabelEncoder
from sklearn import pipeline      # Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier

import prepro



### Get data

df = prepro.load_data()
x_train, x_test, y_train, y_test = prepro.preprocess(df)


### Get model and predict

#loaded_model = prepro.load_model()
#pred = loaded_model.predict(x_test)


### Postprocessing of pred

#smooth_pred = prepro.smoothen(pred, 100)
#accuracy = metrics.accuracy_score(y_test, smooth_pred)
#chunks_output = prepro.chunks(smooth_pred)
#output = prepro.print_chunks(chunks_output)



################ Counting steps ################################

def step_counter_on_walking(accelerometer_mean_list_for_walking):

	step_counter = 0
	for i in range(len(accelerometer_mean_list_for_walking)-1):
		if i > 0:
			y = accelerometer_mean_list_for_walking[i]
			y_before = accelerometer_mean_list_for_walking[i-1]
			y_after = accelerometer_mean_list_for_walking[i+1]
			if (y > y_before) & (y > y_after) & (10 < (y_before+y_after)):
				step_counter += 1
	return step_counter

step_df = x_test.copy()
step_df['target'] = y_test
step_df = step_df[step_df.target == 'Walking']
step_data = step_df['accelerometer_mean'].values
steps = step_counter_on_walking(step_data)

print(steps)


