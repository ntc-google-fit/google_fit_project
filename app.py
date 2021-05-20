import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
import os
import pickle

from sklearn import set_config
# accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import metrics
from sklearn import model_selection  # train_test_split
from sklearn import compose
from sklearn import impute
from sklearn.preprocessing import RobustScaler
from sklearn import preprocessing  # OrdinalEncoder, LabelEncoder
from sklearn import pipeline      # Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Necesary for HistGradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix


sns.set_style("whitegrid")

# CSS
st.markdown(
    """
    <style>
     .main {
    background-color: #ffffff;
    
     }
    </style>
    """,
    unsafe_allow_html=True
)


# header=st.beta_container()
# team=st.beta_container()
# dataset=st.beta_container()
# footer=st.beta_container()


# Load Data
@st.cache(allow_output_mutation=True)
def load_data(filename=None):
    filename_default = './data/dataset_halfSecondWindow.csv'
    if not filename:
        filename = filename_default

    df = pd.read_csv(f"./{filename}")
    return df


df = load_data()

###########
# pickle

# save the model to disk
# filename = './data/final_model_v2.sav'
# pickle.dump(model, open(filename, 'wb'))

# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(x_test, y_test)
# st.write(result)

############


# remove issues from plots
# st.set_option('deprecation.showPyplotGlobalUse', False)

# site title
st.title("Tracker App")
st.subheader(
    "Strive School - Google Fit Project")
st.text(" ")
st.text(" ")

st.write("NTC Team")
st.text(" ")


########

# input predictions for streamlit
# gender = st.selectbox("Select your gender: ", options=['Male', 'Female'])
# weight = st.slider("Enter your weight (kg): ", 1, 200, 1)
# height = st.text_input('Enter heigth (m):')
age = st.slider("Enter your age:", 1, 100, 1)
# heart_rate = st.slider("Enter your heart rate (bpm): ", 1, 200, 1)

st.write('Your age is: ', age)

########

# st.markdown("""---""")

# st.echo()
# with st.echo():
#     # merge car and bus
#     vehicle_dict = {'Car': 'Vehicle', 'Bus': 'Vehicle',
#                     'Train': 'Vehicle', 'Walking': 'Walking'}
#     dataset.replace({'target': vehicle_dict}, inplace=True)


#######

# Clean column names
df.columns = df.columns.str.replace(
    'android.sensor.', '').str.replace('#', '_')

# feature engineering
df['acc_mean*gyro_mean'] = df['accelerometer_mean'] * df['gyroscope_mean']
df['acc_mean*sound_mean'] = df['accelerometer_mean'] * df['sound_mean']

df['rv_gyro_mean'] = df['rotation_vector_mean'] * df['gyroscope_mean']
df['lin_speed_mean'] = df['linear_acceleration_mean'] * df['speed_mean']
df['rv_gyro_prox_mean'] = df['rotation_vector_mean'] * \
    df['gyroscope_mean'] * df['proximity_std']
df['lin_speed_prox_mean'] = df['linear_acceleration_mean'] * \
    df['speed_mean'] * df['proximity_std']
df['rv_gyro__grv_mean'] = df['rotation_vector_mean'] * \
    df['gyroscope_mean'] * df['game_rotation_vector_mean']

vehicle_dict = {'Car': 'Vehicle', 'Bus': 'Vehicle',
                'Train': 'Vehicle', 'Still': 'Still', 'Walking': 'Walking'}
df.replace({'target': vehicle_dict}, inplace=True)

########

df = df[['accelerometer_mean',
         'accelerometer_min',
         'accelerometer_max',
         'accelerometer_std',
         'linear_acceleration_mean', 'linear_acceleration_min', 'linear_acceleration_max', 'linear_acceleration_std',
         'orientation_mean',
         'orientation_min',
         'orientation_max',
         'orientation_std',
         'magnetic_field_mean',
         'magnetic_field_min',
         'magnetic_field_max',
         'magnetic_field_std',
         'gyroscope_mean',
         'gyroscope_min',
         'gyroscope_max',
         'gyroscope_std',
         'gravity_mean',
         'gravity_min',
         'gravity_max',
         'gravity_std',
         'acc_mean*gyro_mean',
         'acc_mean*sound_mean',
         'user',
         'target'
         ]]


df.fillna(0, inplace=True)

########

# train test split
big_users = ['U1', 'U3', 'U6', 'U7', 'U10', 'U12']

train_df = df[df.user.isin(big_users)]
test_df = df[~df.user.isin(big_users)]


train_df.drop('user', axis=1, inplace=True)
test_df.drop('user', axis=1, inplace=True)


num_vars = [
    'accelerometer_mean', 'accelerometer_min', 'accelerometer_max', 'accelerometer_std',
    'linear_acceleration_mean', 'linear_acceleration_min', 'linear_acceleration_max', 'linear_acceleration_std',
    'magnetic_field_mean', 'magnetic_field_min', 'magnetic_field_max', 'magnetic_field_std',
    'gyroscope_mean', 'gyroscope_min', 'gyroscope_max', 'gyroscope_std',
    'orientation_mean', 'orientation_min', 'orientation_max', 'orientation_std',
    'gravity_mean', 'gravity_min', 'gravity_max', 'gravity_std',
    'acc_mean*gyro_mean', 'acc_mean*sound_mean']


########

# split
x_train = train_df[num_vars]
x_test = test_df[num_vars]
y_train = train_df.target
y_test = test_df.target


########

# load the model from disk
filename = './data/final_model_v2.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = (loaded_model.score(x_test, y_test) * 100)
# st.write(result)


##########

# st.markdown("""---""")

# st.subheader("Upload your data")
# st.write(" ")

# st.file_uploader('File uploader')


#######

st.write(" ")
st.write(" ")
# pred_button = st.checkbox("Check prediction")
# if pred_button:
#     st.checkbox(pred, value=True)

if st.button('Check prediction'):
    st.write('result: %s' % result, '%')
    # st.write(result)
