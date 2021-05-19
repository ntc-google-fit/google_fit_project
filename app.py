import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
import os
# import pickle

from sklearn import set_config
# accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import metrics
from sklearn import model_selection  # train_test_split
from sklearn import compose
from sklearn import impute
from sklearn.preprocessing import RobustScaler
from sklearn import preprocessing  # OrdinalEncoder, LabelEncoder
from sklearn import pipeline      # Pipeline


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


# Load Data
@st.cache(allow_output_mutation=True)
def load_data(filename=None):
    filename_default = './data/archive/dataset_5secondWindow%5B1%5D.csv'
    if not filename:
        filename = filename_default

    df = pd.read_csv(f"./{filename}")
    return df


dataset = load_data()

# load_clf = pickle.load(open('./data/model.pkl', "rb"))

# remove issues from plots
st.set_option('deprecation.showPyplotGlobalUse', False)

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
gender = st.selectbox("Select your gender: ", options=['Male', 'Female'])
weight = st.slider("Enter your weight (kg): ", 1, 200, 1)
height = st.text_input('Enter heigth (m):')
age = st.slider("Enter your age:", 1, 100, 1)
heart_rate = st.slider("Enter your heart rate (bpm): ", 1, 200, 1)
time = st.slider('Enter time (min): ', 1, 300, 1)


########

# drop columns

to_drop = [i for i in dataset.columns if 'std' in i]
dataset.drop(to_drop, axis=1, inplace=True)
to_drop = [i for i in dataset.columns if 'accelerometer_max' in i]
dataset.drop(to_drop, axis=1, inplace=True)
# to_drop = [i for i in dataset.columns if 'gyroscope_max' in i]
# dataset.drop(to_drop, axis=1, inplace=True)
# to_drop = [i for i in dataset.columns if 'gyroscope_min' in i]
dataset.drop(to_drop, axis=1, inplace=True)
to_drop = [i for i in dataset.columns if 'sound_min' in i]
dataset.drop(to_drop, axis=1, inplace=True)
to_drop = [i for i in dataset.columns if 'sound_max' in i]
dataset.drop(to_drop, axis=1, inplace=True)

########

# clean
dataset.columns = dataset.columns.str.replace(
    'android.sensor.', '').str.replace('#', '_')


# print head of the dataset
# st.write("Dataset")
# st.table(dataset.head(5))


########

# st.markdown("""---""")

# checking distribution by different sensors (time, accelerometer, gyroscope, sound)

# st.subheader("Plotting")
# st.write(" ")

# plt.figure(figsize=(10, 15))

# plt.subplot(4, 1, 1)
# sns.distplot(dataset.iloc[:, 0])
# plt.xlabel('Time')

# plt.subplot(4, 1, 2)
# for i in range(1, 4):
#     sns.distplot(dataset.iloc[:, i])
# plt.legend(dataset.iloc[:, 1:4].columns)
# plt.xlabel('Accelerometer')

# plt.subplot(4, 1, 3)
# for i in range(4, 7):
#     sns.distplot(dataset.iloc[:, i])
# plt.legend(dataset.iloc[:, 4:7].columns)
# plt.xlabel('Gyroscope')

# plt.subplot(4, 1, 4)
# for i in range(7, 10):
#     sns.distplot(dataset.iloc[:, i])
# plt.legend(dataset.iloc[:, 7:10].columns)
# plt.xlabel('Sound')

# st.pyplot()


########

# st.subheader("Plotting")
# st.write(" ")

# sns.set(style="ticks")
# sns.pairplot(data=dataset.loc[:, :"target"], hue="target")
# st.pyplot()


########

st.markdown("""---""")

st.subheader("Correlation Matrix")
st.write(" ")

# correlation matrix
fig = px.imshow(dataset.corr())
st.plotly_chart(fig)


##########

st.markdown("""---""")

st.subheader("Upload your data")
st.write(" ")

# st.file_uploader('File uploader')


def file_selector(folder_path='./data'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


filename_new = file_selector()
st.write('You selected `%s`' % filename_new)

# new dataset
df_new = pd.read_csv(f"./{filename_new}")

# drop from new dataset
# to_drop = [i for i in dataset.columns if 'std' in i]
# dataset.drop(to_drop, axis=1, inplace=True)
# to_drop = [i for i in dataset.columns if 'gyroscope_min' in i]
# dataset.drop(to_drop, axis=1, inplace=True)
# to_drop = [i for i in dataset.columns if 'gyroscope_max' in i]
# dataset.drop(to_drop, axis=1, inplace=True)
# to_drop = [i for i in dataset.columns if 'accelerometer_min' in i]
# dataset.drop(to_drop, axis=1, inplace=True)
# to_drop = [i for i in dataset.columns if 'accelerometer_max' in i]
# dataset.drop(to_drop, axis=1, inplace=True)

# print head of the new dataset
st.write("Dataset")
st.table(df_new.head(5))

#######

# st.write(" ")
# st.write(" ")

# st.button('Hit me')
