import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import pickle
import scipy
import itertools

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
#from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix


# set_config(display='diagram')
# page config
# set_config(display='diagram', layout="wide")
st.set_page_config(page_title="Ex-stream-ly Cool App",
                   layout="wide", initial_sidebar_state="expanded",)


sns.set_style("whitegrid")

# CSS

#st.markdown(
#    """
#   <style>
#     .main {
#    background-color: #ffffff;
#    
#     }
#    </style>
#    ,
#    unsafe_allow_html=True
#)


# Load Data

@st.cache(allow_output_mutation=True)

################ Get data  #####################################

df = prepro.load_data()
x_train, x_test, y_train, y_test = prepro.preprocess(df)


### Get model and predict

loaded_model = prepro.load_model()
pred = loaded_model.predict(x_test)


################ Postprocessing  ###############################

smooth_pred = prepro.smoothen(pred, 100)
accuracy = metrics.accuracy_score(y_test, smooth_pred)
chunks_output = prepro.chunks(smooth_pred)
output = prepro.print_chunks(chunks_output)

print(output)

###############################################################






# remove issues from plots
# st.set_option('deprecation.showPyplotGlobalUse', False)

# site title
st.title("Tracker App")  # site title h1
st.subheader(
    "Strive School - Google Fit Project")
st.text(" ")
st.text(" ")

image = Image.open('imgs/ntc.jpeg')
st.sidebar.image(image, caption='')


########

# input predictions for streamlit
# gender = st.sidebar.selectbox(
#     "Select your gender: ", options=['Male', 'Female'])
# weight = st.sidebar.slider("Enter your weight (kg): ", 1, 200, 1)
# height = st.sidebar.text_input('Enter heigth (m):')
# age = st.sidebar.slider("Enter your age:", 1, 100, 1)
# heart_rate = st.sidebar.slider("Enter your heart rate (bpm): ", 1, 200, 1)

# gender = ()
# if height != 0:
#     st.write('Height: ', height)
# st.write('Your age is: ', age)
# st.write('Your age is: ', age)


########

# st.markdown("""---""")

# disply code

# st.echo()
# with st.echo():
#     # merge car and bus
#     vehicle_dict = {'Car': 'Vehicle', 'Bus': 'Vehicle',
#                     'Train': 'Vehicle', 'Walking': 'Walking'}
#     dataset.replace({'target': vehicle_dict}, inplace=True)


#######

# model final_final
# Clean column names
# df.columns = df.columns.str.replace(
#     'android.sensor.', '').str.replace('#', '_')

# # feature engineering
# df['acc_mean*gyro_mean'] = df['accelerometer_mean'] * df['gyroscope_mean']
# df['acc_mean*sound_mean'] = df['accelerometer_mean'] * df['sound_mean']

# df['rv_gyro_mean'] = df['rotation_vector_mean'] * df['gyroscope_mean']
# df['lin_speed_mean'] = df['linear_acceleration_mean'] * df['speed_mean']
# df['rv_gyro_prox_mean'] = df['rotation_vector_mean'] * \
#     df['gyroscope_mean'] * df['proximity_std']
# df['lin_speed_prox_mean'] = df['linear_acceleration_mean'] * \
#     df['speed_mean'] * df['proximity_std']
# df['rv_gyro__grv_mean'] = df['rotation_vector_mean'] * \
#     df['gyroscope_mean'] * df['game_rotation_vector_mean']

# vehicle_dict = {'Car': 'Vehicle', 'Bus': 'Vehicle',
#                 'Train': 'Vehicle', 'Still': 'Still', 'Walking': 'Walking'}
# df.replace({'target': vehicle_dict}, inplace=True)

# model v2
# Clean column names
<<<<<<< HEAD
=======
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

# model final_final
# df = df[['accelerometer_mean',
#          'accelerometer_min',
#          'accelerometer_max',
#          'accelerometer_std',
#          'linear_acceleration_mean', 'linear_acceleration_min', 'linear_acceleration_max', 'linear_acceleration_std',
#          'orientation_mean',
#          'orientation_min',
#          'orientation_max',
#          'orientation_std',
#          'magnetic_field_mean',
#          'magnetic_field_min',
#          'magnetic_field_max',
#          'magnetic_field_std',
#          'gyroscope_mean',
#          'gyroscope_min',
#          'gyroscope_max',
#          'gyroscope_std',
#          'gravity_mean',
#          'gravity_min',
#          'gravity_max',
#          'gravity_std',
#          'acc_mean*gyro_mean',
#          'acc_mean*sound_mean',
#          'rv_gyro_prox_mean',
#          'lin_speed_prox_mean',
#          'rv_gyro__grv_mean',
#          'rv_gyro_mean',
#          'lin_speed_mean',

#          'user',
#          'target'
#          ]]

# model V2
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


# model final_final
# num_vars = [
#     'accelerometer_mean', 'accelerometer_min', 'accelerometer_max', 'accelerometer_std',
#     'linear_acceleration_mean', 'linear_acceleration_min', 'linear_acceleration_max', 'linear_acceleration_std',
#     'magnetic_field_mean', 'magnetic_field_min', 'magnetic_field_max', 'magnetic_field_std',
#     'gyroscope_mean', 'gyroscope_min', 'gyroscope_max', 'gyroscope_std',
#     'orientation_mean', 'orientation_min', 'orientation_max', 'orientation_std',
#     'gravity_mean', 'gravity_min', 'gravity_max', 'gravity_std',
#     'acc_mean*gyro_mean', 'acc_mean*sound_mean', 'rv_gyro_prox_mean', 'lin_speed_prox_mean',
#     'rv_gyro__grv_mean', 'rv_gyro_mean', 'lin_speed_mean']

# model v2
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

def smoothen(preds, window):
    def most_freq_val(x): return [scipy.stats.mode(x)[0][0]] * len(x)
    smoothed = [most_freq_val(preds[i:i + window])
                for i in range(0, len(preds), window)]
    result = list(itertools.chain.from_iterable(smoothed))
    result = result[0:-100]
    result.extend(most_freq_val(result[-100:]))

    return result


def chunks(array):
    periods = []
    cntr = 1

    for i in range(0, len(array)-2):
        if (array[i] == array[i+1]):
            cntr += 1
        else:
            periods.append((array[i], cntr))
            cntr = 1

    periods.append((array[-1], cntr+1))

    final = [(item[0], item[1], into_min(item[1])) for item in periods]

    return final


def into_min(secs):
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return h, m, s


def print_chunks(chunks):
    for chunk in chunks:
        if (chunk[0] == 'Vehicle'):
            if chunk[2][0] > 0:
                print('You travelled by transport for {} hour, {} minutes and {} seconds'.format(
                    chunk[2][0], chunk[2][1], chunk[2][2]))
            else:
                print('You travelled by transport for {} minutes and {} seconds'.format(
                    chunk[2][1], chunk[2][2]))
        elif (chunk[0] == 'Walking'):
            if chunk[2][0] > 0:
                print('You walked for {} hour, {} minutes and {} seconds'.format(
                    chunk[2][0], chunk[2][1], chunk[2][2]))
            else:
                print('You walked for {} minutes and {} seconds'.format(
                    chunk[2][1], chunk[2][2]))
        elif (chunk[0] == 'Still'):
            if chunk[2][0] > 0:
                print('You rested for {} hour, {} minutes and {} seconds'.format(
                    chunk[2][0], chunk[2][1], chunk[2][2]))
            else:
                print('You rested for {} minutes and {} seconds'.format(
                    chunk[2][1], chunk[2][2]))

########


# load the model from disk
# filename = './data/final_final_final.sav'
filename = './data/final_model_v2.sav'
loaded_model = pickle.load(open(filename, 'rb'))
pred = loaded_model.predict(x_test)
smoothed_pred = smoothen(pred, 100)
accuracy = metrics.accuracy_score(y_test, smoothed_pred)
chunks_output = chunks(smoothed_pred)

#result = loaded_model.score(x_test, y_test)
# st.write(result)
>>>>>>> ce239c008a7f4a9d3c1cfcb36baafe9bd67cc126


#######

st.write(" ")
st.write(" ")
# pred_button = st.checkbox("Check prediction")
# if pred_button:
#     st.checkbox(pred, value=True)

# st.write('result: %s' % result, '%')

# ####################################################
# header = st.beta_container()
# team = st.beta_container()
# activities = st.beta_container()
# github = st.beta_container()
# footer = st.beta_container()
# ####################################################


def main():
    menu = ["Home", "Data Analysis", "Predictions", "Tests"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        # st.subheader("Home")
        # to_do1 = st.checkbox("Web Scrapping ")
        # to_do2 = st.checkbox("Data Analysis")
        # to_do3 = st.checkbox("Data Prosessing")
        # to_do4 = st.checkbox("Data Visualization")
        # to_do5 = st.checkbox("About Dumblodore Team")

        ###################################################
        header = st.beta_container()
        team = st.beta_container()
        github = st.beta_container()
        ###################################################
        with header:
            # st.title('Track App')
            st.markdown("""---""")
            st.subheader('Machine Learning Project')
            st.text(' ')
            image = Image.open('imgs/ai_2.jpg')
            st.image(image, caption='')

            st.text("NTC Team")
            st.text(" ")

            st.markdown("""---""")

            # display code
            st.echo()
            with st.echo():
                print('Tracking App')

            with team:
                # meet the team button
                st.sidebar.subheader('NTC Team')

                st.sidebar.markdown(
                    '[Fabio Fistarol](https://github.com/fistadev)')
                st.sidebar.markdown(
                    '[Deniz Elci](https://github.com/deniz-shelby)')
                st.sidebar.markdown(
                    '[Farrukh Bulbulov](https://github.com/fbulbulov)')
                st.sidebar.markdown(
                    '[Vladimir Gasanov](https://github.com/VladimirGas)')

                st.sidebar.text(' ')
                st.sidebar.text(' ')

        with github:
            # github section:
            st.subheader('GitHub / Instructions')
            st.markdown(
                'Check the instruction [here](https://ntc-google-fit.github.io/)')
            st.text(' ')


##########################################################################
    elif choice == "Data Analysis":
        dataset = st.beta_container()

        with dataset:
            st.title("Data Analysis")

            #### Data Correlation ####
            st.set_option('deprecation.showPyplotGlobalUse', False)

            st.text('Data Correlation ')
            sns.set(style="white")
            plt.rcParams['figure.figsize'] = (15, 10)
            sns.heatmap(df.corr(), annot=True, linewidths=.5, cmap="Blues")
            plt.title('Correlation Between Variables', fontsize=30)
            plt.show()
            st.pyplot()

    elif choice == "Tests":

        # the input is the column for anroid.sensor.accelerometer#mean and if the target is walking
        # as output i need this array for the step counter
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

        st.subheader("Step Counter")
        if st.button('Amount of steps'):
            with st.spinner("Processing data..."):
                # st.balloons()
                df1 = df['accelerometer_mean']  # but onyl for target walking
                steps = step_counter_on_walking(df1)
                st.write(steps, "steps")

        st.subheader("Distance Covered")
        if st.button('Distance in m'):
            with st.spinner("Processing data..."):
                # distance counter:
                # def steps_to_meters(step_count):
                #     meters = step_count * 0.762
                #     return meters
                # meters = steps_to_meters(steps)
                # st.balloons()
                df1 = df['accelerometer_mean']  # but onyl for target walking
                steps = step_counter_on_walking(df1)
                st.write(steps * 0.762, "meters")

        st.subheader("Calories Burnt")
        if st.button('Calories'):
            with st.spinner("Processing data..."):
                st.write("Your BMI")
                # st.button('Calories')

        st.markdown("""---""")
        st.markdown(" ")
        st.markdown(" ")

        st.subheader("Upload your data")
        st.write(" ")

        st.file_uploader('File uploader')

    elif choice == "ML":
        footer = st.beta_container()

        with footer:
            # Footer
            st.markdown("""---""")
            st.markdown("Tracking App - Machine Learning Project")
            st.markdown("")
            st.markdown(
                "If you have any questions, checkout our [documentation](https://ntc-google-fit.github.io/)")
            st.text(' ')

        ############################################################################################################################
    else:
        st.subheader("Predictions")

        def xgb_page_builder(data):
            st.sidebar.header('Track')
            st.sidebar.markdown('You can tune the parameters by siding')
            st.sidebar.text_input("What's your age?")
            cp = st.sidebar.slider(
                'Select max_depth (default = 30)', 0, 1, 2)
            thalach = st.sidebar.slider(
                'Select learning rate (divided by 10) (default = 0.1)', min_value=50, max_value=300, value=None, step=5)
            slope = st.sidebar.slider(
                'Select min_child_weight (default = 0.3)', 1, 2, 3)

        if st.button('Check prediction'):
            with st.spinner("Processing data..."):
                # st.balloons()
                st.write('result: %s' % accuracy)
                st.write(round(accuracy, 2) * 100, '%')
                st.write(output)

        ##########

        # st.markdown("""---""")

        # st.subheader("Upload your data")
        # st.write(" ")

        # st.file_uploader('File uploader')


main()
