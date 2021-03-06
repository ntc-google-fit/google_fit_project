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


# page config
st.set_page_config(page_title="Ex-stream-ly Cool App",
                   layout="wide", initial_sidebar_state="expanded",)


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


# pickle v2
# import pickle
# pickle_out = open("classifier.pkl", mode = "wb")
# pickle.dump(model, pickle_out)
# pickle_out.close()

############


# remove issues from plots
# st.set_option('deprecation.showPyplotGlobalUse', False)

# site title
st.title("Trackts App v1.2")  # site title h1
st.subheader(
    "Strive School - Google Fit Project")
st.markdown("""---""")

image = Image.open('imgs/logo_trackts.png')
st.sidebar.image(image, caption='')


########

# input predictions for streamlit


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
result = loaded_model.score(x_test, y_test)
# st.write(result)


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
    menu = ["Home", "Predictions", "Calculator", 'Loading']
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        # st.subheader("Home")
        # to_do1 = st.checkbox("Web Scrapping ")
        # to_do2 = st.checkbox("Data Analysis")
        # to_do3 = st.checkbox("Data Prosessing")
        # to_do4 = st.checkbox("Data Visualization")
        # to_do5 = st.checkbox("About Dumblodore Team")
        # image = Image.open('imgs/dumbledore-on-strive.jpeg')
        # st.image(image, caption='Dumbledore')

        ###################################################
        header = st.beta_container()
        team = st.beta_container()
        github = st.beta_container()
        ###################################################
        with header:
            # st.title('Track App')

            st.subheader('Machine Learning Project')
            st.text(' ')
            image = Image.open('imgs/ai_2.jpg')
            st.image(image, caption='')

            # st.text("NTC Team")
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
    # Loading page
    elif choice == "Loading":
        st.subheader("Upload your data")
        st.write(" ")

        st.subheader("Dataset")

        data_file = st.file_uploader("Upload CSV", type=["csv"])
        if data_file is not None:
            st.write(type(data_file))
            df2 = pd.read_csv(data_file)
            st.dataframe(df2)
    # elif choice == "Data Analysis":
    #     dataset = st.beta_container()

    #     with dataset:
    #         st.title("Data Analysis")

    #         #### Data Correlation ####
    #         st.set_option('deprecation.showPyplotGlobalUse', False)

    #         st.text('Data Correlation ')
    #         sns.set(style="white")
    #         plt.rcParams['figure.figsize'] = (15, 10)
    #         sns.heatmap(df.corr(), annot=True, linewidths=.5, cmap="Blues")
    #         plt.title('Correlation Between Variables', fontsize=30)
    #         plt.show()
    #         st.pyplot()

            #### Box Plot #####
            # st.text('Outlier Detection ')
            # fig = plt.figure(figsize=(15, 10))
            # sns.boxplot(data=df)
            # st.pyplot(fig)
            # st.text(' ')

    elif choice == "Calculator":

        # Uploading the data

        st.markdown(" ")
        st.markdown(" ")

        # st.subheader("Upload your data")
        # st.write(" ")

        # st.subheader("Dataset")

        # data_file = st.file_uploader("Upload CSV", type=["csv"])
        # if data_file is not None:
        #     st.write(type(data_file))
        #     df2 = pd.read_csv(data_file)
        #     st.dataframe(df2)

        #st.file_uploader("Upload CSV",type=["csv"])
        #st.write(' ')

        st.subheader("Your Biometrics")
        # input predictions for streamlit
        gender = st.selectbox("Select your gender: ",
                              options=['Male', 'Female'])
        weight = st.slider("Enter your weight (kg): ", 1, 200, 1)
        height = st.slider('Enter heigth (m):', 0.0, 2.3, 0.1)
        age = st.slider("Enter your age:", 1, 100, 1)
        heart_rate = st.slider("Enter your heart rate (bpm): ", 1, 200, 1)
        w_time = st.slider('Activity time (mins): ', 1, 300, 1)

        # the input is the column for anroid.sensor.accelerometer#mean and if the target is walking
        # as output i need this array for the step counter
        # step counter:

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

        # st.subheader("Step Counter")
        # if st.button('Amount of steps'):
        #     with st.spinner("Processing data..."):
        #         # st.balloons()
        #         df1 = df['accelerometer_mean']  # but onyl for target walking
        #         steps = step_counter_on_walking(df1)
            # st.write(steps, "steps")

        # distance counter:

        # st.subheader("Distance Covered")
        # if st.button('Distance in m'):
        #     with st.spinner("Processing data..."):
        #         # st.balloons()
        #         df1 = df['accelerometer_mean']  # but onyl for target walking
        #         steps = step_counter_on_walking(df1)
        #         dist = steps * 0.762
            # st.write(dist, "m")

        # Calories Burnt
        st.subheader("Calories Burnt:")
        calories_m = ((int(age) * 0.2017) + (int(weight) * 0.09036 * 2.20462) +
                      (int(heart_rate) * 0.6309) - 55.0969) * int(w_time) / 4.184
        calories_f = ((int(age) * 0.074) + (int(weight) * 0.05741 * 2.20462) +
                      (int(heart_rate) * 0.4472) - 20.4022) * int(w_time) / 4.184

        if gender == 'Male':
            result1 = ("Calories burnt: " + str(calories_m))
        if gender == 'Female':
            result1 = ("Calories burnt: " + str(calories_f))
        if st.button("Calories"):
            #result1=("Calories burnt: "+ str(calories))
            st.text(result1)

        # BMI
        st.subheader("BMI and Recommendations")
        bmi = weight/(height*height)
        if bmi == 0.11:
            st.text("      ")
        if st.button("BMI"):
            result = ("Your BMI is " + str(bmi))
            st.text(result)

        if bmi < 14:
            st.warning("You are soo thin!")
            if st.checkbox("Click here to know important tips to gain weight."):
                st.info("1) Eat more calories than your body burns.")
                st.info("2) Increase your fibre intake.")
                st.info(
                    "3) Eat Energy-Dense Foods and Use Sauces, Spices and Condiments.")
                st.info("4) Eat plenty of proteins.")

        else:
            if bmi > 35:
                st.warning("You are soo obese!")
                if st.checkbox("Click here to know your eating habits."):
                    st.info("1) Include fruits and veggies on regular basis.")
                    st.info(
                        "2) Eat a whole source of protein with each meal ??? meat, chicken, fish, eggs, etc.")
                    st.info(
                        "3) Make weight gainer shakes by mixing oats, milk, banana, peanut butter and whey protein in your blender.")
                    st.info(
                        "4) Do free weight, compounds like Squats and Deadlifts instead. They trigger more strength and muscle gains to gain weight.")

            else:
                st.success("You have an acceptable BMI")
                if st.checkbox("Click here to know how to maintain a good BMI."):
                    st.info(
                        "1) Try to make physical activity a regular part of your day, just like brushing your teeth.")
                    st.info("2) Stay hydrated and eat balanced diet.")
                    st.info("3) Avoid random snacking.")

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

        image = Image.open('imgs/calculator.jpg')
        st.image(image, caption='')

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
                st.write('result: %s' % result)
                st.write(round(result, 2) * 100, '%')

            st.markdown("20 rows sample:")
            st.dataframe(df.head(20))

        ##########

        # st.markdown("""---""")

        # st.subheader("Upload your data")
        # st.write(" ")

        # st.file_uploader('File uploader')


main()
