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
import prepro

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix


# page config
st.set_page_config(page_title="Ex-stream-ly Cool App",
                   layout="wide", initial_sidebar_state="expanded",)


sns.set_style("whitegrid")

# CSS

# st.markdown(
#    """
#   <style>
#     .main {
#    background-color: #ffffff;
#
#     }
#    </style>
#    ,
#    unsafe_allow_html=True
# )


# Load Data

# @st.cache(allow_output_mutation=True)

################ Get data  #####################################

df = prepro.load_data()
x_train, x_test, y_train, y_test = prepro.preprocess(df)


# Get model and predict

loaded_model = prepro.load_model()
pred = loaded_model.predict(x_test)


################ Postprocessing  ###############################

smooth_pred = prepro.smoothen(pred, 100)
accuracy = metrics.accuracy_score(y_test, smooth_pred)
chunks_output = prepro.chunks(smooth_pred)
output = prepro.print_chunks(chunks_output)


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


################################################################


# site title
st.title("Trackts App")  # site title h1
st.subheader(
    "Strive School - Google Fit Project")
st.markdown("""---""")

image = Image.open('imgs/logo_trackts.png')
st.sidebar.image(image, caption='')


def main():
    menu = ["Home", "Predictions", "Tests", "Loading"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":

        ###################################################
        header = st.beta_container()
        team = st.beta_container()
        github = st.beta_container()
        ###################################################
        with header:
            # st.title('Track App')

            st.subheader('Machine Learning / Feature Engineering Project')
            st.text(' ')
            image = Image.open('imgs/ai_2.jpg')
            st.image(image, caption='')
            st.text(" ")

            st.markdown("""---""")

            # display code
            st.echo()
            with st.echo():
                print('Trackts App')

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
            st.text(' ')
            st.text(' ')
            st.text(' ')
            st.text(' ')
            st.text(' ')
            st.text(' ')
            st.subheader('GitHub / Instructions')
            st.text(' ')
            st.markdown(
                'Check the instruction [here](https://ntc-google-fit.github.io/)')
            st.text(' ')


##########################################################################
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
    #         # st.set_option('deprecation.showPyplotGlobalUse', False)

    #         st.text('Data Correlation ')
    #         sns.set(style="white")
    #         plt.rcParams['figure.figsize'] = (15, 10)
    #         sns.heatmap(df.corr(), annot=True, linewidths=.5, cmap="Blues")
    #         plt.title('Correlation Between Variables', fontsize=30)
    #         plt.show()
    #         st.pyplot()

    elif choice == "Tests":

        st.subheader("Step Counter")
        if st.button('Amount of steps'):
            with st.spinner("Processing data..."):
                # Step Counter
                st.write(steps, "steps")

        st.subheader("Distance Covered")
        if st.button('Distance in m'):
            with st.spinner("Processing data..."):
                # distance counter:
                st.write(steps * 0.762, "meters")

        # st.subheader("Calories Burnt")
        # if st.button('Calories'):
        #     with st.spinner("Processing data..."):
        #         st.write("Your BMI")
                # st.button('Calories')

        # Uploading the data

        # st.markdown("""---""")
        st.markdown(" ")
        st.markdown(" ")

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
                st.write('Accuracy Score: ', round(accuracy, 2) * 100, '%')
                st.markdown(" ")
                st.write(output)

        ##########

        # st.markdown("""---""")

        # st.subheader("Upload your data")
        # st.write(" ")

        # st.file_uploader('File uploader')


main()
