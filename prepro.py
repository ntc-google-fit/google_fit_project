import pandas as pd
import pickle
import scipy
import itertools
import streamlit as st

### Loading data ###


def load_data(filename=None):
    filename_default = './data/dataset_halfSecondWindow.csv'
    if not filename:
        filename = filename_default

    df = pd.read_csv(f"./{filename}")
    return df


### Preprocessing ###
@st.cache(allow_output_mutation=True)
def preprocess(df):

    df.columns = df.columns.str.replace(
        'android.sensor.', '').str.replace('#', '_')

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
             'rv_gyro_prox_mean',
             'lin_speed_prox_mean',
             'rv_gyro__grv_mean',
             'rv_gyro_mean',
             'lin_speed_mean',

             'user',
             'target'
             ]]

    df.fillna(0, inplace=True)

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
        'acc_mean*gyro_mean', 'acc_mean*sound_mean', 'rv_gyro_prox_mean', 'lin_speed_prox_mean',
        'rv_gyro__grv_mean', 'rv_gyro_mean', 'lin_speed_mean']

    x_train = train_df[num_vars]
    x_test = test_df[num_vars]
    y_train = train_df.target
    y_test = test_df.target

    return x_train, x_test, y_train, y_test


### Model ###

def load_model():
    filename = 'skl_gbm.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


### Postprocessing ###


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
    output = ''
    for chunk in chunks:
        if (chunk[0] == 'Vehicle'):
            if chunk[2][0] > 0:
                output = output + 'You travelled by transport for {} hour, {} minutes and {} seconds\n\n'.format(
                    chunk[2][0], chunk[2][1], chunk[2][2])
            else:
                output = output + \
                    'You travelled by transport for {} minutes and {} seconds\n\n'.format(
                        chunk[2][1], chunk[2][2])
        elif (chunk[0] == 'Walking'):
            if chunk[2][0] > 0:
                output = output + 'You walked for {} hour, {} minutes and {} seconds\n\n'.format(
                    chunk[2][0], chunk[2][1], chunk[2][2])
            else:
                output = output + \
                    'You walked for {} minutes and {} seconds\n\n'.format(
                        chunk[2][1], chunk[2][2])
        elif (chunk[0] == 'Still'):
            if chunk[2][0] > 0:
                output = output + 'You rested for {} hour, {} minutes and {} seconds\n\n'.format(
                    chunk[2][0], chunk[2][1], chunk[2][2])
            else:
                output = output + \
                    'You rested for {} minutes and {} seconds\n\n'.format(
                        chunk[2][1], chunk[2][2])
    return output
