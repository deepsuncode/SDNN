  # =========================================================================
#   (c) Copyright 2022
#   All rights reserved
#   Programs written by Haodi Jiang
#   Department of Computer Science
#   New Jersey Institute of Technology
#   University Heights, Newark, NJ 07102, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================

'''
Run the this code to produce five inverted parameters
(three magnetic field parameters and two velocity field parameters)
of input NIRIS Stokes data.

Currently, the code work well if the scanned spectral points is less 60,
please let me know if the NIRIS Stokes profiles with spectral points greater than 60.ast

The code produces Bx, By, Bz, Doppler Width and LOS velocity by default.
If you need b_total, inclination and azimuth as well, please set save_mag_field_o = True in the code.
'''

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from astropy.io import fits
import math
import os
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import time
from collections import deque
import matplotlib.pyplot as plt
import matplotlib


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print('turn off loggins is not supported')


def read_data(data):
    print('Loading test data...')
    pad_start = int((60 - data.shape[1]) / 2)
    pad_end = 60 - data.shape[1] - pad_start
    data = np.pad(data, ((0, 0), (pad_start, pad_end), (0, 0), (0, 0)), 'constant')

    i_s = data[0, :, :, :]
    q = data[1, :, :, :]
    u = data[2, :, :, :]
    v = data[3, :, :, :]

    spectrum_height = q.shape[1]
    spectrum_width = q.shape[2]
    spectrum_length = q.shape[0]
    lists = deque()
    for i in range(spectrum_height):
        for j in range(spectrum_width):
            i_spectrum = list()
            q_spectrum = list()
            u_spectrum = list()
            v_spectrum = list()
            for k in range(spectrum_length):
                i_spectrum.append(int(i_s[k][i][j])/1000)
                q_spectrum.append(int(q[k][i][j])/1000)
                u_spectrum.append(int(u[k][i][j])/1000)
                v_spectrum.append(int(v[k][i][j])/1000)

            record = i_spectrum + q_spectrum + u_spectrum + v_spectrum
            lists.append(record)
    results = np.array(lists)
    X = np.reshape(results, (results.shape[0], 60, 4))
    print('Done loading...')
    return X, spectrum_height, spectrum_width


def SDNN():
    '''with larger dense layer size, 240 '''
    num_channels = 4
    input = Input(shape=(60, num_channels))
    branch_outputs = []
    for i in range(num_channels):
        # Slicing the ith channel:
        out = Lambda(lambda x: x[:, :, i])(input)
        out = Lambda(lambda x: K.expand_dims(out, -1))(out)
        conv0_1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(out)
        conv0_2 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(conv0_1)
        conv0 = MaxPool1D(pool_size=2)(conv0_2)
        conv1_1 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv0)
        conv1_2 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv1_1)
        conv1 = MaxPool1D(pool_size=2)(conv1_2)
        conv2_1 = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(conv1)
        conv2_2 = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(conv2_1)
        conv2 = MaxPool1D(pool_size=2)(conv2_2)
        out = Flatten()(conv2)
        out = Dense(240)(out)
        branch_outputs.append(out)

    # Concatenating together the per-channel results:
    out = Concatenate(axis=-1)(branch_outputs)
    out = Reshape((240, num_channels))(out)  # 1792
    conv0_1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(out)
    # conv0_2 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(conv0_1)
    conv0 = MaxPool1D(pool_size=2)(conv0_1)
    conv1_1 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv0)
    # conv1_2 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv1_1)
    conv1 = MaxPool1D(pool_size=2)(conv1_1)
    conv2_1 = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(conv1)
    # conv2_2 = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(conv2_1)
    conv2 = MaxPool1D(pool_size=2)(conv2_1)
    conv3_1 = Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')(conv2)
    # conv3_2 = Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')(conv3_1)
    # conv3 = MaxPool1D(pool_size=2)(conv3_1)
    # # conv4_1 = Conv1D(filters=1024, kernel_size=3, padding='same', activation='relu')(conv3)
    # # conv4_2 = Conv1D(filters=1024, kernel_size=3, padding='same', activation='relu')(conv4_1)
    # # conv4 = MaxPool1D(pool_size=2)(conv4_2)
    out = Flatten()(conv3_1)
    layer1 = Dense(2048, activation='relu')(out)
    # layer1 = Dropout(0.25)(layer1)
    layer2 = Dense(1024, activation='relu')(layer1)
    # layer2 = Dropout(0.25)(layer2)
    output = Dense(5, activation='linear')(layer2)
    # output = Dense(2, activation='linear')(layer2)  # activation: linear is the best
    model = Model(input, output)
    # model.summary()
    return model


def load_model():
    print('Loading SDNN model...')
    model_file = 'pretrained_model.h5'
    model = SDNN()
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
    model.load_weights(model_file)
    print('Done Loading...')
    return model


def inverse(test_data, model):
    # print('Loading testing data')
    start = time.time()
    X_test = test_data
    print('Start inversion...')
    y_predict = model.predict(X_test)
    end = time.time()
    print('End inversion...')
    # print('Time:', np.round(end - start, 1), 's')
    return y_predict


def save_results(predict_results, output_path, time_stamp, height, width, save_mag_field_o):
    print('Producing inversion results..')
    df_value = predict_results
    df_value[:, [0]] = df_value[:, [0]] * 5000.
    df_value[:, [1, 2]] = df_value[:, [1, 2]] * math.pi

    predicted_bx = np.multiply(np.multiply(df_value[:, 0], np.sin(df_value[:, 1])), np.cos(df_value[:, 2]))
    predicted_by = np.multiply(np.multiply(df_value[:, 0], np.sin(df_value[:, 1])), np.sin(df_value[:, 2]))
    predicted_bz = np.multiply(df_value[:, 0], np.cos(df_value[:, 1]))

    # height = 720
    # width = 720
    predicted_bx_img = predicted_bx.reshape(height, width)
    predicted_by_img = predicted_by.reshape(height, width)
    predicted_bz_img = predicted_bz.reshape(height, width)

    predicted_bx = np.flipud(predicted_bx_img)
    predicted_by = np.flipud(predicted_by_img)
    predicted_bz = np.flipud(predicted_bz_img)

    # Save data to fits
    bx_fits_file = os.path.join(output_path,'bx_{}.fits'.format(time_stamp))
    try:
        os.remove(bx_fits_file)
    except OSError:
        pass
    bx_fits = fits.PrimaryHDU(predicted_bx)
    bx_fits.writeto(bx_fits_file)

    by_fits_file = os.path.join(output_path,'by_{}.fits'.format(time_stamp))
    try:
        os.remove(by_fits_file)
    except OSError:
        pass
    by_fits = fits.PrimaryHDU(predicted_by)
    by_fits.writeto(by_fits_file)

    bz_fits_file = os.path.join(output_path, 'bz_{}.fits'.format(time_stamp))
    try:
        os.remove(bz_fits_file)
    except OSError:
        pass
    bz_fits = fits.PrimaryHDU(predicted_bz)
    bz_fits.writeto(bz_fits_file)

    # save LOS velocity and Doppler width
    df_value[:, [3]] = df_value[:, [3]]
    df_value[:, [4]] = df_value[:, [4]] - 0.5
    Doppler_w = df_value[:, 3].reshape(height, width)
    LOS_V = df_value[:, 4].reshape(height, width)
    LOS_V = LOS_V * 10 ** -10 / (1.56 * 10 ** -6) * (3 * 10 ** 8) / 1000  # convert unit A to km/s

    Doppler_w = np.flipud(Doppler_w)
    LOS_V = np.flipud(LOS_V)

    Doppler_w_fits_file = os.path.join(output_path, 'Doppler_Width_{}.fits'.format(time_stamp))
    try:
        os.remove(Doppler_w_fits_file)
    except OSError:
        pass
    Doppler_w_fits = fits.PrimaryHDU(Doppler_w)
    Doppler_w_fits.writeto(Doppler_w_fits_file)

    LOS_V_fits_file = os.path.join(output_path, 'LOS_Velocity_{}.fits'.format(time_stamp))
    try:
        os.remove(LOS_V_fits_file)
    except OSError:
        pass
    LOS_V_fits = fits.PrimaryHDU(LOS_V)
    LOS_V_fits.writeto(LOS_V_fits_file)

    # save B_total, inclination and azimnuth
    if save_mag_field_o == True:
        # df_value[:, [1, 2]] = df_value[:, [1, 2]] / math.pi * 180.
        predicted_btotal_img = df_value[:, 0].reshape(height, width)
        predicted_btotal_img = np.flipud(predicted_btotal_img)
        # print(predicted_btotal_img[0])
        predicted_inclination_img = df_value[:, 1].reshape(height, width)
        predicted_inclination_img = np.flipud(predicted_inclination_img)
        predicted_azimuth_img = df_value[:, 2].reshape(height, width)
        predicted_azimuth_img = np.flipud(predicted_azimuth_img)

        # Save B_total to fits
        btotal_fits_file = os.path.join(output_path, 'b_total_{}.fits'.format(time_stamp))
        try:
            os.remove(btotal_fits_file)
        except OSError:
            pass
        btotal_fits = fits.PrimaryHDU(predicted_btotal_img)
        btotal_fits.writeto(btotal_fits_file)

        # Save inclination to fits
        inclination_fits_file = os.path.join(output_path, 'inclination_{}.fits'.format(time_stamp))
        try:
            os.remove(btotal_fits_file)
        except OSError:
            pass
        inclination_fits = fits.PrimaryHDU(predicted_inclination_img)
        inclination_fits.writeto(inclination_fits_file)

        # Save azimnuth to fits
        azimuth_fits_file = os.path.join(output_path, 'azimuth_{}.fits'.format(time_stamp))
        try:
            os.remove(btotal_fits_file)
        except OSError:
            pass
        azimuth_fits = fits.PrimaryHDU(predicted_azimuth_img)
        azimuth_fits.writeto(azimuth_fits_file)
    print('Done saving..')


if __name__ == '__main__':
    model = load_model()
    input_path = 'inputs'  # edit your input path
    output_path = 'outputs'  # edit your output path

    for file in os.listdir(input_path):
        print('---------- working on', file, '----------')
        fits_file = os.path.join(input_path, file)
        hdu = fits.open(fits_file)
        hdu.verify('fix')
        data = hdu[0].data
        test_data, data_height, data_width = read_data(data)
        start = time.time()
        predict_results = inverse(test_data, model)
        end = time.time()
        print('Inversion Time:', np.round(end - start, 1), 's')
        # The code will also save b_total, inclination and azimuth if save_mag_field_o = True,
        # otherwise only save bx, by and bz and Doppler width and LOS velocity.
        save_results(predict_results, output_path, file[: file.find('.')], data_height, data_width, save_mag_field_o=False)

