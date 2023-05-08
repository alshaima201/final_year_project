# data manipulation
import pandas as pd
import numpy as np

# for data visualisation
import matplotlib.pyplot as plt
import seaborn as sb

from scipy.stats import skew, kurtosis

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

import sklearn.metrics as mt

import time

# auto- regressive integrated moving average (ARIMA) model
def arima(df):
    '''
        apply arima model to a the input feature of a stock (close prices) and evaluate the performance of the model. returns graphs and results of evalutation metrics
        
            Parameters:
                    df (Pandas Series): input feature data
    '''
    # ARIMA(p, d, q) 
    # p: autoregression
    # d: order of differencing
    # q: moving average
    
    # check if the series is non-stationary (d != 0) with augmented dickey fuller (adf) test 
    result = adfuller(df.dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    
    # transform data into stationary by differencing to estimate d-value
    difference = df - df.shift(1)
    plt.plot(difference)
    plt.title('Differencing Plot')
    plt.show()
    
    # check with adf test
    result = adfuller(difference.dropna())
    print('ADF Statistic: %f' % result[0])
    # p-value < 0.05 means stationary data, value of d = 0
    print('p-value: %f' % result[1])
    
    # autocorrelation and partial autocorrelation to estimate p-value and q-value
    # p, q = 1
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (18, 5))
    
    plot_acf(difference.dropna(), ax = axes[0])
    plot_pacf(difference.dropna(), ax = axes[1])
    
    plt.show()

    # from pandas.plotting import autocorrelation_plot
    # print(autocorrelation_plot(a))

    # p = 1, d = 0, q = 0
    order = (1, 0, 1)
    
    # prepare data for arima model
    full_data = df.values
    
    # random_state generates reproducible output across multiple function calls.
    train, test = train_test_split(full_data, train_size = 0.8, random_state = 999)
    
    # normalize the dataset; make values between 0 and 1
    scaler = MinMaxScaler(feature_range = (0, 1))
    train = scaler.fit_transform(train.reshape(-1, 1))
    test = scaler.fit_transform(test.reshape(-1, 1))
    
    
    # define model
    arima = ARIMA(train, order = order)
    
    # start timer
    timer_start = time.time()
    
    # fit model
    result = arima.fit()
    
    # end timer and print total time taken 
    timer_end = time.time()
    print("Time Taken: ", round(timer_end - timer_start), "seconds")

    print(result.summary())

    # generate forecast
    forecast = result.forecast(steps = len(test))
    
    # denormalise data
    forecast = scaler.inverse_transform(forecast.reshape(-1, 1))
    test = scaler.inverse_transform(test.reshape(-1, 1))

    # plot forecast
    plt.plot(test, label = 'original')
    plt.plot(forecast, label = 'forecast')
    plt.legend()
    plt.show()
    
    # evaluating prediction error from sklearn.metrics
    print('Model Evaluation')
    # MAE
    print("Mean Absolute Error:", mt.mean_absolute_error(test,forecast))
    # RMSE
    print('Root Mean Squared Error:', np.sqrt(mt.mean_squared_error(test, forecast)))
    # MAPE
    print(f'Mean Absolute Percentage Error: {np.mean(np.abs((test - forecast) / test)) * 100}%')
    
    residuals = test - forecast
    
    # evaluation with regression
    lr = LinearRegression()
    lr.fit(residuals.reshape(-1, 1), test)
    
    # plot results
    # plot resulting regression line & 95% confidence interval
    sb.regplot(x = test, y = forecast, scatter = True, fit_reg = True)
    plt.title('Linear Regression of ARIMA')
    plt.xlabel('Original Price')
    plt.ylabel('Predicted Price')
    plt.show()
    
    # evaluating prediction error from sklearn.metrics
    print('Model Evaluation')
    # MAE
    mae = mt.mean_absolute_error(residuals.reshape(-1, 1), test)
    print("Mean Absolute Error:", mae)
    # RMSE
    print('Root Mean Squared Error:', np.sqrt(mt.mean_squared_error(residuals.reshape(-1, 1), test)))