import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

import sklearn.metrics as mt

import time

def rand_forest(df):
    '''
            apply random forest to a the input feature of a stock (close prices) and evaluate the performance of the model. returns graphs and results of evalutation metrics
        
            Parameters:
                    df (Pandas Series): input feature data
    '''
    # prepare data for random forest model
    full_data = df.values

    # independent variable
    sample_x = []
    sample_y = []
    sequence_length = 7
    
    # loop through full data to sample data for training
    for i in range(sequence_length, len(full_data) , 1):
        sample_x.append(full_data[i - sequence_length:i])
        sample_y.append(full_data[i])

    # splitting the dataset into 80% train set, 20% test set
    x =  pd.DataFrame(sample_x)
    # dependent variable
    y = pd.DataFrame(sample_y)
    
    # random_state generates reproducible output across multiple function calls.
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size = 0.8, random_state = 999)

    # normalize the dataset; make values between 0 and 1
    scaler = MinMaxScaler(feature_range = (0, 1))
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)
    train_y = scaler.fit_transform(train_y)
    test_y = scaler.fit_transform(test_y)
    
    # print('Input Data Shape: {}' .format(train_x.shape))
      
    # define regressor
    forest = RandomForestRegressor()
  
    # start timer
    timer_start = time.time()
    
    # fit regressor with x and y data
    forest.fit(train_x, train_y.ravel()) 
    
    # end timer and print total time taken 
    timer_end = time.time()
    print("Time Taken: ", round(timer_end - timer_start), "seconds")
    
    # prediction of test data
    # predicted_y = scaler.inverse_transform(forest.predict(test_x))
    predicted_y = forest.predict(test_x)
    predicted_y = scaler.inverse_transform(predicted_y.reshape(-1, 1))
    test_y = scaler.inverse_transform(test_y)
    # must be (100, 7) (100, 1) (100, 1)
    # print(test_x.shape, predicted_y.shape, test_y.shape)
    
    train_x, train_y = scaler.inverse_transform(train_x), scaler.inverse_transform(train_y.reshape(-1, 1))
    # must be (396, 1) (396, 1)
    # print(train_x.shape, train_y.shape)
    
    # graph of comparison, original vs predictions
    plt.figure(figsize = (18, 9))
    plt.plot(test_y, color = 'blue', label = 'original')
    # line plot predicted data
    plt.plot(predicted_y, color = 'pink', label = 'predicted') 
    plt.title('Random Forest Regression Model of {} Prices' .format(df.name))
    plt.xlabel('Original Price')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.show()
    
    # evaluating prediction error from sklearn.metrics
    print('Model Evaluation')
    # MAE
    print("Mean Absolute Error:", mt.mean_absolute_error(test_y, predicted_y))
    # RMSE
    print('Root Mean Squared Error:', np.sqrt(mt.mean_squared_error(test_y, predicted_y)))
    # MAPE
    print(f'Mean Absolute Percentage Error: {np.mean(np.abs((test_y - predicted_y) / test_y)) * 100}%')
    
    # evaluation with regression
    lr = LinearRegression()
    lr.fit(test_x, test_y)
    
    pred_y = lr.predict(test_x)
    
    # plot results
    # plot resulting regression line & 95% confidence interval
    sb.regplot(x = test_y, y = pred_y, scatter = True, fit_reg = True)
    plt.title('Linear Regression of Random Forest')
    plt.xlabel('Original Price')
    plt.ylabel('Predicted Price')
    plt.show()
    
    # evaluating prediction error from sklearn.metrics
    print('Model Evaluation')
    # MAE
    mae = mt.mean_absolute_error(test_y, predicted_y)
    print("Mean Absolute Error:", mae)    
    # RMSE
    print('Root Mean Squared Error:', np.sqrt(mt.mean_squared_error(test_y, pred_y)))
    # MAPE
    print(f'Mean Absolute Percentage Error: {np.mean(np.abs((test_y - pred_y) / test_y)) * 100}%')