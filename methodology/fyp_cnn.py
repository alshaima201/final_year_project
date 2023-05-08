# data manipulation
import pandas as pd
import numpy as np

# for data visualisation
import matplotlib.pyplot as plt
import seaborn as sb
 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, Flatten, Dense
import sklearn.metrics as mt

from sklearn.linear_model import LinearRegression

import time 

def CNN(df):
    '''
        apply CNN to a the input feature of a stock (close prices) and evaluate the performance of the model. returns graphs and results of evalutation metrics
        
            Parameters:
                    df (Pandas Series): input feature data
    '''
    # prepare data from lstm model
    full_data = df.values

    # independent variable
    sample_x = []
    sample_y = []
    sequence_length = 7
    # loop through close_list to sample data for training
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
    
    # reshape train_x into 3D array(number of samples, time steps, number of features)
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
    # print('Input Data Shape: {}' .format(train_x.shape))

    # sequence_length: no. of time steps in each input sequence
    sequence_length = train_x.shape[1]
    # input_dim no. of features in each input sequence
    input_dim = train_x.shape[2]
    
    # define model
    model = Sequential()
    
    # first 1D convolutional layer
    model.add(Conv1D(filters = 32, kernel_size = 3, activation='relu', input_shape = (sequence_length, input_dim)))
    
    # 2nd 1D convolutional layer
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    
    # maxpool layer
    model.add(MaxPool1D(pool_size = 2))
    # flaten layer
    model.add(Flatten())
    
    # no. of output classes
    output_dim = 1
    # output layer
    model.add(Dense(output_dim))

    # compile model
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

    # start timer
    timer_start = time.time()
    
    # train model
    model.fit(train_x, train_y, epochs = 100)
    
    # end timer and print total time taken 
    timer_end = time.time()
    print("Time Taken: ", round(timer_end - timer_start), "seconds")

    # evaluate model
    test_loss, test_acc = model.evaluate(test_x, test_y, verbose = 2)
    print('Test Accuracy:', test_acc)
    
    # predict test data
    # denormalise data
    predicted_y = scaler.inverse_transform(model.predict(test_x))
    
    # denormalise data
    test_y = scaler.inverse_transform(test_y)

    # evaluating prediction error from sklearn.metrics
    print('Model Evaluation')
    # MAE
    print("Mean Absolute Error:", mt.mean_absolute_error(test_y, predicted_y))
    # RMSE
    print('Root Mean Squared Error:', np.sqrt(mt.mean_squared_error(test_y, predicted_y)))
    # MAPE
    print(f'Mean Absolute Percentage Error: {np.mean(np.abs((test_y - predicted_y) / test_y)) * 100}%')

    # plot results
    plt.plot(predicted_y, color = 'blue', label = 'Predicted Price')
    plt.plot(test_y, color = 'lightblue', label = 'Original Price')
    plt.title('1D Convolutional Neural Network on {} Prices' .format(df.name))
    plt.xlabel('Trading Across Time')
    plt.xticks([], [])
    plt.ylabel('Stock Price')
    plt.legend()
    fig = plt.gcf()
    fig.set_figwidth(20)
    fig.set_figheight(6)
    plt.show()
    
    # evaluation with regression
    lr = LinearRegression()
    lr.fit(test_x, test_y)
    
    pred_y = lr.predict(test_x)
    
    # plot results
    # plot resulting regression line & 95% confidence interval
    sb.regplot(x = test_y, y = pred_y, scatter = True, fit_reg = True)
    plt.title('Linear Regression of CNN')
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