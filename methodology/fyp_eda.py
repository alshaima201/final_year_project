# data manipulation
import pandas as pd
import numpy as np

# for data visualisation
import matplotlib.pyplot as plt
import seaborn as sb

# statistical calculations
from scipy.stats import skew, kurtosis, norm

def reduct(df):
    # specific date range
    # convert date format in dataset to pandas date format
    df['Date'] = pd.to_datetime(df['Date'])

    # specify date range / time period
    start_date = '2009-01-01'
    end_date = '2019-01-01'

    # apply selected range as a mask on dataframe
    mask = (df['Date'] > start_date) & (df['Date'] <= end_date)
    df = df.loc[mask]
    
    return df

def cleaning(df):
    '''
        preparing data for ml models by cleaning dataset from null values and unnecessary
        Parameters:
            df (Pandas DataFrame): target df for cleaning
        Return:
            df (Pandas DataFrame): df after cleaning
    '''
    
    # delete unwanted columns
    df = df.drop(columns = ['Adj Close', 'Volume'])
    
    # calculate number of duplicate rows
    dup = df.duplicated().sum()
    
    # if there are duplicate rows
    if(dup != 0):
        # drop duplicate rows
        df.drop_duplicates()
        
    # drop null values
    df = df.dropna()

    return df

# decision column is to decide: buy, sell, or neutral
# condition: if this open price > previous close price == sell
# condition: if this open price < previous close price == buy
# condition: if this open price == previous close price == neutral
def decision(op, cl):
    '''
    creating a new column called decision to give user an indication of either selling or buying stock.
    '''
    if op > cl:
            return 'sell'
    elif op < cl:
        return 'buy'
    else:
        return 'neutral'
        
def eda(df):
    '''
    creating plots and making calculations on the data to understand the dataset
    
    Parameters:
        df (Pandas Dataframe): target dataframe for eda implementation
    '''
    # display info of column names and datatypes in dataset
    df.info()
    print('\n')

    # pairplot of column relationships in the dataset
    sb.pairplot(df)
    plt.suptitle('Pairplot', y = 1)
    plt.show()
    
    print('\n')
    
    # correlation heatmap of numerical fields in dataset
    # list of numeric fields; open, high, low and close prices
    numeric_columns = ['Open', 'High', 'Low', 'Close']

    # correlation between numeric data variables
    corr = df.loc[:,numeric_columns].corr()
    print('Correlation Table')
    print(corr)
    
    print('\n')

    # plot heatmap using seaborn
    sb.heatmap(df.corr(numeric_only = True), cmap = 'Blues')
    plt.title('Correlation Heatmap')
    plt.show()
    
    print('\n')
    
    # average price plot
    # consider both open and close prices
    plt.figure(figsize = (18, 9))
    plt.plot(range(df.shape[0]), (df['Open'] + df['Close']) / 2)
    plt.xticks([],[])
    plt.ylabel('Average Price')
    plt.title('Average Price  of Stocks Across Time')
    plt.show()
    
    print('\n')
    
    # distribution plots
    # define subplots
    fig, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (18, 5))

    # distribution of open prices
    axes[0].hist(df['Open'], bins = 20)
    axes[0].set_xlabel('Open Prices')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Open Prices')

    # distribution of high prices
    axes[1].hist(df['High'], bins = 20)
    axes[1].set_xlabel('Highest Prices')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of High Prices')

    # distribution of low prices
    axes[2].hist(df['Low'], bins = 20)
    axes[2].set_xlabel('Lowest Prices')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Low Prices')

    # distribution of close prices
    axes[3].hist(df['Close'], bins = 20)
    axes[3].set_xlabel('Close Prices')
    axes[3].set_ylabel('Frequency')
    axes[3].set_title('Distribution of Close Prices')

    # show plot
    plt.show()
    
    print('\n')
    
    # show daily returns of close prices using percentage change method
    # daily returns
    returns = df['Close'].pct_change(1).dropna()
    
    # mean
    # positive mean explains the +ve drift of the price time series
    mean = np.mean(returns)
    print('The Mean Value of Returns =', mean)

    # standard deviation is a measure of risk, higher std == higher risk
    std = np.std(returns)
    print('The Standard Deviation of Returns =', std)
    
    # skewness
    print('Skewness of Returns is:', skew(returns))
    # kurtosis
    print('Kurtosis of Returns is:', kurtosis(returns))
    
    print('\n')
    
    ls = np.linspace(0.01,0.99,1000)
    q1 = np.quantile(returns, ls)
    q2 = norm.ppf(ls, mean, std)
    plt.plot(q1, q2, label = 'distribution of returns')
    plt.plot([min(q1), max(q1)], [min(q2), max(q2)], label = 'normal distribution')
    plt.xlim((min(q1), max(q1)))
    plt.ylim((min(q2), max(q2)))
    plt.xlabel("Daily Returns")
    plt.ylabel("Distribution")
    plt.legend()
    plt.show()
    
    print('\n')
    
    # distribution plot of returns
    sb.displot(returns)
    # density plot of returns
    sb.kdeplot(returns, color = 'pink')
    plt.title('Returns Distribution and Density')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.show()
    
    print('\n')
    