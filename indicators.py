""""
#######  Implementing indicators as functions that operate on DataFrames.
#############################################################################
#######
Student Name: Stella Soh
GT User ID: lsoh3
GT ID: 903641298
"""

import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from util import get_data, plot_data
from marketsimcode import *


def cal_SMA(prices, lookback):
    '''
    Compute the simple moving average.
    
    :param prices: The prices of a stock symbol 
    :type prices: DataFrame object 
    :param lookback : window size to look back 
    :type lookback: int 
    :return:SMA - The simple moving average index for the lookback window
    :rtype: pandas.DataFrame
    '''
    # Credit goes to David Byrd's "Vectorize Me" portion of
    # "Time Series Data 2" video: https://edstem.org/us/courses/5504/lessons/12764/slides/64913
    # for this vectorized version of SMA calculation.

    SMA = prices.cumsum()
    SMA.values[lookback:, :] = (SMA.values[lookback:, :] - SMA.values[:-lookback, :]) / lookback
    SMA.iloc[:lookback, :] = np.nan

    return SMA


def cal_price_over_SMA(prices, lookback):
    '''
    Compute the price / SMA
    :param prices: The prices of a stock symbol
    :type prices: DataFrame object
    :param lookback : window size to look back
    :type lookback: int
    :return: price/SMA
    :rtype: pandas.DataFrame
    '''
    price_over_SMA = prices / cal_SMA(prices, lookback)

    return price_over_SMA


def cal_Bollinger_bands(prices, lookback):
    '''
    Function computes the upper and lower Bollinger bands, which are a set of trendlines
    plotted 2 standard deviations away from a Simple Moving Average of a stock price.

    :param prices: The prices of a stock symbol
    :type prices: DataFrame object
    :param lookback : window size to look back
    :type lookback: int
    :return:  bollinger_up_band, bollinger_lower_band, bollinger_bands_val
    :rtype: pandas.DataFrame
    '''
    # Number of standard deviation
    num_of_std = 2

    # Compute the Bollinger upper and lower bands and the Bollinger bands value
    SMA = cal_SMA(prices, lookback)
    rolling_std = prices.rolling(window=lookback, min_periods=lookback).std()
    bollinger_up_band = SMA + (num_of_std * rolling_std)
    bollinger_lower_band = SMA - (num_of_std * rolling_std)
    bollinger_bands_val = (prices - bollinger_lower_band) / (bollinger_up_band - bollinger_lower_band)

    #return bollinger_up_band, bollinger_lower_band, bollinger_bands_val
    return bollinger_bands_val


def cal_exp_ma(prices, num_of_days=12):
    '''
    Compute the exponential moving average using prices and number of days
    :param prices: The prices of a stock symbol
    :type prices: DataFrame object
    :param num_of_days: Minimum number of observations required
    :type num_of_days: int
    :return: exp_ma: The exponential moving average

    '''
    exp_ma = prices.ewm(span=num_of_days, adjust=False).mean()
    ema = prices / exp_ma
    return ema


def author():
    return 'lsoh3'




















