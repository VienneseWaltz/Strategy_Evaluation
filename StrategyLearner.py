""""""  		  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		   	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			 	 	 		 		 	
or edited.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Student Name: Stella Soh  		  	   		   	 			  		 			 	 	 		 		 	
GT User ID: lsoh3 		  	   		   	 			  		 			 	 	 		 		 	
GT ID: 903641298 	  	   		   	 			  		 			 	 	 		 		 	
"""
import pandas as pd
import numpy as np
import datetime as dt
import BagLearner as bl
import RTLearner as rt
from indicators import *
from util import get_data, plot_data
from scipy.stats import zscore
from marketsimcode import compute_portvals



class StrategyLearner(object):  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 			  		 			 	 	 		 		 	
        If verbose = False your code should not generate ANY output.  		  	   		   	 			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		   	 			  		 			 	 	 		 		 	
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 			  		 			 	 	 		 		 	
    :type impact: float  		  	   		   	 			  		 			 	 	 		 		 	
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 			  		 			 	 	 		 		 	
    :type commission: float  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    # constructor  		  	   		   	 			  		 			 	 	 		 		 	
    def __init__(self, verbose=False, impact=0.005, commission=9.95):
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Constructor method  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        self.verbose = verbose  		  	   		   	 			  		 			 	 	 		 		 	
        self.impact = impact  		  	   		   	 			  		 			 	 	 		 		 	
        self.commission = commission
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={'leaf_size':5}, bags=20, boost=False, verbose=False)


    def add_evidence(  		  	   		   	 			  		 			 	 	 		 		 	
        self,  		  	   		   	 			  		 			 	 	 		 		 	
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),  		  	   		   	 			  		 			 	 	 		 		 	
        ed=dt.datetime(2009, 12, 31),
        sv=100000,
    ):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Trains your strategy learner over a given time frame.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
        :param symbol: The stock symbol to train on  		  	   		   	 			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		   	 			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		   	 			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		   	 			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		   	 			  		 			 	 	 		 		 	
        :type sv: int  		  	   		   	 			  		 			 	 	 		 		 	
        """
        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates, addSPY=False).dropna() # drop SPY, only portfolio symbols
        prices.fillna(method='ffill')
        prices.fillna(method='bfill')

        # Lookback window when computing SMA, EMA and BBP
        lookback = 20

        # Lookforward window when computing N-day returns
        lookforward = 20
        #######################################################
        # Obtain the 3 technical indicators
        ########################################################
        # Simple moving average
        SMA = cal_SMA(prices, lookback)

        # Exponential moving average
        EMA = cal_exp_ma(prices, lookback)

        # Bollinger BBP value
        BBP = cal_Bollinger_bands(prices, lookback)

        ##################################################
        # Construct train_X from the 3 chosen indicators
        ##################################################
        train_X = np.zeros((prices.shape[0] - lookforward - lookback, 3))
        # z-score normalize the 3 indicators
        train_X[:, 0] = zscore(SMA.values[lookback:-lookforward,0])
        train_X[:, 1] = zscore(EMA.values[lookback:-lookforward, 0])
        train_X[:, 2] = zscore(BBP.values[lookback:-lookforward, 0])

        ###################################################
        # Construct train_Y
        ###################################################
        YBUY = self.impact
        YSELL = -YBUY

        # future_returns is a Numpy array of size ((prices.shape[0] - lookback), )
        future_returns = prices.values[lookback:, 0]
        future_returns[:-lookforward] = (future_returns[lookforward:]/future_returns[:-lookforward]) - 1.0
        future_returns = future_returns[:-lookforward]
        train_Y = future_returns
        # Implementing the indicator design mentioned on Classification Trader hints. train_Y values greater than
        # YBUY is set to 1, values less than YSELL, set to -1, and values between YBUY and YSELL, set to 0.
        train_Y[train_Y > YBUY] = 1
        train_Y[train_Y < YSELL] = -1
        train_Y[(train_Y <= YBUY) & (train_Y >= YSELL)] = 0

        # Train the learner
        self.learner.add_evidence(train_X, train_Y)

  		  	   		   	 			  		 			 	 	 		 		 	
    # this method should use the existing policy and test it against new data  		  	   		   	 			  		 			 	 	 		 		 	
    def testPolicy(  		  	   		   	 			  		 			 	 	 		 		 	
        self,  		  	   		   	 			  		 			 	 	 		 		 	
        symbol="JPM",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=100000,
    ):  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
        Tests your learner using data outside of the training data  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
        :param symbol: The stock symbol that you trained on on  		  	   		   	 			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		   	 			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		   	 			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		   	 			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		   	 			  		 			 	 	 		 		 	
        :type sv: int  		  	   		   	 			  		 			 	 	 		 		 	
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		   	 			  		 			 	 	 		 		 	
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		   	 			  		 			 	 	 		 		 	
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		   	 			  		 			 	 	 		 		 	
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		   	 			  		 			 	 	 		 		 	
        :rtype: pandas.DataFrame  		  	   		   	 			  		 			 	 	 		 		 	
        """  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
        # here we build a fake set of trades  		  	   		   	 			  		 			 	 	 		 		 	
        # your code should return the same sort of data  		  	   		   	 			  		 			 	 	 		 		 	
        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates, addSPY=False).dropna()  # only portfolio symbol, SPY is dropped
        prices.fillna(method='ffill')
        prices.fillna(method='bfill')

        # Lookback window used when computing the indicators
        lookback = 20

        # Obtain the 3 technical indicators
        # Simple moving average
        SMA = cal_SMA(prices, lookback)

        # Exponential moving average
        EMA = cal_exp_ma(prices, lookback)

        # Bollinger BBP value
        BBP = cal_Bollinger_bands(prices, lookback)

        ##################################################
        # Construct test_X from the 3 chosen indicators
        ##################################################
        test_X = np.zeros((prices.shape[0] - lookback, 3))
        # z-score normalize the 3 indicators
        test_X[:, 0] = zscore(SMA.values[lookback:, 0])
        test_X[:, 1] = zscore(EMA.values[lookback:, 0])
        test_X[:, 2] = zscore(BBP.values[lookback:, 0])

        # Query the learner to obtain test_Y
        test_Y = self.learner.query(test_X)

        desired_position = 1000 * test_Y
        trades = desired_position.copy()
        trades[1:] = np.diff(desired_position)
        # Just in case trades[0] is NaN, we have copied desired_position[0] into the first entry of trades
        trades[0] = desired_position[0]

        # Construct df_trades
        df_trades = prices.copy()
        # Zero out the first 20 (lookback) df_trades values. No indicators so a HOLD position.
        df_trades.values[:lookback, :] = 0
        # df_trades DataFrame consist of values from the 20th (lookback) items onwards
        df_trades.values[lookback:, 0] = trades

        return df_trades

    def test_code(self):
        symbol = 'ML4T-220'
        self.add_evidence(symbol,sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000 )
        trades = self.testPolicy(symbol, sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
        # Process the portfolio values of 'ML4T-220' using Manual Strategy
        sl_portvals = compute_portvals(trades, start_val=100000, commission=self.commission, impact=self.impact)

        # Get the benchmark trades for the same period. Start the benchmark portfolio with $100,000 cash,
        # investing in 1000 shares of the symbol on the first trading day, and holding that position throughout.
        benchmark_trades = trades.copy()
        benchmark_trades.values[0, 0] = 1000.0
        benchmark_trades.values[1:, :] = 0.0

        # Process the portofolio values of benchmark
        bm_portvals = compute_portvals(benchmark_trades, start_val=100000, commission=self.commission, impact=self.impact)

        ###############################################################
        # Printing Portfolio Statistics
        ###############################################################
        sl_cr, sl_adr, sl_sddr, sl_sharpe_ratio = cal_port_stats(sl_portvals)

        print()
        print(f"************ Strategy Learner Portfolio Statistics ***********")
        print(f"Cumulative Returns: {'%.4f' % sl_cr}")
        print(f"Average Daily Returns: {'%.4f' % sl_adr}")
        print(f"Standard Deviation of Daily Returns: {'%.4f' % sl_sddr}")
        print(f"Sharpe Ratio of Strategy Learner: {'%.4f' % sl_sharpe_ratio}")

        ###############################################################
        # Printing Benchmark Portfolio Statistics
        ###############################################################
        bm_cr, bm_adr, bm_sddr, bm_sharpe_ratio = cal_port_stats(bm_portvals)

        print(f"\n************ Benchmark Portfolio Statistics ***********")
        print(f"Cumulative Return: {'%.4f' % bm_cr}")
        print(f"Average Daily Returns: {'%.4f' % bm_adr}")
        print(f"Standard Deviation of Daily Returns: {'%.4f' % bm_sddr}")
        print(f"Sharpe Ratio of Benchmark: {'%.4f' % bm_sharpe_ratio}")
        print()

    def author(self):
        return "lsoh3"


if __name__ == "__main__":
    sl = StrategyLearner()
    sl.test_code()
    #sl.add_evidence(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    #sl.testPolicy(symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
    print("One does not simply think up a strategy")


