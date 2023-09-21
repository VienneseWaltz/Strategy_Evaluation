""""""
"""  		  	   		   	 			  		 			 	 	 		 		 	
Implementing Manual Strategy of Project 8 
Student Name: Stella Soh		  	   		   	 			  		 			 	 	 		 		 	
GT User ID: lsoh3 	  	   		   	 			  		 			 	 	 		 		 	
GT ID: 903641298  	   		   	 			  		 			 	 	 		 		 	
"""
import numpy as np
import pandas as pd
import datetime as dt
from util import get_data, plot_data
from indicators import *
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals

class ManualStrategy(object):
    """
        A manual rule-based strategy learner using 3 indicators developed in Project 6.

        :param verbose: If “verbose” is True, my code can print out information for debugging.
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

    def author(self):
        return "lsoh3"

    def testPolicy(
            self,
            symbol = 'JPM',
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 12, 31),
            sv=100000
    ):
        '''
        The trading strategies used to enter and exit positions in the stock.
        :param symbol: The symbol under test
        :param sd: The starting date of observation
        :param ed: The ending date of observation
        :param sv: The starting value of investment
        :return:
        '''
        dates = pd.date_range(sd, ed)
        df_prices = get_data([symbol], dates, addSPY=False).dropna()

        # Fill forward, then fill backward to work around the gaps in the DataFrame
        df_prices.fillna(method='ffill')
        df_prices.fillna(method='bfill')

        # Number of days to lookback on
        lookback = 20

        # #########################################
        # Obtain the 3 technical indicators
        ###########################################
        # Simple moving average
        SMA = cal_SMA(df_prices, lookback)

        # Exponential moving average
        EMA = cal_exp_ma(df_prices, lookback)

        # Bollinger BB_value
        BBP = cal_Bollinger_bands(df_prices, lookback)

        ##############################################################################################################
        # Manual Strategy: long or short 'JPM' stock based on BB crossings.
        ###############################################################################################################
        # These are the situations that are addressed:
        # (0): If any of the SMA, EMA, or BBP DataFrames display 'NaN', the desired_position is a HOLD position.
        # (A): If df_prices is higher than Bollinger upper band, this is an overbought situation. The security is
        #      trading at a level above its intrinsic of fair value. This stock may be a good candidate for sale.
        # (B): If df_prices is below Bollinger lower band, this is an oversold situation. This is a signal to buy the
        #      stock.
        # (C): The EMA line pushes past the SMA line, and this is an uptrend. This is a BUY signal
        # (D): The EMA line falls below the SMA line, and this is a downtrend. This is a SELL signal
        ################################################################################################################

        # Make a copy of df_prices
        desired_position = df_prices.copy()

        for date, data in desired_position.iterrows():
            if math.isnan(SMA.at[date, symbol]) or math.isnan(EMA.at[date, symbol]) or math.isnan(BBP.at[date,symbol]):
                desired_position.at[date, symbol] = 0            # No indicator. A HOLD position
            elif BBP.at[date, symbol] > 1:                       # Situation (A): Overbought & overvalued situation -> SELL
                desired_position.at[date, symbol] = -1000

            elif BBP.at[date, symbol] < 0:                       # Situation (B): Oversold & undervalued situation -> BUY
                desired_position.at[date, symbol] = 1000

            elif EMA.at[date, symbol] > SMA.at[date, symbol]:    # Situation (C): Uptrend
                desired_position.at[date, symbol] = 1000

            elif EMA.at[date, symbol] < SMA.at[date, symbol]:    # Situation (D): Downtrend
                desired_position.at[date, symbol] = -1000

            else:
                desired_position.at[date, symbol] = 0

        #########################################################################################
        # Now that the desired_position Dataframe is built up, we are ready to create
        # the trades DataFrame.
        ########################################################################################
        trades = desired_position.diff(periods=1, axis=0)
        # Place 0 for first trading day
        trades.values[0, 0] = 0

        return trades


    def test_code(self, saveFig=True):
        symbol = 'JPM'

        #######################################################
        # Processing in-sample trades and plotting a chart
        #######################################################

        trades_in_sample = self.testPolicy(symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
        ms_portvals_in_sample = compute_portvals(trades_in_sample, start_val=100000, commission=self.commission, impact=self.impact)

        bm_trades_in_sample = trades_in_sample.copy()
        bm_trades_in_sample.values[0, 0] = 1000.0
        bm_trades_in_sample.values[1:, :] = 0.0

        # Process the portofolio values of benchmark_trades_in_sample
        bm_portvals_in_sample = compute_portvals(bm_trades_in_sample, start_val=100000, commission=self.commission,
                                                 impact=self.impact)

        ###############################################################
        # Printing Portfolio Statistics
        ###############################################################
        ms_cr_1, ms_adr_1, ms_sddr_1, ms_sharpe_ratio_1 = cal_port_stats(ms_portvals_in_sample)

        print()
        print(f"************ Manual Strategy Portfolio Statistics ***********")
        print(f"Cumulative Returns: {'%.4f' % ms_cr_1}")
        print(f"Average Daily Returns: {'%.4f' % ms_adr_1}")
        print(f"Standard Deviation of Daily Returns: {'%.4f' % ms_sddr_1}")
        print(f"Sharpe Ratio of Manual Strategy: {'%.4f' % ms_sharpe_ratio_1}")

        ###############################################################
        # Printing Benchmark Portfolio Statistics
        ###############################################################
        bm_cr_1, bm_adr_1, bm_sddr_1, bm_sharpe_ratio_1 = cal_port_stats(bm_portvals_in_sample)

        print(f"\n************ Benchmark Portfolio Statistics ***********")
        print(f"Cumulative Return: {'%.4f' % bm_cr_1}")
        print(f"Average Daily Returns: {'%.4f' % bm_adr_1}")
        print(f"Standard Deviation of Daily Returns: {'%.4f' % bm_sddr_1}")
        print(f"Sharpe Ratio of Benchmark: {'%.4f' % bm_sharpe_ratio_1}")
        print()

        ###############################################################
        # Plot charts
        ###############################################################
        # Normalize - Divide by the first row
        ms_portvals_in_sample = ms_portvals_in_sample / ms_portvals_in_sample[0]
        bm_portvals_in_sample = bm_portvals_in_sample / bm_portvals_in_sample[0]

        # ms_portvals.plotting.register_matplotlib_converters()
        ax = ms_portvals_in_sample.plot(label='Manual Strategy', fontsize=12, color='red')

        bm_portvals_in_sample.plot(ax=ax, color='green', label='Benchmark')

        buy_trades = trades_in_sample.loc[trades_in_sample[symbol] > 0]
        sell_trades = trades_in_sample.loc[trades_in_sample[symbol] < 0]

        # Credit goes to https://stackoverflow.com/questions/52853671
        # /convert-datetimeindex-to-datetime for conversion of datetime index
        # to datetime
        buy_dates = list(buy_trades.index.to_pydatetime())
        sell_dates = list(sell_trades.index.to_pydatetime())

        ymin, ymax = ax.get_ylim()

        # Plotting blue vertical lines to indicate LONG (BUY) entry points
        for d in buy_dates:
            ax.axvline(d, ymin=0, ymax=ymax, color='blue')

        # Plotting black vertical lines to indicate SHORT (SELL) entry points
        for d in sell_dates:
            ax.axvline(d, ymin=0, ymax=ymax, color='black')

        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Values')
        plt.title('Manual Strategy vs Benchmark - In-sample')
        plt.legend()

        if saveFig:
            fig1 = plt.gcf()
            fig1.savefig('Figure1.png', format='png', dpi=120)
            plt.close(fig1)
        else:
            plt.show()

        ################################################################
        # Processing out-of-sample trades and plotting a chart
        ################################################################
        trades = self.testPolicy(symbol, sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
        # Process the portfolio values of 'JPM' using Manual Strategy
        ms_portvals = compute_portvals(trades, start_val=100000, commission=self.commission, impact=self.impact)

        # Get the benchmark trades for the same period. Start the benchmark portfolio with $100,000 cash,
        # investing in 1000 shares of the symbol on the first trading day, and holding that position throughout.
        benchmark_trades = trades.copy()
        benchmark_trades.values[0, 0] = 1000.0
        benchmark_trades.values[1:, :] = 0.0

        # Process the portofolio values of benchmark_trades
        bm_portvals = compute_portvals(benchmark_trades, start_val=100000, commission=self.commission, impact=self.impact)


        ###############################################################
        # Printing Portfolio Statistics
        ###############################################################
        ms_cr_2, ms_adr_2, ms_sddr_2, ms_sharpe_ratio_2 = cal_port_stats(ms_portvals)

        print()
        print(f"************ Manual Strategy Portfolio Statistics ***********")
        print(f"Cumulative Returns: {'%.4f' % ms_cr_2}")
        print(f"Average Daily Returns: {'%.4f' % ms_adr_2}")
        print(f"Standard Deviation of Daily Returns: {'%.4f' % ms_sddr_2}")
        print(f"Sharpe Ratio of Manual Strategy: {'%.4f' % ms_sharpe_ratio_2}")


        ###############################################################
        # Printing Benchmark Portfolio Statistics
        ###############################################################
        bm_cr_2, bm_adr_2, bm_sddr_2, bm_sharpe_ratio_2 = cal_port_stats(bm_portvals)

        print(f"\n************ Benchmark Portfolio Statistics ***********")
        print(f"Cumulative Return: {'%.4f' % bm_cr_2}")
        print(f"Average Daily Returns: {'%.4f' % bm_adr_2}")
        print(f"Standard Deviation of Daily Returns: {'%.4f' % bm_sddr_2}")
        print(f"Sharpe Ratio of Benchmark: {'%.4f' % bm_sharpe_ratio_2}")
        print()


        ###############################################################
        # Plot charts
        ###############################################################
        # Normalize - Divide by the first row
        ms_portvals = ms_portvals / ms_portvals[0]
        bm_portvals = bm_portvals / bm_portvals[0]

        # ms_portvals.plotting.register_matplotlib_converters()
        ax = ms_portvals.plot(label='Manual Strategy', fontsize=12, color='red')

        bm_portvals.plot(ax=ax, color='green', label='Benchmark')
        

        buy_trades = trades.loc[trades[symbol] > 0]
        sell_trades = trades.loc[trades[symbol] < 0]

        # Credit goes to https://stackoverflow.com/questions/52853671
        # /convert-datetimeindex-to-datetime for conversion of datetime index
        # to datetime
        buy_dates = list(buy_trades.index.to_pydatetime())
        sell_dates = list(sell_trades.index.to_pydatetime())

        ymin, ymax = ax.get_ylim()

        # Plotting blue vertical lines to indicate LONG (BUY) entry points
        for d in buy_dates:
            ax.axvline(d, ymin=0, ymax=ymax, color='blue')

        # Plotting black vertical lines to indicate SHORT (SELL) entry points
        for d in sell_dates:
            ax.axvline(d, ymin=0, ymax=ymax, color='black')

        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Values')
        plt.title('Manual Strategy vs Benchmark - out-of-sample')
        plt.legend()

        if saveFig:
            fig2 = plt.gcf()
            fig2.savefig('Figure2.png', format='png', dpi=120)
            plt.close(fig2)
        else:
            plt.show()


if __name__ == "__main__":
    # Create an instance of Manual Strategy
    rule_based = ManualStrategy()
    rule_based.test_code()
    print()





























