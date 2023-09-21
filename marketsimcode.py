""""""  		  	   		   	 			  		 			 	 	 		 		 	
"""MC2-P1: Market simulator.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
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
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd  		  	   		   	 			  		 			 	 	 		 		 	
from util import get_data, plot_data


def compute_portvals(  		  	   		   	 			  		 			 	 	 		 		 	
    trades,
    start_val=1000000,  		  	   		   	 			  		 			 	 	 		 		 	
    commission=0.0,
    impact=0.0
):  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    Computes the portfolio values.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    :param orders_file: Path of the order file or the file object  		  	   		   	 			  		 			 	 	 		 		 	
    :type orders_file: str or file object  		  	   		   	 			  		 			 	 	 		 		 	
    :param start_val: The starting value of the portfolio  		  	   		   	 			  		 			 	 	 		 		 	
    :type start_val: int  		  	   		   	 			  		 			 	 	 		 		 	
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		   	 			  		 			 	 	 		 		 	
    :type commission: float  		  	   		   	 			  		 			 	 	 		 		 	
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		   	 			  		 			 	 	 		 		 	
    :type impact: float  		  	   		   	 			  		 			 	 	 		 		 	
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		   	 			  		 			 	 	 		 		 	
    :rtype: pandas.DataFrame  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    # this is the function the autograder will call to test your code  		  	   		   	 			  		 			 	 	 		 		 	
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		   	 			  		 			 	 	 		 		 	
    # code should work correctly with either input  		  	   		   	 			  		 			 	 	 		 		 	
    # TODO: Your code here

    ##########################################################
    # 1. Build the first dataframe, df_prices
    ##########################################################

    # Obtain the list of symbols from trades.columns
    symbols = list(trades.columns)

    # Start date to track
    start_date = min(trades.index)

    # End date to track
    end_date = max(trades.index)

    # Obtain a DatetimeIndex with the specified start_date and end_date
    dates = pd.date_range(start_date, end_date)

    # SPY is removed. prices_all displays the prices of the symbols for the stated time period.
    df_prices = get_data(symbols, dates, addSPY=False).dropna()

    # From Lesson 1: to work around gaps in a DataFrame, fill forward, then fill backward
    df_prices.fillna(method = 'ffill')
    df_prices.fillna(method = 'bfill')

    # Add in a 'Cash' column at the end of df_prices and populate with 1.0's all the way
    # down to end_date
    # df_prices.insert(len(symbols), 'Cash', [1.0]*df_prices.shape[0], True)
    df_prices.insert(df_prices.shape[1], 'Cash', [1.0] * df_prices.shape[0], True)

    ######################################################################
    # 2. Build and populate the second dataframe, df_trades
    ######################################################################

    # Make a copy of df_prices created above. This DataFrame represents the changes in the number of shares
    df_trades = trades.copy()

    # Add in a 'Cash' column at the end of df_trades and populate with 0.0's all the way
    # down to end_date
    df_trades.insert(df_trades.shape[1], 'Cash', [0.0]*df_trades.shape[0], True)

    # Sum all the rows of resultant product of df_trades and df_prices in DataFrame
    df_trades.iloc[:, -1] = -(df_trades.iloc[:, :-1] * df_prices.iloc[:,:]).sum(axis=1)



    #####################################################################
    # 3. Build and populate the third dataframe, df_holdings
    #####################################################################

    # Make a copy of df_trades created above
    df_holdings = df_trades.copy()

    # Handling the 'Cash' column on the first row
    df_holdings.iloc[0]['Cash'] += start_val

    # Iterate over the rows to find the cumulative sum in each equity column of df_holdings
    df_holdings = df_holdings.cumsum()


    ###############################################################
    # 4. Build and populate the fourth dataframe, df_values
    ################################################################

    # DataFrame df_values is a product of df_prices and df_holdings
    df_values = df_prices * df_holdings


    ##################################################################
    # 5. Build and populate the fifth dataframe, df_port_values
    ###################################################################

    # df_port_values = pd.DataFrame(index=df_holdings.index)
    df_port_values = df_values.sum(axis=1)

    portvals = df_port_values
    #print(f'The portfolio values = {portvals}')

    return portvals

def author():
    return 'lsoh3'


def cal_port_stats(portvals):

    # Cumulative returns
    cr = (portvals[-1]/portvals[0]) - 1

    # Daily returns
    dr = (portvals/portvals.shift(1)) - 1

    # Remove the top row
    dr = dr[1:]

    # Mean or average daily returns
    adr = dr.mean()

    # Standard deviation of daily returns
    sddr = dr.std()

    # Risk-free rate
    rfr = 0.0

    # Sharpe ratio
    sr = math.sqrt(252) * ((dr - rfr).mean() / sddr)

    return cr, adr, sddr, sr


def simulate_mkt(trades, benchmark_trades, start_val=1000000, commission=0.00, impact=0.00):
    """
    Function to simulate the market using trades and benchmark trades DataFrame
    :param trades: Trades of a stock symbol
    :type trades: DataFrame object
    :param benchmark_trades: Trades of the benchmark stock symbol
    :type benchmark_trades: DataFrame object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: A fee deducted for each trade you execute
    :type commission: float
    :param impact: The market impact or direction the stock price moves for each trade you execute
    :type impact: float
    :return:
    """
    # Process the portfolio values of the stock
    portvals = compute_portvals(trades, start_val=start_val, commission=commission, impact=impact)

    # Obtain cumulative returns, average daily returns, standard deviation for daily returns
    # and Sharpe ratio of the portfolio
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = cal_port_stats(portvals)

    # Benchmark trades obtained from get_benchmark_trades()
    # benchmark_trades = get_benchmark_trades(symbol, start_date, end_date, sv)

    # Process the benchmark values
    bm_portvals = compute_portvals(benchmark_trades, start_val=start_val, commission=commission, impact=impact)

    # Obtain the cumulative returns, average daily returns, standard deviation for daily returns
    # and Sharpe ratio for the benchmark
    bm_cum_ret, bm_avg_daily_ret, bm_std_daily_ret, bm_sharpe_ratio = cal_port_stats(bm_portvals)

    # Compare portfolio values stats against benchmark portfolio values stats. Display them to 4 decimal places.

    print(f"Cumulative Return of Fund: {'%.4f'%cum_ret}")
    print(f"Cumulative Return of Benchmark: {'%.4f'%bm_cum_ret}")
    print()
    print(f"Standard Deviation of Fund: {'%.4f'%std_daily_ret}")
    print(f"Standard Deviation of Benchmark : {'%.4f'%bm_std_daily_ret}")
    print()
    print(f"Average Daily Return of Fund: {'%.4f'%avg_daily_ret}")
    print(f"Average Daily Return of Benchmark : {'%.4f'%bm_avg_daily_ret}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")
    print(f"Final Benchmark Value: {bm_portvals[-1]}")
    print()

    # Insert the label 'Portfolio' and 'Benchmark' respectively to portvals and bm_portvals for the date range
    portvals.to_frame('Portfolio')
    bm_portvals.to_frame('Benchmark')

    # Plotting the value of the theoretically optimal portfolio (red line)
    # versus the benchmark (green line)
    plot_norm_portvals_vs_bm_portvals(portvals, bm_portvals)


def normalize_data(df):
    # Stock price returned as a normalized value as we divide by the first row of DataFrame
    return df/df.iloc[0]


def plot_norm_portvals_vs_bm_portvals(portvals, bm_portvals, saveFig=True, figname='Portfolio_vs_Benchmark.png'):
    '''
    :param portvals: Portfolio values
    :param bm_portvals: Benchmark portfolio values
    :param saveFig: Flag set to True to disable plt.show() and save the file to 'Portfolio_vs_Benchmark.png'
                    If set to False, "Portfolio_vs_Benchmark.png" will be displayed.
    :param figname: Filename to save charts to
    :return: Chart that displays the normalized portfolio values versus the benchmark values
    '''
    portvals = normalize_data(portvals)
    bm_portvals = normalize_data(bm_portvals)


    # Plot the normalized portfolio in red and the normalized benchmark values in green
    plt.figure(figsize=(12,6))
    plt.plot(portvals, label='Portfolio', color='red')
    plt.plot(bm_portvals, label='Benchmark', color='green')


    # Label x-axis 'Dates' and y-axis 'Normalized Prices'
    plt.xlabel('Dates', fontsize=12)
    plt.ylabel('Normalized Prices', fontsize=12)
    plt.legend()

    # Display the title
    plt.title('Normalized Portfolio versus Benchmark', fontsize=14)

    if saveFig:
        fig1 = plt.gcf()
        fig1.savefig(figname, format='png', dpi=120)
        plt.close(fig1)
    else:
        plt.show()