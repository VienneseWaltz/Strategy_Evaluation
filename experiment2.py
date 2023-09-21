""""""
""" 
Implementing Experiment 2 of Project 8: Strategy Evaluation. This code shows how changing the values of 
market impact affects in-sample trading behavior. 

Student Name: Stella Soh  		  	   		   	 			  		 			 	 	 		 		 	
GT User ID: lsoh3 		  	   		   	 			  		 			 	 	 		 		 	
GT ID: 903641298 
"""
import pandas as pd
import datetime as dt
import StrategyLearner as sl
from indicators import *
from marketsimcode import *
from util import get_data, plot_data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def author(self):
    return 'lsoh3'

def test_code(saveFig=True):
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    dates = pd.date_range(sd, ed)

    symbol = 'JPM'
    prices = get_data([symbol], dates, addSPY=False).dropna()
    prices.fillna(method='ffill')
    prices.fillna(method='bfill')

    sv = 100000

    # The 3 Strategy Learners with different impact values
    sl_exp1 = sl.StrategyLearner(commission=0.0, impact=0.0025)
    sl_exp2 = sl.StrategyLearner(commission=0.0, impact=0.005)
    sl_exp3 = sl.StrategyLearner(commission=0.0, impact=0.05)


    # Training the 3 Strategy Learners
    sl_exp1.add_evidence(symbol, sd, ed, sv)
    sl_exp2.add_evidence(symbol, sd, ed, sv)
    sl_exp3.add_evidence(symbol, sd, ed, sv)


    # Testing the 3 Strategy Learners
    sl_trading_exp1 = sl_exp1.testPolicy(symbol, sd, ed, sv)
    sl_trading_exp2 = sl_exp2.testPolicy(symbol, sd, ed, sv)
    sl_trading_exp3 = sl_exp3.testPolicy(symbol, sd, ed, sv)


    # Portfolio Values
    sl_exp1_portvals = compute_portvals(sl_trading_exp1)
    sl_exp2_portvals = compute_portvals(sl_trading_exp2)
    sl_exp3_portvals = compute_portvals(sl_trading_exp3)


    # Portfolio Statistics
    sl_exp1_cum_ret, sl_exp1_adr, sl_exp1_std, sl_exp1_sharpe_ratio = cal_port_stats(sl_exp1_portvals)
    sl_exp2_cum_ret, sl_exp2_adr, sl_exp2_std, sl_exp2_sharpe_ratio = cal_port_stats(sl_exp2_portvals)
    sl_exp3_cum_ret, sl_exp3_adr, sl_exp3_std, sl_exp3_sharpe_ratio = cal_port_stats(sl_exp3_portvals)

    #######################
    # Print Portfolio Statistics
    #######################
    print(f"Cumulative Return of Strategy Learner 1: {'%.4f' % sl_exp1_cum_ret}")
    print(f"Cumulative Return of Strategy Learner 2: {'%.4f' % sl_exp2_cum_ret}")
    print(f"Cumulative Return of Strategy Learner 3: {'%.4f' % sl_exp3_cum_ret}")
    print()
    print(f"Standard Deviation of Strategy Learner 1: {'%.4f' % sl_exp1_std}")
    print(f"Standard Deviation of Strategy Learner 2: {'%.4f' % sl_exp2_std}")
    print(f"Standard Deviation of Strategy Learner 3: {'%.4f' % sl_exp3_std}")
    print()
    print(f"Average Daily Return of Strategy Learner 1: {'%.4f' % sl_exp1_adr}")
    print(f"Average Daily Return of Strategy Learner 2: {'%.4f' % sl_exp2_adr}")
    print(f"Average Daily Return of Strategy Learner 3: {'%.4f' % sl_exp3_adr}")
    print()
    print(f"Final Portfolio Value of Strategy Learner 1: {sl_exp1_portvals[-1]}")
    print(f"Final Portfolio Value of Strategy Learner 2: {sl_exp2_portvals[-1]}")
    print(f"Final Portfolio Value of Strategy Learner 3: {sl_exp3_portvals[-1]}")
    print()

    # Insert the labels
    sl_exp1_portvals.to_frame('Strategy Learner 1')
    sl_exp2_portvals.to_frame('Strategy Learner 2')
    sl_exp3_portvals.to_frame('Strategy Learner 3')


    # Normalize - Divide by the first row of each portvals DataFrames
    sl_exp1_portvals /= sl_exp1_portvals.iloc[0]
    sl_exp2_portvals /= sl_exp2_portvals.iloc[0]
    sl_exp3_portvals /= sl_exp3_portvals.iloc[0]


    ########################################
    # Plot the charts
    #########################################
    ax = sl_exp1_portvals.plot(label='Impact: 0.0025', fontsize=12, color='red')
    sl_exp2_portvals.plot(ax=ax, color='green', label='Impact: 0.005')
    sl_exp3_portvals.plot(ax=ax, color='orange', label='Impact: 0.05')


    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Portfolio Values')
    plt.title('Different Impact Values on Strategy Learners', fontsize=14)
    plt.legend()

    if saveFig:
        fig3 = plt.gcf()
        fig3.savefig('Figure4.png', format='png', dpi=120)
        plt.close(fig3)
    else:
        plt.show()



if __name__ == '__main__':
    test_code()




