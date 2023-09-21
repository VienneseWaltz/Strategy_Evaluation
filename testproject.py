import ManualStrategy as ms
import StrategyLearner as sl
import experiment1 as exp1
import experiment2 as exp2
import numpy as np


def main():
    # Setting random seed
    np.random.seed(903641298)

    # Invoke Manual Strategy
    rule_based = ms.ManualStrategy()
    rule_based.test_code()

    # Invoke Strategy Learner
    learner_based = sl.StrategyLearner()
    learner_based.test_code()

    # Invoke experiement1.py to run test_code
    print(f"\n************ Experiment 1 Portfolio Statistics ***********")
    exp1.test_code()

    # Invoke experiment2.py to run test_code
    print(f"\n************ Experiment 2 Portfolio Statistics ***********")
    exp2.test_code()

if __name__ == '__main__':
    main()

