""""""
"""  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			

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
"""
import numpy as np
from scipy import stats


class BagLearner(object):
    def __init__(self, learner, kwargs, bags=20, boost=False, verbose=False):
        """
        Constructor for a Bag Learner

        Parameters
        ----------
        learner: A DTLearner, RTLearner or LinRegLearner
        kwargs : string
            keyword arguments that are passed on to the learner's constructor
        bags : int
            Number of learners to be trained using Bootstrap Aggregation
        boost: False, default
            If boost is True, then implement boosting(?) or print out info about the learner
        verbose : bool, optional
            If true, print out debug messages. The default is False.

        Returns
        -------
        An instance of a Bag Learner

        """
        self.learners = [learner(**kwargs) for i in range(bags)]
        self.bags = bags
        self.boost = boost
        self.verbose = verbose


    def author(self):
        """
        Auther string

        Returns
        -------
        string
            The GT username of the student.

        """

        return "lsoh3"  # Georgia Tech username


    def add_evidence(self, data_x, data_y):
        """
        Iterate over the number of learners and grab n' out of n items (a random sample)in the original dataset.
        In this case, we are told the training set contains n data items, and each bag contains n items as well.
        Therefore, here n' = n. Each bag is trained on a different subset of the data.

        Parameters
        ----------
        data_x : numpy.ndarray
            training data.
        data_y : numpy.ndarray
            labels.

        Returns
        -------
        None.

        """

        num_samples_original_dataset = data_x.shape[0]
        # 60% of the dataset is used as training data
        training_rows = int(0.6 * num_samples_original_dataset)
        for self.learner in self.learners:
            for i in range(training_rows):
                # Generate a random sample from np.range(num_samples_original_dataset) of size
                # num_samples_original_dataset
                index = np.random.choice(num_samples_original_dataset, num_samples_original_dataset)
                bag_x_sample = data_x[index]
                bag_y_sample = data_y[index]
            self.learner.add_evidence(np.array(bag_x_sample), np.array(bag_y_sample))



    def query(self, points):
            """
            Construct a list of predicted values given the model we built.

            Parameters
            ----------
            points : numpy.ndarray
                A numpy array with each row corresponding to a specific query. There are multiple rows in
                this numpy array.

            Returns
            -------
            result : numpy.ndarray
                The predicted result of the input data according to the trained model.

            """
            predicted_values = np.array([self.learner.query(points) for self.learner in self.learners])
            return stats.mode(predicted_values).mode[0]


if __name__ == '__main__':
    print(f'Bag Learner')









