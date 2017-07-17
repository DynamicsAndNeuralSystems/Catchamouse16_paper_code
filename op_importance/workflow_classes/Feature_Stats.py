import modules.feature_importance.PK_test_stats as fistat
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold

from pathos.multiprocessing import ThreadPool as Pool

class Feature_Stats:
    
    def __init__(self, is_pairwise_stat = False, combine_pair_method = 'mean'):
        """
        Constructor
        Parameters:
        -----------
        is_pairwise_stat : bool
            Boolean whether stat is computed in a pairwise manner or directly for each operation
        combine_pair_method : str
            String describing the method used to combine the pairwise calculated statistics (if applicable)
        Returns:
        --------
        None
        """
        self.is_pairwise_stat = is_pairwise_stat

        if combine_pair_method == 'mean':
            self.combine_pair = self.combine_pair_stats_mean
        else:
            self.combine_pair = self.combine_pair_not
   
    def combine_pair_not(self,pair_stats):
        """
        Pairs are not combined either because the statistics used does not require combination or because combination 
        is not desires.
        Parameters:
        -----------
        pair_stats : ndarray
            Array containing the stats for each label pair (row) for all operations (columns)
        Returns:
        --------
        pair_stats: ndarray
            Unchanged input
        """       
        return pair_stats
    
    
    def combine_pair_stats_mean(self,pair_stats):
        """
        Combine the stats of the label pairs to one pooled stat by taking the average along the columns of
        pair_stats (operations)
        Parameters:
        -----------
        pair_stats : ndarray
            Array containing the stats for each label pair (row) for all operations (columns)
        Returns:
        --------
        combined_stats:
            Average over all label pairs for each feature
        """
        # -- combine the stas for all labels pairs by taking the mean
        return np.ma.average(pair_stats,axis=0)

class Decision_Tree(Feature_Stats):

    def __init__(self):
        """
        Constructor
        """
        Feature_Stats.__init__(self, False)

    def calc_tots(self, labels, data):
        """
        Calculate the decision tree accuracy for each operation
        Parameters:
        -----------
        labels : ndarray
            1-D array containing the labels for each row in data.
        data : ndarray
            Array containing the data. Each row corresponds to a timeseries and each column to an operation.
        Returns:
        --------
        ranks : ndarray
            Returns the decision tree accuracy for each operation.
        """

        # Find maximum allowed folds for cross validation
        un, counts = np.unique(labels, return_counts=True)
        max_folds = 10
        min_folds = 2
        folds = np.min([max_folds, np.max([min_folds, np.min(counts)])])
        print "Calculating decision tree classification error, {} fold cross validation, {} classes, {} samples".format(folds, len(un), len(labels))

        # Loop through each operation in a threaded manner
        def process_task_threaded(i):
            # data is type float64 by default but Decision
            operation = np.float32(data[:,i])
            # Use decision tree classifier
            clf = tree.DecisionTreeClassifier()
            # Reshape data as we have only one feature at a time
            operation = operation.reshape(-1, 1)
            # Find accuracy of classifier using cross validation
            scores = cross_val_score(clf, operation, labels, cv=folds)
            return 1 - np.mean(scores)

        pool = Pool()
        error_rate_list = pool.map(process_task_threaded,range(data.shape[1]))
        op_error_rate = np.asarray(error_rate_list)

        '''
        # Loop through each operation
        for i, operation in enumerate(data.T):
            # Use decision tree classifier
            clf = tree.DecisionTreeClassifier()
            # Reshape data as we have only one feature at a time
            operation = operation.reshape(-1, 1)
            # Find accuracy of classifier using cross validation
            scores = cross_val_score(clf, operation, labels, cv=folds)
            op_error_rate[i] = 1 - np.mean(scores)
        '''

        print "Min decision tree classification error is {} ({} labels)".format(np.min(op_error_rate), len(un))
        return op_error_rate


class U_Stats(Feature_Stats):
    def __init__(self, combine_pair_method = 'mean'):
        """
        Constructor
        Parameters:
        -----------
        combine_pair_method : str
            String describing the method used to combine the pairwise calculated statistics (if applicable)
        Returns:
        --------
        None
        """
        Feature_Stats.__init__(self, True, combine_pair_method = 'mean')
    
    def calc_pairs(self,labels,data):
        """
        Calculate the ustatistic for each operation and every label pairing
        Parameters:
        -----------
        labels : ndarray
            1-D array containing the labels for each row in data.
        data : ndarray
            Array containing the data. Each row corresponds to a timeseries and each column to an operation.
        Returns:
        --------
        ranks : ndarray
            Returns the scaled U statistic for each label pairing and each operation.
    
        """
        ranks,ustat_norm = fistat.u_stat_all_label(data,labels=labels)[0:2]
        return ranks/ustat_norm[:,np.newaxis]
