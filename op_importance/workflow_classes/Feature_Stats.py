import modules.feature_importance.PK_test_stats as fistat
import numpy as np
from sklearn import tree, svm
from sklearn.model_selection import cross_val_score
import random

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
        print "Decision tree: true calculations - labels have not been shuffled"
        return train_decision_tree(labels, data)


class Null_Decision_Tree(Feature_Stats):

    def __init__(self):
        """
        Constructor
        """
        Feature_Stats.__init__(self, False)

    def calc_tots(self, labels, data):
        # Shuffle labels
        random.seed(25)
        random.shuffle(labels)
        print "Decision tree: null calculations - labels have been shuffled"
        return train_decision_tree(labels,data)


class Linear_Classifier(Feature_Stats):

    def __init__(self):
        """
        Constructor
        """
        Feature_Stats.__init__(self, False)

    def calc_tots(self, labels, data):
        print "SVM: true calculations - labels have not been shuffled"
        return train_svm(labels,data)


class Null_Linear_Classifier(Feature_Stats):

    def __init__(self):
        """
        Constructor
        """
        Feature_Stats.__init__(self, False)

    def calc_tots(self, labels, data):
        # Shuffle labels
        random.seed(25)
        random.shuffle(labels)
        print "SVM - labels have been shuffled"

        return train_svm(labels,data)

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


def train_svm(labels, data):
    clf = svm.SVC(class_weight="balanced", decision_function_shape='ovo', kernel='linear', random_state=23)
    return train_model_template(labels, data, clf)


def train_decision_tree(labels, data):
    clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=23)
    return train_model_template(labels, data, clf)


def train_model_template(labels,data,clf):
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
    print "Calculating classification error, {} fold cross validation, {} classes, {} samples".format(
        folds, len(un), len(labels))

    # Loop through each operation in a threaded manner
    def process_task_threaded(i):
        # data is type float64 by default but Decision tree classifier works with float32
        operation = np.float32(data[:, i])
        # Use decision tree classifier
        clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=23)
        # Reshape data as we have only one feature at a time
        operation = operation.reshape(-1, 1)
        # Find accuracy of classifier using cross validation
        scores = cross_val_score(clf, operation, labels, cv=folds)
        return 1 - scores

    pool = Pool()
    error_rates = pool.map(process_task_threaded, range(data.shape[1]))
    op_error_rates = np.vstack(error_rates)
    mean_error_rates = np.mean(op_error_rates, axis=1)
    print "Mean classification error is {}".format(np.mean(mean_error_rates))

    return (op_error_rates, mean_error_rates)

