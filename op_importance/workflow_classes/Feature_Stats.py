import modules.feature_importance.PK_test_stats as fistat
import numpy as np
from sklearn import tree, svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import random
import copy

from pathos.multiprocessing import ThreadPool as Pool

class Feature_Stats:
    
    def __init__(self, is_pairwise_stat = False, combine_pair_method = 'mean'):

        self.is_pairwise_stat = is_pairwise_stat

        if combine_pair_method == 'mean':
            self.combine_pair = self.combine_pair_stats_mean
        else:
            self.combine_pair = self.combine_pair_not
   
    def combine_pair_not(self,pair_stats):
        return pair_stats
    
    
    def combine_pair_stats_mean(self,pair_stats):
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
        clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=23)
        return train_model_template(labels, data, clf)


class Null_Decision_Tree(Feature_Stats):

    def __init__(self):
        """
        Constructor
        """
        Feature_Stats.__init__(self, False)

    def calc_tots(self, labels, data):
        clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=23)
        return calc_null_template(labels,data,clf)

class Linear_Classifier(Feature_Stats):

    def __init__(self):
        """
        Constructor
        """
        Feature_Stats.__init__(self, False)

    def calc_tots(self, labels, data):
        print "SVM: true calculations - labels have not been shuffled"
        clf = svm.SVC(class_weight="balanced", decision_function_shape='ovo', kernel='linear', random_state=23)
        return train_model_template(labels, data, clf)


class Null_Linear_Classifier(Feature_Stats):

    def __init__(self):
        """
        Constructor
        """
        Feature_Stats.__init__(self, False)

    def calc_tots(self, labels, data):
        clf = svm.SVC(class_weight="balanced", decision_function_shape='ovo', kernel='linear', random_state=23)
        return calc_null_template(labels,data,clf)


class U_Stats(Feature_Stats):
    def __init__(self, combine_pair_method = 'mean'):
        Feature_Stats.__init__(self, True, combine_pair_method = 'mean')
    
    def calc_pairs(self,labels,data):
        ranks,ustat_norm = fistat.u_stat_all_label(data,labels=labels)[0:2]
        return ranks/ustat_norm[:,np.newaxis]

def calc_null_template(labels,data,clf):
    # Find maximum allowed folds for cross validation
    un, counts = np.unique(labels, return_counts=True)
    max_folds = 10
    min_folds = 2
    folds = np.min([max_folds, np.max([min_folds, np.min(counts)])])
    print "Calculating null classification error, {} classes, {} samples".format(
        len(un), len(labels))

    # Loop through each operation in a threaded manner
    def process_task_threaded(i):
        null_reps = 1000
        op_errs = np.zeros(null_reps)
        for j in range(null_reps):
            # Shuffle labels
            random.shuffle(labels)
            # data is type float64 by default but Decision tree classifier works with float32
            operation = np.float32(data[:, i])
            # Reshape data as we have only one feature at a time
            operation = operation.reshape(-1, 1)
            # Split into training and test data
            t_size = 1/float(folds)
            op_train, op_test, labels_train, labels_test = train_test_split(operation, labels, test_size=t_size, random_state=23)
            op_train = op_train.reshape(-1, 1)
            op_test = op_test.reshape(-1, 1)
            # Fit classifier on training data
            use_clf = copy.deepcopy(clf)
            use_clf = use_clf.fit(op_train, labels_train)
            # Calculate accuracy on test data
            labels_test_predicted = use_clf.predict(op_test)
            op_errs[j] = 1 - accuracy_score(labels_test, labels_test_predicted)

        return op_errs

    random.seed(25)
    pool = Pool(processes=8)
    error_rates = pool.map(process_task_threaded, range(data.shape[1]))
    op_error_rates = np.vstack(error_rates)
    mean_error_rates = np.mean(op_error_rates, axis=1)
    print "Mean classification error is {}".format(np.mean(mean_error_rates))

    return (op_error_rates, mean_error_rates)

def train_model_template(labels,data,clf):
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
        # Reshape data as we have only one feature at a time
        operation = operation.reshape(-1, 1)
        # Find accuracy of classifier using cross validation
        scores = cross_val_score(clf, operation, labels, cv=folds, n_jobs=1)
        return 1 - scores

    pool = Pool(processes=8)
    error_rates = pool.map(process_task_threaded, range(data.shape[1]))
    op_error_rates = np.vstack(error_rates)
    mean_error_rates = np.mean(op_error_rates, axis=1)
    print "Mean classification error is {}".format(np.mean(mean_error_rates))

    return (op_error_rates, mean_error_rates)