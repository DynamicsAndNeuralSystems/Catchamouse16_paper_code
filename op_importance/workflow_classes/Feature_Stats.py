import modules.feature_importance.PK_test_stats as fistat
import numpy as np
from sklearn import tree, svm
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import make_scorer
import random
import copy
import scipy.stats
import collections

from pathos.multiprocessing import ThreadPool as Pool
from tqdm import tqdm
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

    def __init__(self, null_pattern):
        """
        Constructor
        """
        self.null_pattern = null_pattern

        Feature_Stats.__init__(self, False)

    def calc_tots(self, labels, data, task_name):
        print "Decision tree: true calculations - labels have not been shuffled"
        clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=23)
        op_error_rates, mean_error_rates = train_model_template(labels, data, clf)

        '''
        null_stats = np.loadtxt(self.null_pattern.format(task_name))
        num_ops = np.size(mean_error_rates)
        if np.size(null_stats,0) != num_ops:
            raise Exception('Number of operations do not match number of rows in null stats file')

        num_null_reps = np.size(null_stats,1)
        p_vals = np.empty(num_ops)
        for i in range(num_ops):
            null_dist = null_stats[i,:]
            p_vals[i] = scipy.stats.norm(np.mean(null_dist), np.std(null_dist)).cdf(mean_error_rates[i])
            #num_null_more_accurate = (null_dist < mean_error_rates[i]).sum()
            #p_vals[i] = np.float(num_null_more_accurate) / np.float(num_null_reps)'''
        p_vals = np.zeros( mean_error_rates.shape )
        print "finished decision tree"
        return (op_error_rates, mean_error_rates, p_vals)

    def get_null_stats(self, task_name):
        return np.loadtxt(self.null_pattern.format(task_name))



class Null_Decision_Tree(Feature_Stats):

    def __init__(self):
        """
        Constructor
        """
        Feature_Stats.__init__(self, False)

    def calc_tots(self, labels, data, task_name):
        clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=23)
        op_error_rates, mean_error_rates = calc_null_template(labels,data,clf)
        p_vals = np.array([]) # np.empty([])
        return (op_error_rates, mean_error_rates, p_vals)

class Linear_Classifier(Feature_Stats):

    def __init__(self):
        """
        Constructor
        """
        Feature_Stats.__init__(self, False)

    def calc_tots(self, labels, data, task_name):
        print "SVM: true calculations - labels have not been shuffled"
        clf = svm.SVC(class_weight="balanced", decision_function_shape='ovo', kernel='linear', random_state=23)
        op_error_rates, mean_error_rates = train_model_template(labels, data, clf)
        p_vals = np.zeros( mean_error_rates.shape )
        return (op_error_rates, mean_error_rates, p_vals)

class Null_Linear_Classifier(Feature_Stats):

    def __init__(self):
        """
        Constructor
        """
        Feature_Stats.__init__(self, False)

    def calc_tots(self, labels, data, task_name):
        clf = svm.SVC(class_weight="balanced", decision_function_shape='ovo', kernel='linear', random_state=23)
        op_error_rates, mean_error_rates = calc_null_template(labels,data,clf)
        
        p_vals = np.empty([])
        return (op_error_rates, mean_error_rates, p_vals)


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
        # total number of train and test splits ~ 1000
        null_reps = 100*folds 
        op_errs = np.zeros(null_reps)
        operation = np.float32(data[:,i])
        # Reshape data as we have only one feature at a time
        operation = operation.reshape(-1, 1)
        rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=100, random_state=23)
        j = 0
        for trainIndices, testIndices in rskf.split(operation,labels):
            trainIndices = trainIndices.tolist()
            testIndices = testIndices.tolist()

            op_train, op_test = operation[trainIndices,:], operation[testIndices,:]
            labels_train, labels_test = labels[trainIndices], labels[testIndices]

            # Reshape train and test data as we have only one feature at a time
            op_train = op_train.reshape(-1, 1)
            op_test = op_test.reshape(-1, 1)   
    
            try:
                use_clf = copy.deepcopy(clf)
                use_clf = use_clf.fit(op_train, labels_train)
            except ValueError as e:
                print("ValueError while doing 10 fold CV and training. Labels are: ")
                print(np.unique(labels_train))
                print("Error message:", e)
                exit()
            labels_test_predicted = use_clf.predict(op_test)
            accuracy = balanced_accuracy_score(labels_test, labels_test_predicted)
            op_errs[j] = 1 - accuracy
            j += 1
        return op_errs

    random.seed(25)
    pool = Pool(processes=8)
    error_rates = pool.map(process_task_threaded, range(data.shape[1]))
    op_error_rates = np.vstack(error_rates)
    mean_error_rates = np.mean(op_error_rates, axis=1)
    print "Mean classification error is {}".format(np.mean(mean_error_rates))

    return (op_error_rates, mean_error_rates)


def accuracy_score_class_balanced(y_true, y_pred):
    class_counts = collections.Counter(y_true)
    weights = np.zeros(np.size(y_true))
    for this_class in class_counts.keys():
        weights[np.array(y_true)==this_class] = 1./class_counts[this_class]
    return accuracy_score(y_true, y_pred, sample_weight=weights)


def train_model_template(labels,data,clf):
    # Find maximum allowed folds for cross validation
    un, counts = np.unique(labels, return_counts=True)
    max_folds = 10
    min_folds = 2
    folds = np.min([max_folds, np.max([min_folds, np.min(counts)])])
    # print un
    print "Calculating classification error, {} fold cross validation, {} classes, {} samples".format(
        folds, len(un), len(labels))

    # Loop through each operation in a threaded manner
    def process_task_threaded(i):
        # data is type float64 by default but Decision tree classifier works with float32
        operation = np.float32(data[:, i])
        # Reshape data as we have only one feature at a time
        operation = operation.reshape(-1, 1)
        # # Find accuracy of classifier using cross validation
        # scores = cross_val_score(clf, operation, labels, cv=folds, n_jobs=1)
        # todo: this is a new score that corrects for class imbalances
        scorer = make_scorer(accuracy_score_class_balanced)
        # Find accuracy of classifier using cross validation
        scores = cross_val_score(clf, operation, labels, scoring=scorer, cv=folds, n_jobs=1)
        return 1 - scores

    pool = Pool(processes=8)
    # error_rates = pool.map(process_task_threaded, range(data.shape[1]))


    error_rates=[]
    for i in tqdm(range(0,data.shape[1])):
        error_rates.append(process_task_threaded(i))

    # with tqdm(total=data.shape[1]) as pbar:
    #     results = []
        

    #     for result in pool.imap(process_task_threaded, range(data.shape[1])):
    #         results.append(result)
    #         pbar.update(1)


    op_error_rates = np.vstack(error_rates)
    mean_error_rates = np.mean(op_error_rates, axis=1)
    print "Mean classification error is {}".format(np.mean(mean_error_rates))

    return (op_error_rates, mean_error_rates)
