import numpy as np
import matplotlib as mpl
import os
import sys
import collections
import scipy.stats
import warnings
from pathos.multiprocessing import ThreadPool as Pool # ProcessingPool

mpl.use("TkAgg")
#mpl.plt.ion()

runtypes = ["svm_maxmin","dectree_maxmin","svm_maxmin_null","dectree_maxmin_null"]
linkage_methods = ['average','complete']
default_task_names = ["Left_CAMK_excitatory","Left_CAMK_PVCre","Left_CAMK_SHAM",
                    "Left_excitatory_PVCre","Left_excitatory_SHAM","Left_PVCre_SHAM",
                    "Right_CAMK_excitatory","Right_CAMK_PVCre","Right_CAMK_SHAM",
                    "Right_excitatory_PVCre","Right_excitatory_SHAM","Right_PVCre_SHAM"]
 # These are the default settings

PARAMS = {
            'runtype':          "svm_maxmin",
            'linkage_method':   'average',
            'task_names':       default_task_names,
            'n_good_perf_ops':  100, # intermediate number of good performers to cluster
            'compute_features': False, # False or True : compute classification accuracies?
            'max_dist_cluster': 0.2,# gamma in paper, maximum allowed correlation distance within a cluster
            'calculate_mat':    False,
            'complete_average_logic': 'plot'} # 'calculate' or 'plot'
if len(sys.argv)>1:
    if sys.argv[1] in runtypes:
        PARAMS['runtype'] = sys.argv[1]
    else:
        warnings.warn("{} is invalid runtype - defaultying to {}".format(sys.argv[1],PARAMS['runtype']),Warning)
if len(sys.argv)>2:
    if sys.argv[2] in linkage_methods:
        PARAMS['linkage_method'] = sys.argv[2]
    else:
        warnings.warn("{} is invalid linkage_method - defaultying to {}".format(sys.argv[2],PARAMS['linkage_method']),Warning)
if len(sys.argv)>3:
    PARAMS['task_names'] = sys.argv[3].split(",")

PARAMS['figure_dir'] = "svgs_%s_%s" %(PARAMS['runtype'], PARAMS['linkage_method'])
if not os.path.isdir(PARAMS['figure_dir']):
    os.mkdir(PARAMS['figure_dir'])



# normalisation of features as done in hctsa TS_normalize
if 'maxmin' in PARAMS['runtype']:
    datatype = 'maxmin'
elif 'scaledrobustsigmoid' in PARAMS['runtype']:
    datatype = 'scaledrobustsigmoid'
else:
    datatype = 'maxmin'
    PARAMS['runtype'] = PARAMS['runtype'] + '_maxmin'
    raise Warning('normalisation not specified! Using maxmin')



import Task
import Data_Input
import Feature_Stats
import Reducing_Redundancy
import Plotting

import modules.misc.PK_helper as hlp
import modules.feature_importance.PK_feat_array_proc as fap
import locations



# import statsmodels
# import statsmodels.sandbox.stats.multicomp


# To support Python2.7 functions from statsmodels, the reference link to the function is added below
# https://github.com/statsmodels/statsmodels/blob/master/statsmodels/stats/multitest.py#L65
def multipletests(pvals, alpha=0.05, method='hs', is_sorted=False,
                  returnsorted=False):
    """
    Test results and p-value correction for multiple tests
    Parameters
    ----------
    pvals : array_like, 1-d
        uncorrected p-values.   Must be 1-dimensional.
    alpha : float
        FWER, family-wise error rate, e.g. 0.1
    method : str
        Method used for testing and adjustment of pvalues. Can be either the
        full name or initial letters. Available methods are:
        - `bonferroni` : one-step correction
        - `sidak` : one-step correction
        - `holm-sidak` : step down method using Sidak adjustments
        - `holm` : step-down method using Bonferroni adjustments
        - `simes-hochberg` : step-up method  (independent)
        - `hommel` : closed method based on Simes tests (non-negative)
        - `fdr_bh` : Benjamini/Hochberg  (non-negative)
        - `fdr_by` : Benjamini/Yekutieli (negative)
        - `fdr_tsbh` : two stage fdr correction (non-negative)
        - `fdr_tsbky` : two stage fdr correction (non-negative)
    is_sorted : bool
        If False (default), the p_values will be sorted, but the corrected
        pvalues are in the original order. If True, then it assumed that the
        pvalues are already sorted in ascending order.
    returnsorted : bool
         not tested, return sorted p-values instead of original sequence
    Returns
    -------
    reject : ndarray, boolean
        true for hypothesis that can be rejected for given alpha
    pvals_corrected : ndarray
        p-values corrected for multiple tests
    alphacSidak: float
        corrected alpha for Sidak method
    alphacBonf: float
        corrected alpha for Bonferroni method
    Notes
    -----
    There may be API changes for this function in the future.
    Except for 'fdr_twostage', the p-value correction is independent of the
    alpha specified as argument. In these cases the corrected p-values
    can also be compared with a different alpha. In the case of 'fdr_twostage',
    the corrected p-values are specific to the given alpha, see
    ``fdrcorrection_twostage``.
    The 'fdr_gbs' procedure is not verified against another package, p-values
    are derived from scratch and are not derived in the reference. In Monte
    Carlo experiments the method worked correctly and maintained the false
    discovery rate.
    All procedures that are included, control FWER or FDR in the independent
    case, and most are robust in the positively correlated case.
    `fdr_gbs`: high power, fdr control for independent case and only small
    violation in positively correlated case
    **Timing**:
    Most of the time with large arrays is spent in `argsort`. When
    we want to calculate the p-value for several methods, then it is more
    efficient to presort the pvalues, and put the results back into the
    original order outside of the function.
    Method='hommel' is very slow for large arrays, since it requires the
    evaluation of n partitions, where n is the number of p-values.
    """
    import gc
    pvals = np.asarray(pvals)
    alphaf = alpha  # Notation ?

    if not is_sorted:
        sortind = np.argsort(pvals)
        pvals = np.take(pvals, sortind)

    ntests = len(pvals)
    alphacSidak = 1 - np.power((1. - alphaf), 1./ntests)
    alphacBonf = alphaf / float(ntests)
    if method.lower() in ['b', 'bonf', 'bonferroni']:
        reject = pvals <= alphacBonf
        pvals_corrected = pvals * float(ntests)

    elif method.lower() in ['s', 'sidak']:
        reject = pvals <= alphacSidak
        pvals_corrected = 1 - np.power((1. - pvals), ntests)

    elif method.lower() in ['hs', 'holm-sidak']:
        alphacSidak_all = 1 - np.power((1. - alphaf),
                                       1./np.arange(ntests, 0, -1))
        notreject = pvals > alphacSidak_all
        del alphacSidak_all

        nr_index = np.nonzero(notreject)[0]
        if nr_index.size == 0:
            # nonreject is empty, all rejected
            notrejectmin = len(pvals)
        else:
            notrejectmin = np.min(nr_index)
        notreject[notrejectmin:] = True
        reject = ~notreject
        del notreject

        pvals_corrected_raw = 1 - np.power((1. - pvals),
                                           np.arange(ntests, 0, -1))
        pvals_corrected = np.maximum.accumulate(pvals_corrected_raw)
        del pvals_corrected_raw

    elif method.lower() in ['h', 'holm']:
        notreject = pvals > alphaf / np.arange(ntests, 0, -1)
        nr_index = np.nonzero(notreject)[0]
        if nr_index.size == 0:
            # nonreject is empty, all rejected
            notrejectmin = len(pvals)
        else:
            notrejectmin = np.min(nr_index)
        notreject[notrejectmin:] = True
        reject = ~notreject
        pvals_corrected_raw = pvals * np.arange(ntests, 0, -1)
        pvals_corrected = np.maximum.accumulate(pvals_corrected_raw)
        del pvals_corrected_raw
        gc.collect()

    elif method.lower() in ['sh', 'simes-hochberg']:
        alphash = alphaf / np.arange(ntests, 0, -1)
        reject = pvals <= alphash
        rejind = np.nonzero(reject)
        if rejind[0].size > 0:
            rejectmax = np.max(np.nonzero(reject))
            reject[:rejectmax] = True
        pvals_corrected_raw = np.arange(ntests, 0, -1) * pvals
        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        del pvals_corrected_raw

    elif method.lower() in ['ho', 'hommel']:
        # we need a copy because we overwrite it in a loop
        a = pvals.copy()
        for m in range(ntests, 1, -1):
            cim = np.min(m * pvals[-m:] / np.arange(1,m+1.))
            a[-m:] = np.maximum(a[-m:], cim)
            a[:-m] = np.maximum(a[:-m], np.minimum(m * pvals[:-m], cim))
        pvals_corrected = a
        reject = a <= alphaf

    elif method.lower() in ['fdr_bh', 'fdr_i', 'fdr_p', 'fdri', 'fdrp']:
        # delegate, call with sorted pvals
        reject, pvals_corrected = fdrcorrection(pvals, alpha=alpha,
                                                 method='indep',
                                                 is_sorted=True)
    elif method.lower() in ['fdr_by', 'fdr_n', 'fdr_c', 'fdrn', 'fdrcorr']:
        # delegate, call with sorted pvals
        reject, pvals_corrected = fdrcorrection(pvals, alpha=alpha,
                                                 method='n',
                                                 is_sorted=True)
    elif method.lower() in ['fdr_tsbky', 'fdr_2sbky', 'fdr_twostage']:
        # delegate, call with sorted pvals
        reject, pvals_corrected = fdrcorrection_twostage(pvals, alpha=alpha,
                                                         method='bky',
                                                         is_sorted=True)[:2]
    elif method.lower() in ['fdr_tsbh', 'fdr_2sbh']:
        # delegate, call with sorted pvals
        reject, pvals_corrected = fdrcorrection_twostage(pvals, alpha=alpha,
                                                         method='bh',
                                                         is_sorted=True)[:2]

    elif method.lower() in ['fdr_gbs']:
        #adaptive stepdown in Gavrilov, Benjamini, Sarkar, Annals of Statistics 2009
##        notreject = pvals > alphaf / np.arange(ntests, 0, -1) #alphacSidak
##        notrejectmin = np.min(np.nonzero(notreject))
##        notreject[notrejectmin:] = True
##        reject = ~notreject

        ii = np.arange(1, ntests + 1)
        q = (ntests + 1. - ii)/ii * pvals / (1. - pvals)
        pvals_corrected_raw = np.maximum.accumulate(q) #up requirementd

        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        del pvals_corrected_raw
        reject = pvals_corrected <= alpha

    else:
        raise ValueError('method not recognized')

    if pvals_corrected is not None: #not necessary anymore
        pvals_corrected[pvals_corrected>1] = 1
    if is_sorted or returnsorted:
        return reject, pvals_corrected, alphacSidak, alphacBonf
    else:
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[sortind] = pvals_corrected
        del pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[sortind] = reject
        return reject_, pvals_corrected_, alphacSidak, alphacBonf


class Workflow:

    def __init__(self, task_names, input_method, stats_method, redundancy_method, combine_tasks_method = 'mean',
                 combine_tasks_norm = None,
                 select_good_perf_ops_method = 'sort_asc',
                 select_good_perf_ops_norm = 'zscore',
                 select_good_perf_ops_combination = 'mean',
                 n_good_perf_ops = None):
        """
        Constructor
        Parameters:
        -----------
        task_names : list of str
            A list of task names to be included in this workflow
        input_method : Data_Input
            The data input method used to read the data from disk.
        stat_method : Feature_Stats
            The mehtod used to calculate the statistics
        redundancy_method : Reducing_Redundancy
            The method used to reduce the redundancy in the well performing features
        combine_tasks_method : str
            The name describing the method used to combine the statistics for each task to create a single 1d arrray with
            a single entry for every operation
        combine_tasks_norm : str
            The name of the normalisation method applied to the stats of each task before the statistics for each task are combined
        select_good_perf_ops_method : str
            The name describing the method used to sort the operations so the best operations come first in the self.stats_good_perf_op_comb
             and self.good_perf_op_ids
        select_good_perf_ops_norm : str
            The name describing the norm used when combining all statistics for all tasks for each operations
        self.n_good_perf_op_ids : int, optional
            Maximum entries in self.stats_good_perf_op_comb and self.good_perf_op_ids. If None, all good operations are used.
        """
        self.task_names = task_names
        self.input_method = input_method
        self.stats_method = stats_method
        self.redundancy_method = redundancy_method
        self.combine_tasks_norm = combine_tasks_norm
        # normalise_array(data,axis,norm_type = 'zscore')
        if combine_tasks_method == 'mean':
            self.combine_tasks = self.combine_task_stats_mean

        if combine_tasks_norm == 'zscore':
            self.combine_task_norm_method = lambda y : fap.normalise_masked_array(y,axis = 1,norm_type = 'zscore')[0]
        else:
            # -- no normalisation - id-function
            self.combine_task_norm_method = lambda y : y

        if select_good_perf_ops_method == 'sort_asc':
            self.select_good_perf_ops = self.select_good_perf_ops_sort_asc
        self.select_good_perf_ops_norm = select_good_perf_ops_norm
        self.select_good_perf_ops_combination = select_good_perf_ops_combination

        self.n_good_perf_ops = n_good_perf_ops
        # -- list of Tasks for this workflow
        self.tasks = [Task.Task(task_name,self.input_method,self.stats_method) for task_name in task_names]

        # -- Counter array for number of problems calculated successfully for each operation

        # -- place holders
        self.good_op_ids = []
        self.stats_good_op = None
        self.pvals_good_op = None
        self.stats_good_op_comb = None
        self.stats_good_perf_op_comb = None
        self.good_perf_op_ids = None

    def calculate_stats(self, save_attribute_name, out_path_pattern, is_keep_data = False):
        """
        Calculate the statistics of the features for each task using the method given by stats_method
        """

        for task in self.tasks:
            task.calc_stats(is_keep_data = is_keep_data)
            task.save_attribute(save_attribute_name, out_path_pattern)

    def collect_stats_good_op_ids(self, leave_task):
        """
        Collect all combined stats for each task and take stats for good operations only
        """
        #stats_good_op_ma = np.empty((data.shape[0],np.array(self.good_op_ids).shape[0]))
        stats_good_op_tmp = []
        pvals_good_op_tmp = []
        #stats_good_op_ma[:] = np.NaN

        for taskIndex, task in enumerate(self.tasks):
            if taskIndex == leave_task:
                continue
            # -- create tmp array for good stats for current task. For sake of simplicity when dealing with different
            # dimensions of task.tot_stats we transpose stats_good_op_ma_tmp so row corresponds to feature temporarily
            if task.tot_stats.ndim > 1:
                stats_good_op_ma_tmp = np.empty((self.good_op_ids.shape[0],task.tot_stats.shape[0]))
                pvals_good_op_ma_tmp = np.empty((self.good_op_ids.shape[0],task.tot_stats.shape[0]))
            else:
                stats_good_op_ma_tmp = np.empty((self.good_op_ids.shape[0]))
                pvals_good_op_ma_tmp = np.empty((self.good_op_ids.shape[0]))
            stats_good_op_ma_tmp[:] = np.NaN
            pvals_good_op_ma_tmp[:] = np.NaN

            ind = hlp.ismember(task.op_ids,self.good_op_ids,is_return_masked_array = True,return_dtype = int)
            # -- it is position in task.op_ids and i is position in self.good_op_ids
            for it,i in enumerate(ind):
                if i is not np.ma.masked: # -- that means the entry in task.op_ids is also in self.good_op_ids
                    stats_good_op_ma_tmp[i] = task.tot_stats[it].T
                    pvals_good_op_ma_tmp[i] = task.tot_stats_p_vals[it].T
            # -- We return to the usual ordering: column equals feature
            stats_good_op_tmp.append(stats_good_op_ma_tmp.T)
            pvals_good_op_tmp.append(pvals_good_op_ma_tmp.T)
        self.stats_good_op = np.ma.masked_invalid(np.vstack(stats_good_op_tmp))
        self.pvals_good_op = np.ma.masked_invalid(np.vstack(pvals_good_op_tmp))


    def combine_task_stats_mean(self):
        """
        Combine the stats of all the tasks using the average over all tasks

        """
        self.stats_good_op_comb = self.combine_task_norm_method(self.stats_good_op).mean(axis=0)

    def combine_task_pvals(self, min_p=None):
        """
        Combine the p-values of all the tasks using the method of Fisher

        """

        self.pvals_good_op_comb = np.zeros(self.pvals_good_op.shape[1])
        for i, pval_this_task in enumerate(self.pvals_good_op.T):
            # print scipy.stats.combine_pvalues(pval_this_task[np.logical_not(np.isnan(pval_this_task))])
            # print np.mean(pval_this_task), np.std(pval_this_task)
            # for pval in pval_this_task:
            #     print '%1.5f' % pval

            pval_this_task_no_nan = pval_this_task[np.logical_not(np.isnan(pval_this_task))]

            # correct to minimum p-value to not run into numerical issues.
            if min_p:
                pval_this_task_no_nan[pval_this_task_no_nan < min_p] = min_p

            _, self.pvals_good_op_comb[i] = scipy.stats.combine_pvalues(pval_this_task_no_nan)


        print 'combined p-values using Fisher''s method'

    def correct_pvals_multiple_testing(self):
        """
        Apply Bonferroni-Holm correction on combined p-values to compensate for multiple hypothesis testing

        """
	stats_out = multipletests(self.pvals_good_op_comb, alpha=0.05, method='h')
        # pvals_corrected = stats_out[1]
        # print 'no corr: mean p-value %.6f, std p-value %.6f' % (np.mean(self.pvals_good_op_comb), np.std(self.pvals_good_op_comb))
        # print 'corr: mean p-value %.6f, std p-value %.6f' % (
        # np.mean(pvals_corrected), np.std(pvals_corrected))

        self.pvals_good_op_comb = stats_out[1]


    def init_redundancy_method_problem_space(self):
        """
        Initialises the redundancy_method with the required parameters depending on the redundancy_method.compare_space
        """
        if self.redundancy_method.compare_space == 'problem_stats':
            self.redundancy_method.set_parameters(self.stats_good_op,self.good_op_ids,self.good_perf_op_ids)

    def find_good_op_ids(self, threshold):
        """
        Find the features that have been successfully calculated for more then threshold problems.
        Parameters:
        -----------
        threshold : int
            Only keep features that have been calculated for at least threshold tasks.

        """
        # -- List of all op_ids for each task (with duplicates)
        op_ids_tasks = [item for task in self.tasks for item in task.op_ids.tolist()]

        [self.op_ids, uniqueInds] = np.unique(op_ids_tasks, return_index=True)

        # also find task names
        op_names_tasks = [item for task in self.tasks for item in task.op['code_string']]
        self.op_names = np.array(op_names_tasks)[uniqueInds]

        # and keywords
        op_keywords_tasks = [item for task in self.tasks for item in task.op['keywords']]
        self.op_keywords = np.array(op_keywords_tasks)[uniqueInds]

        c = collections.Counter(op_ids_tasks)
        for key in c.keys():
            if c[key] >= threshold:
                self.good_op_ids.append(key)
        self.good_op_ids = np.array(self.good_op_ids)
        self.good_op_names = np.array(self.op_names[np.isin(self.op_ids, self.good_op_ids)])
        self.good_op_keywords = np.array(self.op_keywords[np.isin(self.op_ids, self.good_op_ids)])

    def exclude_good_ops_keyword(self, excludeKeyword):
        """
        Exclude features that have the keyword `excludeKeyword'.
        Parameters:
        -----------
        excludeKeyword : string
            If an operations has this keyword, it will be removed.

        """
        includeIndicator = np.array([excludeKeyword not in item for item in self.good_op_keywords])

        self.good_op_ids = self.good_op_ids[includeIndicator]
        self.good_op_names = self.good_op_names[includeIndicator]
        self.good_op_keywords = self.good_op_keywords[includeIndicator]

    def exclude_ops_keyword_per_task(self, excludeKeyword):
        """
        Exclude features that have the keyword `excludeKeyword'.
        Parameters:
        -----------
        excludeKeyword : string
            If an operations has this keyword, it will be removed.

        """

        for task in self.tasks:

            includeIndicator = np.array([excludeKeyword not in item for item in task.keywords_op])

            task.data = np.array(task.data)[:,includeIndicator]
            task.op['code_string'] = np.array(task.op['code_string'])[includeIndicator]
            task.op['id'] = np.array(task.op['id'])[includeIndicator]
            task.op['keywords'] = np.array(task.op['keywords'])[includeIndicator]
            task.op['master_id'] = np.array(task.op['master_id'])[includeIndicator]
            task.op['name'] = np.array(task.op['name'])[includeIndicator]
            task.keywords_op = np.array(task.keywords_op)[includeIndicator]
            task.op_ids = np.array(task.op_ids)[includeIndicator]
            task.tot_stats = np.array(task.tot_stats)[includeIndicator]
            task.tot_stats_all_runs = np.array(task.tot_stats_all_runs)[includeIndicator,:]
            task.tot_stats_p_vals = np.array(task.tot_stats_p_vals)[includeIndicator]

    def list_bad_op_ids_and_tasks(self):
        """
        Which features fail for what tasks?

        """

        bad_op_dict = {}
        for task in self.tasks:

            op_name_array = np.array(task.op['code_string'])

            # operations that are present in this task, but were filtered out in others
            bad_op_ids = set(task.op_ids.tolist()) - set(self.good_op_ids)
            for bad_op_id in bad_op_ids:

                # find operation name
                bad_op_name = op_name_array[bad_op_id == task.op_ids][0]

                if bad_op_dict.has_key(bad_op_name):
                    bad_op_dict[bad_op_name].append(task.name)
                else:
                    bad_op_dict[bad_op_name] = [task.name]

        # write to file
        with open(locations.rootDir() + '/bad_operations_and_tasks_they_fail_on.txt', 'w') as f:
            for item in sorted(bad_op_dict.items(), key=lambda t: len(t[1]), reverse=True):
                f.write("%s: %i bad tasks\n" % (item[0], len(item[1])))
                for bad_task in item[1]:
                        f.write("%s," % bad_task)
                f.write("\n\n")

    def list_bad_and_non_significant_ops(self):
        """
        Which features fail and which don't classify significantly?

        """

        # failing operations
        bad_op_ids = set(self.op_ids) - set(self.good_op_ids)

        # non-significant operations (only on good ops, naturally)
        non_sign_op_ids = self.good_op_ids[self.pvals_good_op_comb > 0.05]

        # print non significant ops
        a = self.pvals_good_op_comb
        b = self.good_op_names

        sortedOpInds = np.lexsort((b, a))[::-1]  # Sort by a then by b

        insignificantOpIds = self.good_op_ids[
            np.logical_or(np.isnan(self.pvals_good_op_comb), self.pvals_good_op_comb > 0.05)]

        if not len(set(insignificantOpIds).intersection(self.good_perf_op_ids)) == 0:
            raise Warning('Insignificant operations in top op set! Strange!!')

        # sortedOpInds = np.argsort(self.pvals_good_op_comb)[::-1]
        ind = 0
        while np.isnan(self.pvals_good_op_comb[sortedOpInds[ind]]) or self.pvals_good_op_comb[sortedOpInds[ind]] > 0.05:
            print '%1.5f %i %s' % (self.pvals_good_op_comb[sortedOpInds[ind]], self.good_op_ids[sortedOpInds[ind]], self.good_op_names[sortedOpInds[ind]])
            ind += 1

        # display the non-significant operations as a matrix
        maxInd = ind-1

        f = mpl.pyplot.figure()
        mpl.pyplot.imshow(self.pvals_good_op[:, sortedOpInds[0:maxInd]], aspect='auto')
        mpl.pyplot.yticks(np.arange(len(self.task_names)),self.task_names, fontsize=4)
        xTicks = ["%i, %s" % (self.good_op_ids[i], self.good_op_names[i]) for i in sortedOpInds[0:maxInd]]
        mpl.pyplot.xticks(np.arange(maxInd), xTicks, fontsize=5, rotation='vertical')
        cb = mpl.pyplot.colorbar()
        cb.set_label('p-value')
        mpl.pyplot.title('insignificant operations')

        # plot again with top-plot
        f, axarr = mpl.pyplot.subplots(2,1, sharex=True)
        pvals_good_op_filled = np.ma.filled(self.pvals_good_op, np.nan)
        axarr[1].imshow(pvals_good_op_filled[:, sortedOpInds[0:maxInd]], aspect='auto')
        axarr[1].set_yticks(np.arange(len(self.task_names)))
        axarr[1].set_yticklabels(self.task_names, fontdict={'fontsize':4})
        axarr[1].set_xticks(np.arange(maxInd))
        axarr[1].set_xticklabels(xTicks, fontdict={'fontsize':4}, rotation='90')

        axarr[0].plot(np.nanmean(pvals_good_op_filled[:,sortedOpInds[0:maxInd]], axis=0), label='mean p-value')
        n_datasets = np.shape(pvals_good_op_filled)[0]
        axarr[0].plot(np.tile(np.arange(maxInd), n_datasets), pvals_good_op_filled[:, sortedOpInds[0:maxInd]].flatten(), 'o', markersize=2, label='single p-values')
        axarr[0].plot(self.pvals_good_op_comb[sortedOpInds[0:maxInd]], label='Fisher + Bonferroni-Holm')
        # axarr[0].plot(np.sum(np.isnan(pvals_good_op_filled[:,sortedOpInds[0:maxInd]]).astype(float)/len(self.tasks),0), label='nan share')
        axarr[0].legend()
        mpl.pyplot.show()


    def load_task_attribute(self,attribute_name,in_path_pattern):
        """
        Load an attribute for all tasks from separate files
        Parameters:
        -----------
        attribute_name : string
            The name of the attribute of Task to be saved
        out_path-pattern : string
            A string containing the pattern for the path pointing to the input file. Formatted as in_path_pattern.format(self.name,attribute_name)
        """
        for task in self.tasks:
            task.load_attribute(attribute_name,in_path_pattern)

    def read_data(self, is_read_feature_data=True, old_matlab=False):
        """
        Read the data for all tasks from disk using the method given by self.input_method
        Parameters:
        -----------
        is_read_feature_data : bool
            Is the feature data to be read
        """

        # def read_task_parallel(t):
        #     t.read_data(is_read_feature_data=is_read_feature_data, old_matlab=old_matlab)
        #
        # pool = Pool(processes=8)
        # pool.map(read_task_parallel, self.tasks)

        for task in self.tasks:
          task.read_data(is_read_feature_data=is_read_feature_data, old_matlab=old_matlab)

    def save_task_attribute(self, attribute_name, out_path_pattern):
        """
        Save an attribute of of all tasks to separate files
        Parameters:
        -----------
        attribute_name : string
            The name of the attribute of Task to be saved
        out_path-pattern : string
            A string containing the pattern for the path pointing to the output file. Formatted as out_path_pattern.format(self.name,attribute_name)
        """
        for task in self.tasks:
            task.save_attribute(attribute_name, out_path_pattern)

    def select_good_perf_ops_sort_asc(self, accuracyNorm=True):
        """
        Select a subset of well performing operations
        """

        # performance in the overall pipeline is error, not accuracy. If we want to norm accuracy, first switch to it
        if accuracyNorm:
            stats_good_op = 1 - self.stats_good_op
        else:
            stats_good_op = self.stats_good_op

        if self.select_good_perf_ops_norm in ['z-score','zscore'] :
            all_classes_good_norm = fap.normalise_masked_array(stats_good_op,axis = 1,norm_type = 'zscore')[0]

        elif self.select_good_perf_ops_norm == 'mean-norm':
            all_classes_good_mean = np.ma.masked_invalid(np.ma.mean(stats_good_op,axis = 1))
            all_classes_good_norm = (stats_good_op.T / all_classes_good_mean).T

        elif self.select_good_perf_ops_norm == 'median-diff':
            all_classes_good_norm = fap.normalise_masked_array(stats_good_op, axis=1, norm_type='median-diff')[0]
        else:
            all_classes_good_norm =  stats_good_op

        if self.select_good_perf_ops_combination == 'mean':
            all_classes_norm_comb = all_classes_good_norm.mean(axis=0)
        elif self.select_good_perf_ops_combination == 'pos_sum':
            all_classes_good_norm_pos = np.array(all_classes_good_norm)
            all_classes_good_norm_pos[np.isnan(all_classes_good_norm_pos)] = 0
            all_classes_good_norm_pos[all_classes_good_norm_pos > 0] = 0
            all_classes_norm_comb = all_classes_good_norm_pos.sum(axis=0)
        elif self.select_good_perf_ops_combination == 'max':
            all_classes_norm_comb = all_classes_good_norm.max(axis=0)
        elif self.select_good_perf_ops_combination == 'min':
            all_classes_norm_comb = all_classes_good_norm.max(axis=0)
        else:
            raise NameError('No valid performance combination identifier.')

        # back to error
        if accuracyNorm:
            all_classes_norm_comb = 1 - all_classes_norm_comb
            all_classes_good_norm = 1 - all_classes_good_norm

        # write back to central combined-variable (skipped the separate combine step)
        self.stats_good_op_comb = all_classes_norm_comb
        self.stats_good_op_norm = all_classes_good_norm

        # super top performers
        superIds = np.squeeze(np.where(all_classes_norm_comb < np.mean(all_classes_norm_comb) - 3*np.std(all_classes_norm_comb)))
        self.super_perf_op_ids = self.good_op_ids[superIds]
        # mpl.pyplot.imshow(np.ma.corrcoef(self.stats_good_op[:, superIds].T), cmap=mpl.pyplot.cm.get_cmap('jet', 10))
        # cb = mpl.pyplot.colorbar()
        # mpl.pyplot.clim(0, 1)
        # ax = mpl.pyplot.gca()
        # # ax.set_xticks([])
        # # ax.set_yticks([])
        # xTicks = ["%s" % self.good_op_names[i] for i in superIds]
        # mpl.pyplot.xticks(np.arange(len(superIds)), xTicks, fontsize=5, rotation='vertical')
        # mpl.pyplot.yticks(np.arange(len(superIds)), xTicks, fontsize=5)
        #
        # mpl.pyplot.show()

        sort_ind_tmp = np.argsort(all_classes_norm_comb)

        if self.n_good_perf_ops == None:
            self.stats_good_perf_op_comb  = self.stats_good_op_comb[sort_ind_tmp]
            self.good_perf_op_ids =  self.good_op_ids[sort_ind_tmp]
        else:
            self.stats_good_perf_op_comb  = self.stats_good_op_comb[sort_ind_tmp][:self.n_good_perf_ops]
            self.good_perf_op_ids =  self.good_op_ids[sort_ind_tmp][:self.n_good_perf_ops]

        # # save to file for plotting
        # accuracyErrorString = ('accuracy' if accuracyNorm else 'error')
        #
        # filenameSuffix = '_' + self.select_good_perf_ops_norm + '_' + \
        #            self.select_good_perf_ops_combination + '_' + accuracyErrorString + '.txt'
        #
        # np.savetxt(locations.rootDir() + '/performance_all_ops_tasks_normed' + filenameSuffix,
        #            1 - all_classes_good_norm)
        # np.savetxt(locations.rootDir() + '/performance_all_ops_tasks_raw' + filenameSuffix,
        #            self.stats_good_op)

    def select_good_perf_ops_sort_asc_input_params_to_file(self, norm='zscore', comb='mean', accuracyNorm=False):
        """
        Select a subset of well performing operations, write to file. This is not usually used by the workflow. Just for
        outputting the selected features for different settings and checking results.
        """

        # performance in the overall pipeline is error, not accuracy. If we want to norm accuracy, first switch to it
        if accuracyNorm:
            stats_good_op = 1 - self.stats_good_op
        else:
            stats_good_op = self.stats_good_op

        # normalise across all features for each task
        if norm in ['z-score', 'zscore']:
            all_classes_good_norm = fap.normalise_masked_array(stats_good_op, axis=1, norm_type='zscore')[0]

        elif norm == 'mean-norm':
            all_classes_good_mean = np.ma.masked_invalid(np.ma.mean(stats_good_op, axis=1))
            all_classes_good_norm = (stats_good_op.T / all_classes_good_mean).T

        elif norm == 'median-norm':
            all_classes_good_median = np.ma.masked_invalid(np.ma.median(self.stats_good_op, axis=1))
            all_classes_good_norm = (stats_good_op.T / all_classes_good_median).T

        elif norm == 'median-diff':
            all_classes_good_norm = fap.normalise_masked_array(stats_good_op, axis=1, norm_type='median-diff')[0]

        else:
            all_classes_good_norm = stats_good_op

        # combine tasks
        if comb == 'mean':
            stats_good_op_comb = all_classes_good_norm.mean(axis=0)
        elif comb == 'pos_sum':
            all_classes_good_norm_pos = np.array(all_classes_good_norm)
            all_classes_good_norm_pos[np.isnan(all_classes_good_norm_pos)] = 0
            all_classes_good_norm_pos[all_classes_good_norm_pos > 0] = 0 # it's error, so >0 -> bad
            stats_good_op_comb = all_classes_good_norm_pos.sum(axis=0)
        elif comb == 'max':
            stats_good_op_comb = all_classes_good_norm.max(axis=0)
        elif comb == 'min':
            stats_good_op_comb = all_classes_good_norm.max(axis=0)
        else:
            raise NameError('No valid performance combination identifier.')

        # back to error
        if accuracyNorm:
            stats_good_op_comb = 1 - stats_good_op_comb

        # sort combined
        sort_ind_tmp = np.argsort(stats_good_op_comb)

        # sort according to performance
        stats_good_perf_op_comb  = stats_good_op_comb[sort_ind_tmp]
        good_perf_op_ids =  self.good_op_ids[sort_ind_tmp]

        accuracyErrorString = ('accuracy' if accuracyNorm else 'error')

        # write to file
        filename = 'topTops_' + norm + '_' + comb  +  '_' + accuracyErrorString + '_788.txt'

        np.savetxt(locations.rootDir() + '/topOps/' + filename, np.column_stack((good_perf_op_ids, stats_good_op_comb)))

    def plot_perf_histograms(self, norm='zscore', comb='mean'):
        """
        Select a subset of well performing operations, write to file. This is not usually used by the workflow. Just for
        outputting the selected features for different settings and checking results.
        """

        # normalise across all features for each task
        if norm in ['z-score', 'zscore']:
            all_classes_good_norm = fap.normalise_masked_array(self.stats_good_op, axis=1, norm_type='zscore')[0]

        elif norm == 'mean-norm':
            all_classes_good_mean = np.ma.masked_invalid(np.ma.mean(self.stats_good_op, axis=1))
            all_classes_good_norm = (self.stats_good_op.T / all_classes_good_mean).T

        elif norm == 'median-diff':
            all_classes_good_norm = fap.normalise_masked_array(self.stats_good_op, axis=1, norm_type='median-diff')[0]

        else:
            all_classes_good_norm = self.stats_good_op

        # combine tasks
        if comb == 'mean':
            stats_good_op_comb = all_classes_good_norm.mean(axis=0)
        elif comb == 'pos_sum':
            all_classes_good_norm_pos = np.array(all_classes_good_norm)
            all_classes_good_norm_pos[np.isnan(all_classes_good_norm_pos)] = 0
            all_classes_good_norm_pos[all_classes_good_norm_pos > 0] = 0
            stats_good_op_comb = all_classes_good_norm_pos.sum(axis=0)
        elif comb == 'max':
            stats_good_op_comb = all_classes_good_norm.max(axis=0)
        elif comb == 'min':
            stats_good_op_comb = all_classes_good_norm.min(axis=0)
        else:
            raise NameError('No valid performance combination identifier.')

        f = mpl.pyplot.figure()
        mpl.pyplot.hist(stats_good_op_comb, 50)
        mpl.pyplot.xlabel('error')
        mpl.pyplot.ylabel('feature frequency')
        mpl.pyplot.title(norm + '_' + comb)
        mpl.pyplot.savefig(locations.rootDir() + '/results/figures/norm='+norm+' comb='+comb+'.png')
        mpl.pyplot.close(f)

    def mask_pvals_too_few_unique_outputs(self):

        # todo: this replaces the mask of the p-values for simplicity. Remove from workflow later!!

        masked_pvals_all_tasks = self.pvals_good_op

        good_pval_mask = np.ma.getmask(masked_pvals_all_tasks)

        for task_ind, task in enumerate(self.tasks):

            # load number of unique values per operations from file
            n_uniques = np.loadtxt(locations.rootDir() + '/uniqueValues/' + task.name + '.txt')

            # keep the ones of the good operations
            n_uniques_good = n_uniques[np.isin(task.op_ids, self.good_op_ids)];

            # we want features that give different values for every other time series, so how many ts are there?
            n_ts = len(task.labels)

            # mask the ones that don't produce a diverse output
            unique_mask = np.full(np.size(good_pval_mask,1), False)
            unique_mask[np.isin(self.good_op_ids, task.op_ids)] = n_uniques_good < 0.10*n_ts

            # join the masks
            good_pval_mask[task_ind,:] = np.logical_or(good_pval_mask[task_ind,:], unique_mask)

        self.pvals_good_op.mask = good_pval_mask
        self.pvals_good_op.data[good_pval_mask] = np.nan

    def mask_pvals_too_few_unique_nulls(self):

        # todo: this replaces the mask of the p-values for simplicity. Remove from workflow later!!

        masked_pvals_all_tasks = self.pvals_good_op

        good_pval_mask = np.ma.getmask(masked_pvals_all_tasks)

        for task_ind, task in enumerate(self.tasks):

            # load number of unique values per operations from file
            n_uniques = np.load(locations.rootDir() + '/results/intermediate_results_dectree_maxmin_unique_nulls_npy/task_' + task.name + '_tot_stats_all_runs.npy')

            # keep the ones of the good operations
            n_uniques_good = n_uniques[np.isin(task.op_ids, self.good_op_ids)];

            # mask the ones that don't produce a diverse output
            unique_mask = np.full(np.size(good_pval_mask,1), False)
            unique_mask[np.isin(self.good_op_ids, task.op_ids)] = n_uniques_good < 200

            # join the masks
            good_pval_mask[task_ind,:] = np.logical_or(good_pval_mask[task_ind,:], unique_mask)

        self.pvals_good_op.mask = good_pval_mask
        self.pvals_good_op.data[good_pval_mask] = np.nan

    def plot_null_distributions(self, p_min):

        from scipy import stats
        from matplotlib.pyplot import cm

        nullFigDir = locations.rootDir() + '/nullHists/'
        if not os.path.exists(nullFigDir):
            os.makedirs(nullFigDir)

        # get all null stats
        null_stats_all_tasks = []
        print 'loading null stats'
        for task_ind, task in enumerate(self.tasks):
            # null_stats_all_tasks.append(self.stats_method.get_null_stats(task.name))
            null_stats_all_tasks.append(np.load(locations.rootDir() + '/results/intermediate_results_dectree_maxmin_null_npy/task_'+task.name+'_tot_stats_all_runs.npy'))
            print 'null stats for task %s loaded. (%i/%i)' % (task.name, task_ind, len(self.tasks))

        # print non significant ops
        a = self.pvals_good_op_comb
        b = self.good_op_names

        sortedOpInds = np.lexsort((b, a))[::-1]  # Sort by a then by b

        ind = 0
        while np.isnan(self.pvals_good_op_comb[sortedOpInds[ind]]) or self.pvals_good_op_comb[sortedOpInds[ind]] >= p_min:

            if np.isnan(self.pvals_good_op_comb[sortedOpInds[ind]]):
                ind+=1
                continue

            pvals_all_tasks = self.pvals_good_op[:,sortedOpInds[ind]]
            pvals_all_tasks_filled = np.ma.filled(pvals_all_tasks, np.nan)

            # sortedTaskInds = np.argsort(pvals_all_tasks)[::-1]
            # for taskInd in sortedTaskInds:
            #     print '%1.5f %s' % (pvals_all_tasks[taskInd], self.task_names[taskInd])

            # different colors for p-values
            color = cm.jet(np.linspace(0, 1, 1000))

            for task_ind, task in enumerate(self.tasks):

                if np.isnan(pvals_all_tasks_filled[task_ind]):
                    continue

                f, ax = mpl.pyplot.subplots(1)

                # if pvals_all_tasks[task_ind] < 0.9:
                #     continue

                null_stats = null_stats_all_tasks[task_ind]

                null_stats_this_op = null_stats[task.op_ids == self.good_op_ids[sortedOpInds[ind]],:]

                pcolorval = pvals_all_tasks_filled[task_ind]*999;
                if np.isnan(pcolorval):
                    c = [0.5, 0.5, 0.5]
                else:
                    c = color[int(pcolorval),:]
                mpl.pyplot.hist(np.squeeze(null_stats_this_op), 100, color=c, label="%s, p=%1.3f, N=%i" % (task.name, pvals_all_tasks_filled[task_ind],len(task.labels)))

                # density = stats.kde.gaussian_kde(null_stats_good_ops[~np.isnan(null_stats_good_ops)])
                # x = np.arange(0., 1, .001)
                # mpl.pyplot.plot(x, density(x), color=c)

                stats_this_op = self.stats_good_op[task_ind, sortedOpInds[ind]]
                ax.axvline(stats_this_op, linestyle=':', color=c)

                # mpl.pyplot.legend()
                mpl.pyplot.xlabel('error')
                mpl.pyplot.ylabel('frequency')


                fileName = '%03i, %i, %s (p=%1.3f), %s.png' % (
                    ind,
                    self.good_op_ids[sortedOpInds[ind]],
                    self.good_op_names[sortedOpInds[ind]],
                    self.pvals_good_op_comb[sortedOpInds[ind]],
                    task.name)

                titleString = '%s (p=%1.3f)\n%s, p=%1.3f, N=%i' % (
                    self.good_op_names[sortedOpInds[ind]],
                    self.pvals_good_op_comb[sortedOpInds[ind]],
                    task.name, pvals_all_tasks_filled[task_ind], len(task.labels))

                mpl.pyplot.title(titleString)
                # print 'hm'

                mpl.pyplot.savefig(os.path.join(nullFigDir, fileName))

                mpl.pyplot.close(f)

            ind += 1

    def kstest_null_distributions(self):

        from scipy import stats
        from matplotlib.pyplot import cm

        saveDir = locations.rootDir() + '/'

        # get all null stats
        null_stats_all_tasks = []
        print 'loading null stats'
        for task_ind, task in enumerate(self.tasks):
            # null_stats_all_tasks.append(self.stats_method.get_null_stats(task.name))
            null_stats_all_tasks.append(np.load(locations.rootDir() + '/results/intermediate_results_dectree_maxmin_null_npy/task_'+task.name+'_tot_stats_all_runs.npy'))
            print 'null stats for task %s loaded. (%i/%i)' % (task.name, task_ind, len(self.tasks))


        kstest_pvals = np.full((len(self.tasks), len(self.op_ids)), np.nan)

        for task_ind, task in enumerate(self.tasks):

            null_stats = np.array(null_stats_all_tasks[task_ind])

            print "now working on task " + task.name

            for op_ind, op_id in enumerate(self.op_ids):

                if op_ind%100==0:
                    print "op id %i/%i" %(op_ind, len(self.op_ids))

                op_id_indicator = task.op_ids == op_id

                if sum(op_id_indicator) > 0:

                    nullStatsTemp = null_stats[op_id_indicator, :]
                    nullStatsTempSqueezed = np.squeeze(nullStatsTemp)

                    mu = np.mean(nullStatsTempSqueezed)
                    std = np.std(nullStatsTempSqueezed)

                    nullStatsZscored = (nullStatsTempSqueezed - mu)/std

                    ksresult = stats.kstest(nullStatsZscored, 'norm')

                    kstest_pvals[task_ind,op_ind] = ksresult[1]


        np.savez(os.path.join(saveDir, 'KStest.npz'), kstest_pvals, self.task_names, self.op_ids, self.op_names, self.good_op_ids, self.good_perf_op_ids)

    def plot_one_null_distribution(self, op_ID, task_name):

        from scipy import stats
        from matplotlib.pyplot import cm

        # get null stats
        null_stats = self.stats_method.get_null_stats(task_name)

        good_op_ind = np.where(self.good_op_ids == op_ID)

        print "operation: %s" % self.good_op_names[good_op_ind]

        pvals_all_tasks = self.pvals_good_op[:, good_op_ind]
        pvals_all_tasks_filled = np.ma.filled(pvals_all_tasks, np.nan)

        task = self.tasks[self.task_names == task_name]
        null_stats_this_op = null_stats[task.op_ids == op_ID, :]

    def select_good_pval_ops_sort_asc(self):
        """
        Select a subset of well performing operations by p-value
        """

        all_classes_good_norm =  self.pvals_good_op_comb

        sort_ind_tmp = np.argsort(all_classes_good_norm)

        if self.n_good_perf_ops == None:
            self.pval_good_perf_op_comb  = self.pvals_good_op_comb[sort_ind_tmp]
            self.pval_good_perf_op_ids =  self.good_op_ids[sort_ind_tmp]
        else:
            self.pval_good_perf_op_comb  = self.pvals_good_op_comb[sort_ind_tmp][:self.n_good_perf_ops]
            self.pval_good_perf_op_ids =  self.good_op_ids[sort_ind_tmp][:self.n_good_perf_ops]

    def select_good_perf_cluster_center_ops(self):
        """
        Select a one best feature for each cluster
        """

        # if self.select_good_perf_ops_norm in ['z-score','zscore'] :
        #     all_classes_good_norm = fap.normalise_masked_array(self.stats_good_op,axis = 1,norm_type = 'zscore')[0]
        #
        # elif self.select_good_perf_ops_norm == 'mean-norm':
        #     all_classes_good_mean = np.ma.masked_invalid(np.ma.mean(self.stats_good_op,axis = 1))
        #     all_classes_good_norm = (self.stats_good_op.T / all_classes_good_mean).T
        #
        # else:
        #     all_classes_good_norm =  self.stats_good_op

        # # we don't want the normalisation to be the same as for ranking, just take the z-score
        # all_classes_good_norm = fap.normalise_masked_array(self.stats_good_op, axis=1, norm_type='zscore')[0]

        # sort operations once, then filter this index list to contain only indices of the cluster at hand
        sorted_op_ids = self.good_op_ids[np.argsort(self.stats_good_op_comb)] # all_classes_good_norm.mean(axis=0))

        cluster_center_op_ids = []
        for i, cluster_op_ids in enumerate(self.redundancy_method.cluster_op_id_list):

            cluster_center_op_ids.append(sorted_op_ids[np.isin(sorted_op_ids, cluster_op_ids)][0])

        self.good_perf_cluster_center_op_ids  = np.array(cluster_center_op_ids)


        print 'Cluster centers:'
        print cluster_center_op_ids
        print 'with names:'
        for id in cluster_center_op_ids:
            print '%s,' % (self.good_op_names[self.good_op_ids == id][0])

    def classify_good_perf_ops_vs_good_ops(self):

        import sklearn.tree as tree
        from sklearn.model_selection import cross_val_score
        import time

        # initialise tree
        clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=23)

        # load class balanced scorer
        from sklearn.metrics import make_scorer
        scorer = make_scorer(Feature_Stats.accuracy_score_class_balanced)

        # reference line (below other data)
        mpl.pyplot.plot((0, 1.5), (0, 1.5), '--', color=np.array((1, 1, 1)) * 0.7)


        perfmat = np.zeros((len(self.tasks), 6))
        for task_ind, task in enumerate(self.tasks):

            t = time.time()
            print 'classifying task %s' % task.name

            # decide on number of folds
            un, counts = np.unique(task.labels, return_counts=True)
            max_folds = 10
            min_folds = 2
            folds = np.min([max_folds, np.max([min_folds, np.min(counts)])])

            # -- do cross-validated scoring for full and reduced matrix

            # only good operations
            score_this_task_top_ops = cross_val_score(clf, task.data[:, np.isin(task.op_ids, self.good_perf_op_ids)], task.labels, cv=folds, scoring=scorer)

            # only cluster centers
            score_this_task_cluster_ops = cross_val_score(clf, task.data[:, np.isin(task.op_ids, self.good_perf_cluster_center_op_ids)],
                                                      task.labels, cv=folds, scoring=scorer)

            # whole matrix
            score_this_task_whole = cross_val_score(clf, task.data, task.labels, cv=folds, scoring=scorer)

            # # plot immediately
            # p1 = mpl.pyplot.errorbar(np.mean(score_this_task_whole), np.mean(score_this_task_top_ops),
            #                     xerr=np.std(score_this_task_whole), yerr=np.std(score_this_task_top_ops), fmt='o', color='b', ecolor='b')
            # p2 = mpl.pyplot.errorbar(np.mean(score_this_task_whole), np.mean(score_this_task_cluster_ops),
            #                     xerr=np.std(score_this_task_whole), yerr=np.std(score_this_task_cluster_ops), fmt='o',
            #                     color='r', ecolor='r')

            # save scores
            perfmat[task_ind, 0] = np.mean(score_this_task_whole)
            perfmat[task_ind, 1] = np.std(score_this_task_whole)
            perfmat[task_ind, 2] = np.mean(score_this_task_top_ops)
            perfmat[task_ind, 3] = np.std(score_this_task_top_ops)
            perfmat[task_ind, 4] = np.mean(score_this_task_cluster_ops)
            perfmat[task_ind, 5] = np.std(score_this_task_cluster_ops)

            print 'Done. Took %1.1f minutes.' % ((time.time() - t)/60)

        np.savetxt(locations.rootDir() + '/peformance_mat_fullMeanStd_topMeanStd_clusterMeanStd_new710.txt', perfmat)

        # mpl.pyplot.legend((p1, p2), ('500 top ops', 'only cluster centers'))
        # # mpl.pyplot.xlim((0, 1))
        # # mpl.pyplot.ylim((0, 1))
        # mpl.pyplot.xlabel('performance on whole feature set')
        # mpl.pyplot.ylabel('performance only selected features')
        # mpl.pyplot.ylabel('class imbalance corrected performance')
        # mpl.pyplot.show()

    def classify_selected_ops(self, opIdsSelect):

        import sklearn.tree as tree
        from sklearn.svm import LinearSVC
        from sklearn.model_selection import cross_val_score
        import time

        # initialise tree
        clf = LinearSVC(random_state=23) # tree.DecisionTreeClassifier(class_weight="balanced", random_state=23)

        # load class balanced scorer
        from sklearn.metrics import make_scorer
        scorer = make_scorer(Feature_Stats.accuracy_score_class_balanced)


        perfmat = np.zeros((len(self.tasks), 2))
        for task_ind, task in enumerate(self.tasks):

            t = time.time()
            print 'classifying task %s' % task.name

            # decide on number of folds
            un, counts = np.unique(task.labels, return_counts=True)
            max_folds = 10
            min_folds = 2
            folds = np.min([max_folds, np.max([min_folds, np.min(counts)])])

            # -- do cross-validated scoring for full and reduced matrix

            # only good operations
            score_this_task = cross_val_score(clf, task.data[:, np.isin(task.op_ids, opIdsSelect)], task.labels, cv=folds, scoring=scorer)

            # save scores
            perfmat[task_ind, 0] = np.mean(score_this_task)
            perfmat[task_ind, 1] = np.std(score_this_task)

            print 'Done. Took %1.1f minutes.' % ((time.time() - t)/60)

        np.savetxt(locations.rootDir() + '/peformance_canonical_linear.txt', perfmat)

    def classify_selected_ops_internalSet(self):

        import sklearn.tree as tree
        from sklearn.svm import LinearSVC
        from sklearn.model_selection import cross_val_score
        import time

        # initialise tree
        clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=23)

        # # feature names to filter for
        # featureNamesCatch22 = ['DN_HistogramMode_5',
        #                        'DN_HistogramMode_10',
        #                        'CO_f1ecac',
        #                        'CO_FirstMin_ac',
        #                        'CO_HistogramAMI_even_2.ami5',
        #                        'IN_AutoMutualInfoStats_40_gaussian.fmmi',
        #                        'CO_trev_1.num',
        #                        'SB_TransitionMatrix_3ac.sumdiagcov',
        #                        'PD_PeriodicityWang.th2',
        #                        'CO_Embed2_Dist_tau.d_expfit_meandiff',
        #                        'FC_LocalSimple_mean1.tauresrat',
        #                        'FC_LocalSimple_mean3.stderr',
        #                        'DN_OutlierInclude_p_001.mdrmd',
        #                        'DN_OutlierInclude_n_001.mdrmd',
        #                        'SB_BinaryStats_diff.longstretch0',
        #                        'SB_BinaryStats_mean.longstretch1',
        #                        'SB_MotifThree_quantile.hh',
        #                        'SC_FluctAnal_2_rsrangefit_50_1_logi.prop_r1',
        #                        'SC_FluctAnal_2_dfa_50_1_2_logi.prop_r1',
        #                        'SP_Summaries_welch_rect.centroid',
        #                        'SP_Summaries_welch_rect.area_5_1',
        #                        'MD_hrv_classic.pnn40']

        featureNamesCatch22 = ['CO_Embed2_Basic_tau.incircle_1',
                                'CO_Embed2_Basic_tau.incircle_2',
                                'FC_LocalSimple_mean1.taures',
                                'SY_SpreadRandomLocal_ac2_100.meantaul',
                                'DN_HistogramMode_10',
                                'SY_StdNthDer_1',
                                'AC_9',
                                'SB_MotifTwo_mean.hhh',
                                'EN_SampEn_5_03.sampen1',
                                'CO_FirstMin_ac',
                                'DN_OutlierInclude_abs_001.mdrmd',
                                'CO_trev_1.num',
                                'FC_LocalSimple_lfittau.taures',
                                'SY_SpreadRandomLocal_50_100.meantaul',
                                'SC_FluctAnal_2_rsrangefit_50_1_logi.prop_r1',
                                'PH_ForcePotential_sine_1_1_1.proppos',
                                'SP_Summaries_pgram_hamm.maxw',
                                'SP_Summaries_welch_rect.maxw']

        # catch22 feature indicator
        catch22Indicator = [item in featureNamesCatch22 for item in self.good_op_names];
        catch22IDs = [self.good_op_ids[i] for i in range(len(self.good_op_ids)) if
                      self.good_op_names[i] in featureNamesCatch22];

        # load class balanced scorer
        from sklearn.metrics import make_scorer
        scorer = make_scorer(Feature_Stats.accuracy_score_class_balanced)


        perfmat = np.zeros((len(self.tasks), 2))
        for task_ind, task in enumerate(self.tasks):

            t = time.time()
            print 'classifying task %s' % task.name

            # decide on number of folds
            un, counts = np.unique(task.labels, return_counts=True)
            max_folds = 10
            min_folds = 2
            folds = np.min([max_folds, np.max([min_folds, np.min(counts)])])

            # -- do cross-validated scoring for full and reduced matrix

            # only good operations
            score_this_task = cross_val_score(clf, task.data[:, np.isin(task.op_ids, catch22IDs)], task.labels, cv=folds, scoring=scorer)

            # save scores
            perfmat[task_ind, 0] = np.mean(score_this_task)
            perfmat[task_ind, 1] = np.std(score_this_task)

            print 'Done. Took %1.1f minutes.' % ((time.time() - t)/60)

        np.savetxt('/Users/carl/PycharmProjects/op_importance/peformance_sarab_nameSelect.txt', perfmat)

    def classify_random_features(self, nFeatures=22, nReps=100):

        import sklearn.tree as tree
        from sklearn.svm import LinearSVC
        from sklearn.model_selection import cross_val_score
        import time
        import random

        # initialise tree
        clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=23)

        # load class balanced scorer
        from sklearn.metrics import make_scorer
        scorer = make_scorer(Feature_Stats.accuracy_score_class_balanced)


        perfmat = np.zeros((len(self.tasks), 2))
        for task_ind, task in enumerate(self.tasks):

            t = time.time()
            print 'classifying task %s' % task.name

            # decide on number of folds
            un, counts = np.unique(task.labels, return_counts=True)
            max_folds = 10
            min_folds = 2
            folds = np.min([max_folds, np.max([min_folds, np.min(counts)])])

            # -- do cross-validated scoring for full and reduced matrix

            meanScores = np.full(nReps, fill_value=np.nan)
            for repInd in range(nReps):

                selectedIDs = random.sample(task.op_ids, nFeatures)

                # only good operations
                score_this_task = cross_val_score(clf, task.data[:, np.isin(task.op_ids, selectedIDs)], task.labels, cv=folds, scoring=scorer)

                meanScores[repInd] = np.mean(score_this_task)

            # save scores
            perfmat[task_ind, 0] = np.nanmean(meanScores)
            perfmat[task_ind, 1] = np.nanstd(meanScores)

            print 'Done. Took %1.1f minutes.' % ((time.time() - t)/60)

        np.savetxt('/Users/carl/PycharmProjects/op_importance/peformance_randsample_22_100.txt', perfmat)

    def classify_good_perf_ops_vs_super_vs_good_ops(self):

        CATCH16_PROPER =  [ 'SY_DriftingMean50.min',
                            'DN_RemovePoints_absclose_05.ac2rat',
                            'AC_nl_036',
                            'AC_nl_112',
                            'ST_LocalExtrema_n100.diffmaxabsmin',
                            'CO_TranslateShape_circle_35_pts.statav4_m',
                                                            'CO_TranslateShape_circle_35_pts.std',
                            'SC_FluctAnal_2_dfa_50_2_logi.r2_se2',
                            'IN_AutoMutualInfoStats_diff_20_gaussian.ami8',
                            'PH_Walker_momentum_5.w_propzcross',
                                                'PH_Walker_biasprop_05_01.sw_meanabsdiff',
                                                'FC_LoopLocalSimple_mean.stderr_chn',
                                                'CO_HistogramAMI_even_10.ami3',
                            'CO_HistogramAMI_even_2.ami3',
                                                'AC_nl_035',
                            'CO_AddNoise_1_even_10.ami_at_10']
        CATCH16_COMPLETELINKAGE = [  
                            'SY_DriftingMean50.min',
                            'DN_RemovePoints_absclose_05.ac2rat',
                            'AC_nl_036',
                            'AC_nl_112',
                            'ST_LocalExtrema_n100.diffmaxabsmin',
                            'CO_TranslateShape_circle_35_pts.statav4_m',
                                                            'MF_CompareAR_1_10_05_stddiff', 
                            # MISSING 'SC_FluctAnal_2_dfa_50_2_logi.r2_se2',
                            # MISSING 'IN_AutoMutualInfoStats_diff_20_gaussian.ami8',
                            'PH_Walker_momentum_5_w_momentumzcross', 
                                                            'MF_steps_ahead_arma_3_1_6_ac1_6',
                                   # MISSING   'FC_LoopLocalSimple_mean.stderr_chn',
                            'SP_Summaries_welch_rect_fpolysat_rmse',  'CO_HistogramAMI_even_10_3', # DOUBLING UP
                            'CO_HistogramAMI_even_2_3', 
                                                        'MF_StateSpace_n4sid_1_05_1_ac2', 
                            'CO_AddNoise_1_even_10_ami_at_10', 
                            #OTHER IDENTIFIED CENTROIDS        
                            'SC_FluctAnal_2_std_50_logi_ssr', 
                            'RM_ami_3', 
]

        if PARAMS['linkage_method']=='average':
            featureNamesCatch16 = CATCH16_PROPER
        elif PARAMS['linkage_method']=='complete':
            featureNamesCatch16 = CATCH16_COMPLETELINKAGE
        else:
            print("PARAMS Linkage Method set improperly. Defaulting to catchamouse16 set")
            featureNamesCatch16 = CATCH16_PROPER



        featureNamesCatch22 = ['CO_Embed2_Basic_tau.incircle_1',
                                'CO_Embed2_Basic_tau.incircle_2',
                                'FC_LocalSimple_mean1.taures',
                                'SY_SpreadRandomLocal_ac2_100.meantaul',
                                'DN_HistogramMode_10',
                                'SY_StdNthDer_1',
                                'AC_9',
                                'SB_MotifTwo_mean.hhh',
                                'EN_SampEn_5_03.sampen1',
                                'CO_FirstMin_ac',
                                'DN_OutlierInclude_abs_001.mdrmd',
                                'CO_trev_1.num',
                                'FC_LocalSimple_lfittau.taures',
                                'SY_SpreadRandomLocal_50_100.meantaul',
                                'SC_FluctAnal_2_rsrangefit_50_1_logi.prop_r1',
                                'PH_ForcePotential_sine_1_1_1.proppos',
                                'SP_Summaries_pgram_hamm.maxw',
                                'SP_Summaries_welch_rect.maxw']

        # catch16 feature indicator
        catch16Indicator = [item in featureNamesCatch16 for item in self.good_op_names]
        print([self.good_op_names[i] for i in range(len(self.good_op_ids)) if
                      self.good_op_names[i] in featureNamesCatch16])
        catch16IDs = [self.good_op_ids[i] for i in range(len(self.good_op_ids)) if
                      self.good_op_names[i] in featureNamesCatch16]
        print(len(catch16IDs))
        print(catch16IDs)
        
        catch22Indicator = [item in featureNamesCatch22 for item in self.good_op_names]
        catch22IDs = [self.good_op_ids[i] for i in range(len(self.good_op_ids)) if
                      self.good_op_names[i] in featureNamesCatch22]
        
        #import sklearn.tree as tree
        import sklearn.svm as svm
        from sklearn.model_selection import cross_val_score
        import time
        import numpy as np
        import operator

        # initialise tree
        #clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=23)
        clf = svm.LinearSVC(random_state=23)

        # load class balanced scorer
        from sklearn.metrics import make_scorer
        scorer = make_scorer(Feature_Stats.accuracy_score_class_balanced)

        # reference line (below other data)
        mpl.pyplot.figure()
        mpl.pyplot.plot((0, 1.5), (0, 1.5), '--', color=np.array((1, 1, 1)) * 0.7)


        perfmat = np.zeros((len(self.tasks), 8))
        # sort_dict = dict()
        # mf_id = 7385
        for task_ind, task in enumerate(self.tasks):

            t = time.time()
            print 'classifying task %s' % task.name

            # decide on number of folds
            un, counts = np.unique(task.labels, return_counts=True)
            max_folds = 10
            min_folds = 2
            folds = np.min([max_folds, np.max([min_folds, np.min(counts)])])

            # -- do cross-validated scoring for full and reduced matrix
            # To find highly correlated features w.r.t. MF (mf_id = 7385):
            # print(task.data.shape)
            # for i in range(len(self.good_op_ids)):
            #     r = np.corrcoef(task.data[:, np.isin(task.op_ids, mf_id)].T, task.data[:, np.isin(task.op_ids, self.good_op_ids[i])].T)
            #     #r[0, 1]
            #     if (self.good_op_names[i],self.good_op_ids[i]) not in sort_dict:
            #         sort_dict[self.good_op_names[i],self.good_op_ids[i]] = - abs(r[0, 1])
            #     else:
            #         sort_dict[self.good_op_names[i],self.good_op_ids[i]] = sort_dict[self.good_op_names[i],self.good_op_ids[i]] - abs(r[0, 1])

            # only good operations
            score_this_task_top_ops = cross_val_score(clf, task.data[:, np.isin(task.op_ids, self.good_perf_op_ids)], task.labels, cv=folds, scoring=scorer)

            # only cluster centers
            score_this_task_cluster_ops = cross_val_score(clf, task.data[:, np.isin(task.op_ids, catch16IDs)],#self.good_perf_cluster_center_op_ids)],
                                                      task.labels, cv=folds, scoring=scorer)

            # only super ops
            score_this_task_super_ops = cross_val_score(clf, task.data[:, np.isin(task.op_ids,
                                                                                    self.super_perf_op_ids)],
                                                          task.labels, cv=folds, scoring=scorer)

            # whole matrix
            score_this_task_whole = cross_val_score(clf, task.data, task.labels, cv=folds, scoring=scorer)

            # plot immediately
            '''p1 = mpl.pyplot.errorbar(np.mean(score_this_task_whole), np.mean(score_this_task_top_ops),
                                xerr=np.std(score_this_task_whole), yerr=np.std(score_this_task_top_ops), fmt='o', color='b', ecolor='b')'''
            p2 = mpl.pyplot.errorbar(np.mean(score_this_task_whole), np.mean(score_this_task_cluster_ops),
                                xerr=np.std(score_this_task_whole), yerr=np.std(score_this_task_cluster_ops), fmt='o',
                                color='r', ecolor='r')
            '''p3 = mpl.pyplot.errorbar(np.mean(score_this_task_whole), np.mean(score_this_task_super_ops),
                                     xerr=np.std(score_this_task_whole), yerr=np.std(score_this_task_super_ops),
                                     fmt='o',
                                     color='g', ecolor='g')'''

            # save scores
            perfmat[task_ind, 0] = np.mean(score_this_task_whole)
            perfmat[task_ind, 1] = np.std(score_this_task_whole)
            perfmat[task_ind, 2] = np.mean(score_this_task_top_ops)
            perfmat[task_ind, 3] = np.std(score_this_task_top_ops)
            perfmat[task_ind, 4] = np.mean(score_this_task_cluster_ops)
            perfmat[task_ind, 5] = np.std(score_this_task_cluster_ops)
            perfmat[task_ind, 6] = np.mean(score_this_task_super_ops)
            perfmat[task_ind, 7] = np.std(score_this_task_super_ops)

            print 'Done. Took %1.1f minutes.' % ((time.time() - t)/60)

        # sorted_tuple = sorted(sort_dict.items(), key=operator.itemgetter(1))
        # print(sorted_tuple[:30])
        # measures0 =  self.good_op_ids
        # measures2 = (self.stats_good_op_comb - np.nanmean(self.stats_good_op_comb))/np.nanstd(self.stats_good_op_comb)
        # count = 30
        # for elem in sorted_tuple:
        #     count = count - 1
        #     if count == 0:
        #         break
        #     op = elem[0][1] # id
        #     ind_tmp = np.nonzero(measures0==op)[0]
        #     norm_ustats = measures2[ind_tmp]
        #     print(elem[0][1], elem[0][0], -elem[1]/12, norm_ustats[0])
        
        mpl.pyplot.xlabel('performance on whole feature set')
        mpl.pyplot.ylabel('performance with catchaMouse16') # catchaMouse16
        np.savetxt(locations.rootDir() + "/peformance_mat_fullMeanStd_topMeanStd_clusterMeanStd_givenSplit_nonBalanced_new710_{}.txt".format(PARAMS['linkage_method']), perfmat)

        mpl.pyplot.savefig("{}/perf_compare_corr.svg".format(PARAMS['figure_dir']), format = 'svg', dpi=400, bbox_inches='tight', pad_inches = 0.25, transparent=True)
        #mpl.pyplot.show()

    def UMAP_all_topOps_clusters(self):

        import umap
        import pandas as pd
        import seaborn as sns

        reducer = umap.UMAP()

        for task_ind, task in enumerate(self.tasks):


            # # whole matrix
            # embedding = reducer.fit_transform(task.data)
            # df = pd.DataFrame(data=embedding, columns=('umap-1', 'umap-2'))
            # df['label'] = task.labels
            # lowDim = sns.lmplot(x='umap-1', y='umap-2', data=df, fit_reg=False, markers='.',
            #                     hue='label', legend=True, legend_out=True, palette='Set2')
            # titleString = '%s UMAP all %i features' % (task.name, len(task.op_ids))
            # mpl.pyplot.title(titleString)
            # mpl.pyplot.tight_layout()
            #
            # lowDim.savefig(locations.rootDir() + '/UMAP/' + titleString + '.png')

            # top ops
            embedding = reducer.fit_transform(task.data[:, np.isin(task.op_ids, self.good_perf_op_ids)])
            df = pd.DataFrame(data=embedding, columns=('umap-1', 'umap-2'))
            df['label'] = task.labels
            lowDim = sns.lmplot(x='umap-1', y='umap-2', data=df, fit_reg=False, markers='.',
                                hue='label', legend=True, legend_out=True, palette='Set2')
            titleString = '%s UMAP %i top ops' % (task.name, len(self.good_perf_op_ids))
            mpl.pyplot.title(titleString)
            mpl.pyplot.tight_layout()

            lowDim.savefig(locations.rootDir() + '/UMAP/' + titleString + '.png')

            # only cluster centers
            embedding = reducer.fit_transform(task.data[:, np.isin(task.op_ids, self.good_perf_cluster_center_op_ids)])
            df = pd.DataFrame(data=embedding, columns=('umap-1', 'umap-2'))
            df['label'] = task.labels
            lowDim = sns.lmplot(x='umap-1', y='umap-2', data=df, fit_reg=False, markers='.',
                                hue='label', legend=True, legend_out=True, palette='Set2')
            titleString = '%s UMAP %i cluster centers of %i top ops' % (task.name, len(self.good_perf_cluster_center_op_ids),
                                                                        len(self.good_perf_op_ids))
            mpl.pyplot.title(titleString)
            mpl.pyplot.tight_layout()

            lowDim.savefig(locations.rootDir() + '/UMAP/' + titleString + '.png')

            mpl.pyplot.close('all')



    def classify_good_perf_ops_vs_good_ops_givenSplit(self):

        import sklearn.tree as tree
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score
        import time

        # initialise tree
        clf = tree.DecisionTreeClassifier(random_state=23) # class_weight="balanced",

        # # load class balanced scorer
        # from sklearn.metrics import make_scorer
        # scorer = make_scorer(Feature_Stats.accuracy_score_class_balanced)

        # # reference line (below other data)
        # mpl.pyplot.plot((0, 1.5), (0, 1.5), '--', color=np.array((1, 1, 1)) * 0.7)

        perfmat = np.zeros((len(self.tasks), 6))
        for task_ind, task in enumerate(self.tasks):

            t = time.time()
            print 'classifying task %s' % task.name

            # get the train and test indices
            trainInds = np.array(['TRAIN' in elem for elem in task.keywords_ts])
            testInds = np.array(['TEST' in elem for elem in task.keywords_ts])

            if np.sum(trainInds)==0 or np.sum(testInds)==0:
                continue

            data_train = task.data[trainInds,:]
            data_test = task.data[testInds, :]

            labels_train = task.labels[trainInds]
            labels_test = task.labels[testInds]

            # top ops
            clf.fit(data_train[:, np.isin(task.op_ids, self.good_perf_op_ids)], labels_train)
            labels_test_predicted = clf.predict(data_test[:, np.isin(task.op_ids, self.good_perf_op_ids)])
            # score_this_task_top_ops = Feature_Stats.accuracy_score_class_balanced(labels_test,
            #                                                                       labels_test_predicted)
            score_this_task_top_ops = accuracy_score(labels_test, labels_test_predicted)

            # only cluster centers
            clf.fit(data_train[:, np.isin(task.op_ids, self.good_perf_cluster_center_op_ids)], labels_train)
            labels_test_predicted = clf.predict(data_test[:, np.isin(task.op_ids, self.good_perf_cluster_center_op_ids)])
            # score_this_task_cluster_ops = Feature_Stats.accuracy_score_class_balanced(labels_test,
            #                                                                       labels_test_predicted)
            score_this_task_cluster_ops = accuracy_score(labels_test, labels_test_predicted)

            # whole matrix
            clf.fit(data_train, labels_train)
            labels_test_predicted = clf.predict(data_test)
            # score_this_task_whole = Feature_Stats.accuracy_score_class_balanced(labels_test,
            #                                                                     labels_test_predicted)
            score_this_task_whole = accuracy_score(labels_test, labels_test_predicted)

            # save scores
            perfmat[task_ind, 0] = np.mean(score_this_task_whole)
            perfmat[task_ind, 1] = np.std(score_this_task_whole)
            perfmat[task_ind, 2] = np.mean(score_this_task_top_ops)
            perfmat[task_ind, 3] = np.std(score_this_task_top_ops)
            perfmat[task_ind, 4] = np.mean(score_this_task_cluster_ops)
            perfmat[task_ind, 5] = np.std(score_this_task_cluster_ops)

            print 'Done. Took %1.1f minutes.' % ((time.time() - t)/60)

        np.savetxt(locations.rootDir() + '/peformance_mat_fullMeanStd_topMeanStd_clusterMeanStd_givenSplit_nonBalanced_new710.txt', perfmat)

    def classify_selectedOps_givenSplit(self):

        import sklearn.tree as tree
        from sklearn.svm import LinearSVC
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score
        import time

        featureNamesCatch22 = ['DN_HistogramMode_5',
                               'DN_HistogramMode_10',
                               'CO_f1ecac',
                               'CO_FirstMin_ac',
                               'CO_HistogramAMI_even_2.ami5',
                               'IN_AutoMutualInfoStats_40_gaussian.fmmi',
                               'CO_trev_1.num',
                               'SB_TransitionMatrix_3ac.sumdiagcov',
                               'PD_PeriodicityWang.th2',
                               'CO_Embed2_Dist_tau.d_expfit_meandiff',
                               'FC_LocalSimple_mean1.tauresrat',
                               'FC_LocalSimple_mean3.stderr',
                               'DN_OutlierInclude_p_001.mdrmd',
                               'DN_OutlierInclude_n_001.mdrmd',
                               'SB_BinaryStats_diff.longstretch0',
                               'SB_BinaryStats_mean.longstretch1',
                               'SB_MotifThree_quantile.hh',
                               'SC_FluctAnal_2_rsrangefit_50_1_logi.prop_r1',
                               'SC_FluctAnal_2_dfa_50_1_2_logi.prop_r1',
                               'SP_Summaries_welch_rect.centroid',
                               'SP_Summaries_welch_rect.area_5_1',
                               'MD_hrv_classic.pnn40']

        # catch22 feature indicator
        catch22Indicator = [item in featureNamesCatch22 for item in self.good_op_names];
        catch22IDs= [self.good_op_ids[i] for i in range(len(self.good_op_ids)) if self.good_op_names[i] in featureNamesCatch22];

        # initialise tree
        clf = tree.DecisionTreeClassifier(random_state=23) # class_weight="balanced", # LinearSVC(random_state=23)  #

        perfmat = np.zeros(len(self.tasks))
        for task_ind, task in enumerate(self.tasks):

            t = time.time()
            print 'classifying task %s' % task.name

            # get the train and test indices
            trainInds = np.array(['TRAIN' in elem for elem in task.keywords_ts])
            testInds = np.array(['TEST' in elem for elem in task.keywords_ts])

            if np.sum(trainInds)==0 or np.sum(testInds)==0:
                continue

            data_train = task.data[trainInds,:]
            data_test = task.data[testInds, :]

            labels_train = task.labels[trainInds]
            labels_test = task.labels[testInds]

            # top ops
            clf.fit(data_train[:, np.isin(task.op_ids, catch22IDs)], labels_train)
            labels_test_predicted = clf.predict(data_test[:, np.isin(task.op_ids, catch22IDs)])

            score_this_task = accuracy_score(labels_test, labels_test_predicted)

            # save scores
            perfmat[task_ind] = score_this_task

            print 'Done. Took %1.1f minutes.' % ((time.time() - t)/60)

        np.savetxt(locations.rootDir() + '/peformance_mat_canonical_givenSplit_nonBalanced_linear.txt', perfmat)

    def greedy_selectedOps(self):

        import sklearn.tree as tree
        from sklearn.svm import LinearSVC
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score
        import time

        featureNamesCatch22 = ['DN_HistogramMode_5',
                               'DN_HistogramMode_10',
                               'CO_f1ecac',
                               'CO_FirstMin_ac',
                               'CO_HistogramAMI_even_2.ami5',
                               'IN_AutoMutualInfoStats_40_gaussian.fmmi',
                               'CO_trev_1.num',
                               'SB_TransitionMatrix_3ac.sumdiagcov',
                               'PD_PeriodicityWang.th2',
                               'CO_Embed2_Dist_tau.d_expfit_meandiff',
                               'FC_LocalSimple_mean1.tauresrat',
                               'FC_LocalSimple_mean3.stderr',
                               'DN_OutlierInclude_p_001.mdrmd',
                               'DN_OutlierInclude_n_001.mdrmd',
                               'SB_BinaryStats_diff.longstretch0',
                               'SB_BinaryStats_mean.longstretch1',
                               'SB_MotifThree_quantile.hh',
                               'SC_FluctAnal_2_rsrangefit_50_1_logi.prop_r1',
                               'SC_FluctAnal_2_dfa_50_1_2_logi.prop_r1',
                               'SP_Summaries_welch_rect.centroid',
                               'SP_Summaries_welch_rect.area_5_1',
                               'MD_hrv_classic.pnn40']

        # catch22 feature indicator
        catch22Indicator = [item in featureNamesCatch22 for item in self.good_op_names];
        catch22IDs= [self.good_op_ids[i] for i in range(len(self.good_op_ids)) if self.good_op_names[i] in featureNamesCatch22];

        # initialise tree
        clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=23)

        # load class balanced scorer
        from sklearn.metrics import make_scorer
        scorer = make_scorer(Feature_Stats.accuracy_score_class_balanced)

        for task_ind, task in enumerate(self.tasks):

            # decide on number of folds
            un, counts = np.unique(task.labels, return_counts=True)
            max_folds = 10
            min_folds = 2
            folds = np.min([max_folds, np.max([min_folds, np.min(counts)])])

            # only keep catch22 features
            thisTask22Indi = np.isin(task.op_ids, catch22IDs)
            filteredData = task.data[:, thisTask22Indi]
            filteredOpIDs = task.op_ids[thisTask22Indi]
            task_op_names = task.op['code_string']
            filteredOpNames = np.array(task_op_names)[thisTask22Indi]
            # filteredLabels = task.labels[thisTask22Indi]

            t = time.time()
            print '\nclassifying task %s' % task.name

            chosenInds = []
            remainingInds = range(len(filteredOpIDs))

            # number of features to select
            for k in range(3):

                meanErrors = np.full(len(remainingInds), fill_value=np.nan)
                for j, remainingIndTemp in enumerate(remainingInds):

                    indsTemp = chosenInds + [remainingIndTemp]

                    # only cluster centers
                    score = cross_val_score(clf, filteredData[:, indsTemp],
                                                                  task.labels, cv=folds, scoring=scorer)
                    meanErrors[j] = 1 - np.mean(score)

                # find minimum error and select corresponding remaining ind to be added
                addedInd = remainingInds[np.argmin(meanErrors)]
                chosenInds = chosenInds + [addedInd]

                # remove selected remaining ind from list
                remainingInds.remove(addedInd)

                # print operation selection
                print "%1.3f, " % np.min(meanErrors);
                for i in chosenInds:
                    print "%s, " % filteredOpNames[i];

            # # save scores
            # perfmat[task_ind] = score_this_task
            #
            # print 'Done. Took %1.1f minutes.' % ((time.time() - t)/60)

        # np.savetxt('/Users/carl/PycharmProjects/op_importance/peformance_mat_canonical_givenSplit_nonBalanced_linear.txt', perfmat)

    def selectBestTwoOf_selectedOps(self):

        import sklearn.tree as tree
        from sklearn.svm import LinearSVC
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score
        import time

        featureNamesCatch22 = ['DN_HistogramMode_5',
                               'DN_HistogramMode_10',
                               'CO_f1ecac',
                               'CO_FirstMin_ac',
                               'CO_HistogramAMI_even_2.ami5',
                               'IN_AutoMutualInfoStats_40_gaussian.fmmi',
                               'CO_trev_1.num',
                               'SB_TransitionMatrix_3ac.sumdiagcov',
                               'PD_PeriodicityWang.th2',
                               'CO_Embed2_Dist_tau.d_expfit_meandiff',
                               'FC_LocalSimple_mean1.tauresrat',
                               'FC_LocalSimple_mean3.stderr',
                               'DN_OutlierInclude_p_001.mdrmd',
                               'DN_OutlierInclude_n_001.mdrmd',
                               'SB_BinaryStats_diff.longstretch0',
                               'SB_BinaryStats_mean.longstretch1',
                               'SB_MotifThree_quantile.hh',
                               'SC_FluctAnal_2_rsrangefit_50_1_logi.prop_r1',
                               'SC_FluctAnal_2_dfa_50_1_2_logi.prop_r1',
                               'SP_Summaries_welch_rect.centroid',
                               'SP_Summaries_welch_rect.area_5_1',
                               'MD_hrv_classic.pnn40']

        # catch22 feature indicator
        catch22Indicator = [item in featureNamesCatch22 for item in self.good_op_names];
        catch22IDs= [self.good_op_ids[i] for i in range(len(self.good_op_ids)) if self.good_op_names[i] in featureNamesCatch22];

        # initialise tree
        clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=23)

        # load class balanced scorer
        from sklearn.metrics import make_scorer
        scorer = make_scorer(Feature_Stats.accuracy_score_class_balanced)

        for task_ind, task in enumerate(self.tasks):

            if not task.name in ['ShapeletSim', 'Plane']:
                continue

            # decide on number of folds
            un, counts = np.unique(task.labels, return_counts=True)
            max_folds = 10
            min_folds = 2
            folds = np.min([max_folds, np.max([min_folds, np.min(counts)])])

            # only keep catch22 features
            thisTask22Indi = np.isin(task.op_ids, catch22IDs)
            filteredData = task.data[:, thisTask22Indi]
            filteredOpIDs = task.op_ids[thisTask22Indi]
            task_op_names = task.op['code_string']
            filteredOpNames = np.array(task_op_names)[thisTask22Indi]
            # filteredLabels = task.labels[thisTask22Indi]

            t = time.time()
            print '\nclassifying task %s' % task.name

            featureInds = range(len(filteredOpIDs))

            meanErrors = np.full(len(featureInds), fill_value=np.nan)
            for j in featureInds:

                # only cluster centers
                score = cross_val_score(clf, filteredData[:, [j]], task.labels, cv=folds, scoring=scorer)

                meanErrors[j] = 1 - np.mean(score)

            # find minimum error and select corresponding remaining ind to be added
            sortedFeatureInds = np.argsort(meanErrors)

            chosenInds = sortedFeatureInds[0:2]

            mpl.pyplot.figure()
            uniqueLabels = np.unique(task.labels)
            for uniqueLabel in uniqueLabels:
                mpl.pyplot.scatter(filteredData[task.labels==uniqueLabel,chosenInds[0]], filteredData[task.labels==uniqueLabel,chosenInds[1]], label=uniqueLabel)
            mpl.pyplot.xlabel(filteredOpNames[chosenInds[0]])
            mpl.pyplot.ylabel(filteredOpNames[chosenInds[1]])
            mpl.pyplot.title(task.name)
            mpl.pyplot.legend(uniqueLabels)

        mpl.pyplot.show()



    def classify_N_clusters(self):

        import sklearn.tree as tree
        from sklearn.model_selection import cross_val_score
        import time

        # initialise tree
        clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=23)

        # load class balanced scorer
        from sklearn.metrics import make_scorer
        scorer = make_scorer(Feature_Stats.accuracy_score_class_balanced)

        # # reference line (below other data)
        # mpl.pyplot.plot((0, 1.5), (0, 1.5), '--', color=np.array((1, 1, 1)) * 0.7)

        n_clust_max = 51
        n_clust_step = 5
        n_clust_array = range(1,n_clust_max+1,n_clust_step)
        n_clust_steps = len(n_clust_array)

        n_topOps_array = np.arange(1,11)*100 # [815] # [200, 300, 400, 500, 700, 800, 900] # [100, 250, 500, 750, 1000]

        # format: 4 lines per number of clusters: (0) n_topOps (1) op ids (2) mean accuracy (3) std accuracy
        perfmat = np.full((n_clust_steps*len(n_topOps_array)*4, len(self.tasks)), np.nan)

        for n_topOpsInd, n_topOps in enumerate(n_topOps_array):

            print "\nNow taking %i topOps." % n_topOps

            # re-select the top ops
            self.n_good_perf_ops = n_topOps
            self.select_good_perf_ops()

            # -- intitialise the redundancy method with the calculated results
            self.init_redundancy_method_problem_space()
            # re-calculate the correlation matrix between top ops
            self.redundancy_method.calc_similarity()

            for clust_ind, n_clust in enumerate(n_clust_array):

                t = time.time()

                print '\n%i clusters' % n_clust

                # -- create n_clust clusters
                self.redundancy_method.calc_hierch_cluster(t=n_clust, criterion='maxclust') 

                # -- single features for each cluster
                self.select_good_perf_cluster_center_ops()

                # save number of top ops
                perfmat[4 * clust_ind + 4 * n_clust_steps * n_topOpsInd, 0] = n_topOps
                # save op ids
                perfmat[4 * clust_ind + 4 * n_clust_steps * n_topOpsInd + 1, 0:n_clust] = self.good_perf_cluster_center_op_ids

                for task_ind, task in enumerate(self.tasks):

                    print 'classifying task %s' % task.name

                    # decide on number of folds
                    un, counts = np.unique(task.labels, return_counts=True)
                    max_folds = 10
                    min_folds = 2
                    folds = np.min([max_folds, np.max([min_folds, np.min(counts)])])

                    # -- do cross-validated scoring for full and reduced matrix

                    # # only good operations
                    # score_this_task_top_ops = cross_val_score(clf,
                    #                                           task.data[:, np.isin(task.op_ids, self.good_perf_op_ids)],
                    #                                           task.labels, cv=folds)  # , scoring=scorer)

                    if task.name == 'Meat':
                        print 'hm'

                    # only cluster centers
                    thisClusterData = task.data[:, np.isin(task.op_ids,self.good_perf_cluster_center_op_ids)];
                    if np.size(thisClusterData) == 0:
                        score_this_task_cluster_ops = np.full((1,2), np.nan)
                    else:
                        score_this_task_cluster_ops = cross_val_score(clf, task.data[:, np.isin(task.op_ids,
                                                                                                self.good_perf_cluster_center_op_ids)],
                                                                      task.labels, cv=folds, scoring=scorer)

                    # # whole matrix
                    # score_this_task_whole = cross_val_score(clf, task.data, task.labels, cv=folds, scoring=scorer)

                    # # plot immediately
                    # p1 = mpl.pyplot.errorbar(np.mean(score_this_task_whole), np.mean(score_this_task_top_ops),
                    #                          xerr=np.std(score_this_task_whole), yerr=np.std(score_this_task_top_ops),
                    #                          fmt='o', color='b', ecolor='b')
                    # p2 = mpl.pyplot.errorbar(np.mean(score_this_task_whole), np.mean(score_this_task_cluster_ops),
                    #                          xerr=np.std(score_this_task_whole), yerr=np.std(score_this_task_cluster_ops),
                    #                          fmt='o',
                    #                          color='r', ecolor='r')

                    # save scores
                    perfmat[4*clust_ind + 4 * n_clust_steps * n_topOpsInd + 2, task_ind] = np.mean(score_this_task_cluster_ops)
                    perfmat[4*clust_ind + 4 * n_clust_steps * n_topOpsInd + 3, task_ind] = np.std(score_this_task_cluster_ops)

                print 'Done. Took %1.1f minutes.' % ((time.time() - t) / 60)

        np.savetxt(
            locations.rootDir() + '/peformance_mat_n_clusters_new_variableNTopOps_noRaw.txt',
            perfmat)

        # mpl.pyplot.legend((p1, p2), ('500 top ops', 'only cluster centers'))
        # # mpl.pyplot.xlim((0, 1))
        # # mpl.pyplot.ylim((0, 1))
        # mpl.pyplot.xlabel('performance on whole feature set')
        # mpl.pyplot.ylabel('performance only selected features')
        # mpl.pyplot.ylabel('class imbalance corrected performance')
        # mpl.pyplot.show()

    def show_high_performer_dist(self, op_name):

        opInd = np.where(op_name == workflow.good_op_names)

        zscores = []

        for taskInd, taskName in enumerate(self.task_names):

            f = mpl.pyplot.figure()

            statsThisTask = self.stats_good_op[taskInd, :]
            statsThisTask = np.ma.filled(statsThisTask, np.nan)

            meanErrorThisTask = np.nanmean(statsThisTask)
            stdErrorThisTask = np.nanstd(statsThisTask)

            statsThisTaskThisOp = statsThisTask[opInd]
            if np.isnan(statsThisTaskThisOp):
                print 'error NaN for ' + taskName + ', ' + op_name
                continue

            zScoreThisOp = (statsThisTaskThisOp - meanErrorThisTask)/stdErrorThisTask

            zscores.append(zScoreThisOp)

            mpl.pyplot.hist(statsThisTask[np.logical_not(np.isnan(statsThisTask))], np.linspace(0,1,100))
            ax = mpl.pyplot.gca()
            ax.axvline(statsThisTaskThisOp, color='r', linewidth=2)

            mpl.pyplot.title(taskName + ', ' + op_name + ', z-score=' + str(zScoreThisOp))

            mpl.pyplot.xlabel('error')
            mpl.pyplot.ylabel('frequency')

            # mpl.pyplot.show()

            mpl.pyplot.savefig(locations.rootDir() + '/errorHistogramsHighPerformers/' + op_name + ',' + taskName + '.png')

            mpl.pyplot.close(f)

        # -- not for all tasks pooled
        meanErrorsAllTasks = np.nanmean(np.ma.filled(self.stats_good_op, np.nan), 0)
        meanErrorsAllTasksThisOp = meanErrorsAllTasks[opInd]

        mpl.pyplot.hist(meanErrorsAllTasks[np.logical_not(np.isnan(meanErrorsAllTasks))], np.linspace(0, 1, 100))
        ax = mpl.pyplot.gca()
        ax.axvline(meanErrorsAllTasksThisOp, color='r', linewidth=2)

        zScoreThisOp = (meanErrorsAllTasksThisOp - np.nanmean(meanErrorsAllTasks))/np.nanstd(meanErrorsAllTasks)

        mpl.pyplot.title('allTasks, ' + op_name + ', z-score=' + str(zScoreThisOp))

        mpl.pyplot.xlabel('error')
        mpl.pyplot.ylabel('frequency')

        # mpl.pyplot.show()

        mpl.pyplot.savefig(
            locations.rootDir() + '/errorHistogramsHighPerformers/' + op_name + ',_allTasks.png')

        mpl.pyplot.close(f)


        print 'mean over z-scores ' + str(np.mean(zscores))

    def show_catch22_perfmat(self):

        featureNamesCatch22 = ['DN_HistogramMode_5',
                                'DN_HistogramMode_10',
                                'CO_f1ecac',
                                'CO_FirstMin_ac',
                                'CO_HistogramAMI_even_2.ami5',
                                'IN_AutoMutualInfoStats_40_gaussian.fmmi',
                                'CO_trev_1.num',
                                'SB_TransitionMatrix_3ac.sumdiagcov',
                                'PD_PeriodicityWang.th2',
                                'CO_Embed2_Dist_tau.d_expfit_meandiff',
                                'FC_LocalSimple_mean1.tauresrat',
                                'FC_LocalSimple_mean3.stderr',
                                'DN_OutlierInclude_p_001.mdrmd',
                                'DN_OutlierInclude_n_001.mdrmd',
                                'SB_BinaryStats_diff.longstretch0',
                                'SB_BinaryStats_mean.longstretch1',
                                'SB_MotifThree_quantile.hh',
                                'SC_FluctAnal_2_rsrangefit_50_1_logi.prop_r1',
                                'SC_FluctAnal_2_dfa_50_1_2_logi.prop_r1',
                                'SP_Summaries_welch_rect.centroid',
                                'SP_Summaries_welch_rect.area_5_1',
                                'MD_hrv_classic.pnn40']

        # catch22 feature indicator
        catch22Indicator = [item in featureNamesCatch22 for item in self.good_op_names];

        # filter normalised performance matrix
        perfMatCatch22 = self.stats_good_op_norm[:, catch22Indicator];

        # -- norm each column to z-score of feature per task
        perfMatCatch22Norm = -self.stats_good_op[:, catch22Indicator].T;

        for colInd in range(np.shape(perfMatCatch22Norm)[1]):

            perfMatCatch22Norm[:, colInd] = \
                (perfMatCatch22Norm[:, colInd] - np.mean(perfMatCatch22Norm[:, colInd]))\
                /np.std(perfMatCatch22Norm[:, colInd]);



        # order columns and rows
        import modules.misc.PK_helper as hlp
        import modules.feature_importance.PK_ident_top_op as idtop
        from scipy.spatial.distance import pdist, squareform
        from scipy.cluster.hierarchy import linkage, dendrogram

        # abs_corr_array = abs_corr_array[index, :]
        # abs_corr_array = abs_corr_array[:, index]

        # # sort cols
        # abs_corr_array , _, _ = \
        #     idtop.calc_perform_corr_mat(perfMatCatch22Norm, norm=None,
        #                                  type='abscorr')
        # corr_linkage = idtop.calc_linkage(abs_corr_array)[0]
        # corr_dendrogram = dendrogram(corr_linkage,no_plot=True)
        # sort_ind_cols = corr_dendrogram['leaves']
        # perfMatCatch22Norm = perfMatCatch22Norm[:, sort_ind_cols]

        # # sort rows
        # abs_corr_array, sort_ind_rows, _ = \
        #     idtop.calc_perform_corr_mat(perfMatCatch22Norm.T, norm=None,
        #                                 type='euc')
        # corr_linkage = idtop.calc_linkage(abs_corr_array)[0]
        # corr_dendrogram = dendrogram(corr_linkage,no_plot=True)
        # sort_ind_rows = corr_dendrogram['leaves']
        # perfMatCatch22Norm = perfMatCatch22Norm[sort_ind_rows, :]

        # sort with ignored NaNs
        perfMatCatch22NormArray = np.ma.filled(perfMatCatch22Norm, np.nan)

        # cols
        perMatFilt = perfMatCatch22NormArray[np.logical_not(np.any(np.isnan(perfMatCatch22NormArray), 1)), :]
        distanceCols = squareform(pdist(perMatFilt.T, 'correlation'))
        Z = linkage(distanceCols)
        R = dendrogram(Z, no_plot=True)
        sort_ind_cols = R['leaves']
        perfMatCatch22Norm = perfMatCatch22Norm[:, sort_ind_cols]

        # rows
        perMatFilt = perfMatCatch22NormArray[:,np.logical_not(np.any(np.isnan(perfMatCatch22NormArray), 0))]
        distanceCols = squareform(pdist(perMatFilt, 'correlation'))
        Z = linkage(distanceCols)
        R = dendrogram(Z, no_plot=True)
        sort_ind_rows = R['leaves']
        perfMatCatch22Norm = perfMatCatch22Norm[sort_ind_rows, :]

        absMax = np.max((-np.min(perfMatCatch22Norm), np.max(perfMatCatch22Norm)))

        mpl.pyplot.figure()
        # mpl.pyplot.imshow(perfMatCatch22Norm,
        #                   vmin=-absMax, vmax=absMax,
        #                   cmap='coolwarm')
        mpl.pyplot.imshow(perfMatCatch22Norm,
                          vmin=-3, vmax=3,
                          cmap='coolwarm')

        yTicks = self.good_op_names[catch22Indicator]
        yTicks = yTicks[sort_ind_rows]
        mpl.pyplot.yticks(range(22))

        ax = mpl.pyplot.gca()
        ax.set_yticklabels(yTicks, fontdict={'fontsize': 6})

        xTicks = self.task_names
        xTicks = np.array(xTicks)[sort_ind_cols]
        mpl.pyplot.xticks(range(len(self.task_names)))
        ax.set_xticklabels(xTicks, fontdict={'fontsize': 6}, rotation='90')

        cb = mpl.pyplot.colorbar();
        cb.set_label('accuracy z-scored per task')

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.pyplot.savefig(
            locations.rootDir() + '/catch22_performance_zscore_clustered.pdf')
        mpl.pyplot.show()


        # rows, tasks
        dist_mat = squareform(1 - np.abs(1 - pdist(perfMatCatch22, 'correlation')))
        N = len(dist_mat)
        flat_dist_mat = squareform(dist_mat)
        res_linkage = linkage(flat_dist_mat, method='complete')
        R = dendrogram(res_linkage, no_plot=True)
        task_order = R['leaves']
        perfMatCatch22 = perfMatCatch22[task_order, :]

        # columns, ops
        dist_mat = squareform(1 - np.abs(1 - pdist(perfMatCatch22.T, 'correlation')))
        N = len(dist_mat)
        flat_dist_mat = squareform(dist_mat)
        res_linkage = linkage(flat_dist_mat, method='complete')
        R = dendrogram(res_linkage, no_plot=True)
        op_order = R['leaves']
        perfMatCatch22 = perfMatCatch22[:, op_order]

        # -- show different plot, corr-distance of catch22 features
        #
        # mpl.pyplot.figure()
        # dist_mat_o = dist_mat[op_order,:]
        # dist_mat_o = dist_mat_o[:, op_order]
        # mpl.pyplot.imshow(dist_mat_o)
        # yTicks = self.good_op_names[catch22Indicator]
        # yTicks = yTicks[op_order]
        # mpl.pyplot.yticks(range(22))
        # ax = mpl.pyplot.gca()
        # ax.set_yticklabels(yTicks, fontdict={'fontsize': 4})
        #
        # cb = mpl.pyplot.colorbar();
        # cb.set_label('1-abscorr')
        #
        # mpl.pyplot.show()

        # corr_linkage = idtop.calc_linkage(abs_corr_array)[0]
        #
        # corr_dendrogram = hierarchy.dendrogram(corr_linkage, orientation='left')
        # # axdendro.set_xticks([])
        # axdendro.set_yticks([])
        # axdendro.axvline(max_dist_cluster, ls='--', c='k')
        # # Plot distance matrix.
        # axmatrix = fig.add_axes(rect_matrix)
        # index = corr_dendrogram['leaves']
        # abs_corr_array = abs_corr_array[index, :]
        # abs_corr_array = abs_corr_array[:, index]

        # show
        mpl.pyplot.figure()
        mpl.pyplot.imshow(perfMatCatch22.T,
                          vmin=-np.max(perfMatCatch22), vmax=np.max(perfMatCatch22),
                          cmap='coolwarm')

        yTicks = self.good_op_names[catch22Indicator]
        yTicks = yTicks[op_order]
        mpl.pyplot.yticks(range(22))

        ax = mpl.pyplot.gca()
        ax.set_yticklabels(yTicks, fontdict={'fontsize': 4})

        xTicks = self.task_names
        xTicks = np.array(xTicks)[task_order]
        mpl.pyplot.xticks(range(len(self.task_names)))
        ax.set_xticklabels(xTicks, fontdict={'fontsize': 4}, rotation='90')

        cb = mpl.pyplot.colorbar();
        cb.set_label('normalised performance')

        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.pyplot.savefig(
            locations.rootDir() + '/catch22_performance_corr.pdf')

        mpl.pyplot.show()
    
    def classify_using_features(self, feature, mode, which_split = 1 ): # mode = 'selection' or 'evaluate'
        '''
        Classify the dataset using the given feature set

        Parameters:
        -----------
        feature : numpy array
            The set of features to be used

        mode : string
            Selection or Evaluation of features on the dataset

        which_split : integer
            Different splits
        
        Returns:
        --------
            Mean task accuracy
        '''

        import time
        import random
        from sklearn.svm import LinearSVC
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import StratifiedShuffleSplit
        # initialise tree
        #clf = tree.DecisionTreeClassifier(class_weight="balanced", random_state=23)
        clf = LinearSVC(random_state=23)

        # load class balanced scorer
        from sklearn.metrics import make_scorer
        scorer = make_scorer(Feature_Stats.accuracy_score_class_balanced)

        task_acc = []
        for task_ind, task in enumerate(self.tasks):

            # One split of data into 90% and 10%
            # -- uses random_state to get split
            sss = StratifiedShuffleSplit(n_splits=3, test_size=0.1, random_state=23) # leave 10% data for final evaluation
            count = 1
            for train_index, test_index in sss.split(task.data, task.labels): # mutually exclusive splits
                if count == which_split:
                    X_train, X_test = task.data[train_index,:], task.data[test_index,:]
                    y_train, y_test = task.labels[train_index], task.labels[test_index]
                count = count + 1

            # decide on number of folds
            if mode == 'selection':
                un, counts = np.unique(y_train, return_counts=True)
            else:
                un, counts = np.unique(task.labels, return_counts=True)
            max_folds = 10
            min_folds = 2
            folds = np.min([max_folds, np.max([min_folds, np.min(counts)])])

            with_cv = False # do cross-validation or not ?
            if mode == 'selection':
                if with_cv:
                    # -- do cross-validated scoring for 90% data (in-sample testing)
                    score_this_task_whole = cross_val_score(clf, X_train[:, np.isin( task.op_ids, feature ) ], y_train, cv=folds, scoring=scorer)
                else:
                    #train_test_split --- to implement
                    #train_data, test_data, train_label, test_label = train_test_split(X_train, y_train, test_size=0.2, random_state=23, stratify = y_train)
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=23)
                    for train_index, test_index in sss.split(X_train, y_train): # mutually exclusive splits
                        train_data, test_data = X_train[train_index,:], X_train[test_index,:]
                        train_label, test_label = y_train[train_index], y_train[test_index]
                    clf.fit(train_data[:, np.isin( task.op_ids, feature ) ], train_label)
                    pred_label = clf.predict(test_data[:, np.isin( task.op_ids, feature ) ])
                    score_this_task_whole = Feature_Stats.accuracy_score_class_balanced(test_label,pred_label)
            else:
                # -- Train on 90% data and test it on 10% (out-of-sample testing)
                clf.fit(X_train[:, np.isin( task.op_ids, feature ) ], y_train)
                y_pred = clf.predict(X_test[:, np.isin( task.op_ids, feature ) ])
                score_this_task_whole = Feature_Stats.accuracy_score_class_balanced(y_test,y_pred)
            task_acc.append( np.mean( score_this_task_whole ) ) # append mean of cross validation accuracy
        return ( task_acc ) # min or mean

    def greedy_fwd_selection(self, which_split): # think of using 'min' instead of 'mean'
        '''
        Greedy forward selection algorithm to select the features based it's performance on the given dataset

        Parameters:
        -----------
        which_split: integer
            To get different splits

        Returns:
        --------
        selected_ids: list
            The feature ids of selected features 
        selected_acc: list
            Iteration wise best accuracy in greedy algo
        dataplot: list 2d
            Contains accuracy across the tasks
        '''
        import time
        from pathos.multiprocessing import ThreadPool as Pool
        
        # get ids for 7000 features
        rest_ids = self.good_op_ids.tolist() # np array
        #rest_ids = rest_ids[0:8] ## comment later to get it working
        dataplot = []
        # select best performing feature initially
        best_id = 0
        best_acc = 0
        best_overall_acc = []
        for id in rest_ids: # to get first best single feature
            overall_acc = self.classify_using_features( np.array([id]), 'selection', which_split )
            avg_acc = np.mean(overall_acc)
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_id = id
                best_overall_acc = overall_acc
        dataplot.append(best_overall_acc)
        #print('selected features are: ')
        #print(best_op_id, best_acc)

        selected_ids = []
        selected_acc = []
        #selected_ids = [7257, 117, 3161, 6968, 3662, 7784, 2883, 4062, 2743, 728, 3789, 3902, 2072, 6839, 6970, 4707, 204] # []
        #best_acc = self.classify_using_features(np.array(selected_ids))
        
        #print(best_acc)
        new_acc = best_acc
        prev_acc = 0
        selected_ids.append(best_id)
        rest_ids.remove(best_id)
        selected_acc.append(best_acc)
        # threshold = 0.00001 # in percentage: 10^-3 error
        for r in range(5):# upto 5 features only #while new_acc > prev_acc: 
            best_acc = 0
            best_id = 0
            def find_best_feat(id): # Find best
                candidate_ids = list(selected_ids) # copy
                candidate_ids.append(id)
                overall_acc = self.classify_using_features( np.array(candidate_ids), 'selection', which_split )
                avg_acc = np.mean(overall_acc)
                return avg_acc, id, overall_acc
            p = Pool(processes=8)
            arr = [ id for id in rest_ids]
            t = p.map(find_best_feat, arr)
            pair = sorted(t,key=lambda x: x[0], reverse=True)[0] # three return values?? how does this changes code?
            best_acc = pair[0]
            best_id = pair[1]
            best_overall_acc = pair[2]
            dataplot.append(best_overall_acc)
            #Add the best performing feature to selected_feature_set
            #print(best_id,max_acc)
            selected_ids.append( best_id )

            # Remove it from rest_of_the_features set
            rest_ids.remove( best_id )
            if len(rest_ids)==0:
                break
            prev_acc = new_acc
            new_acc = best_acc
            selected_acc.append(new_acc)
            #print(abs(prev_acc - new_acc))
        return selected_ids, selected_acc, dataplot # row for a feature
    
    def test_greedy(self):
        
        for i in range(1,4): # different splits
            print('Split',i)
            ids, acc, dataplot = self.greedy_fwd_selection(i)
            print(ids)
            print(acc)
            fig = mpl.pyplot.figure(figsize=(13,6))
            mpl.pyplot.matshow( np.array(dataplot).T.tolist(), cmap = 'OrRd', fignum=1)
            clb = mpl.pyplot.colorbar()
            clb.ax.set_title('accuracy')
            mpl.pyplot.xticks(range(0,len(ids)), range(1,len(ids)+1), rotation = 45 ) 
            mpl.pyplot.yticks(np.arange(0,len(self.task_names)), self.task_names )

            mpl.pyplot.title('Selection performance matrix for split '+str(i)+'\n', fontsize= 16, horizontalalignment='center')
            mpl.pyplot.xlabel('\nfeature (or iterations)')
            #mpl.pyplot.show()

            mpl.pyplot.savefig('Plots/Select_Split_'+str(i)+'_performance_plot.png')
            mpl.pyplot.close()
            eval_acc = []
            avg_eval_acc = []
            current_ids = []
            for id in ids:
                current_ids.append(id)
                temp = self.classify_using_features( np.array(current_ids), mode = 'evaluate', which_split = i )
                eval_acc.append(temp)
                avg_eval_acc.append( np.mean(temp))
            print(avg_eval_acc)
            print('\n')
            mpl.pyplot.figure(figsize=(13,6))
            mpl.pyplot.matshow( np.array(eval_acc).T.tolist(), cmap = 'OrRd', fignum=1)
            clb = mpl.pyplot.colorbar()
            clb.ax.set_title('accuracy')
            mpl.pyplot.xticks(range(0,len(ids)), range(1,len(ids)+1), rotation = 45 ) 
            mpl.pyplot.yticks(np.arange(0,len(self.task_names)), self.task_names )
            mpl.pyplot.title('Evaluation performance matrix for split '+str(i)+'\n', fontsize= 16, horizontalalignment='center')
            mpl.pyplot.xlabel('\nfeature (or iterations)')
            #mpl.pyplot.show()
            mpl.pyplot.savefig('Plots/Eval_Split_'+str(i)+'_performance_plot.png')
            mpl.pyplot.close()


    def testing_parameters(self):
        '''
        Testing the parameters of hierarchical clustering

        '''
        from sklearn.svm import LinearSVC
        from sklearn.model_selection import cross_val_score
        import matplotlib.cm as cm
        from matplotlib.colors import ListedColormap
        turbo_colormap_data = [[0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],[0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],[0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],[0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],[0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],[0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],[0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],[0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],[0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],[0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],[0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],[0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],[0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],[0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],[0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],[0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],[0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],[0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],[0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],[0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],[0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],[0.27381,0.52069,0.97899],[0.27273,0.53040,0.98461],[0.27106,0.54015,0.98930],[0.26878,0.54995,0.99303],[0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],[0.25862,0.57958,0.99876],[0.25425,0.58950,0.99896],[0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],[0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],[0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],[0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],[0.20021,0.67842,0.96833],[0.19326,0.68812,0.96190],[0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],[0.17223,0.71680,0.93981],[0.16529,0.72620,0.93161],[0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],[0.14519,0.75381,0.90496],[0.13886,0.76279,0.89550],[0.13278,0.77165,0.88580],[0.12698,0.78037,0.87590],[0.12151,0.78896,0.86581],[0.11639,0.79740,0.85559],[0.11167,0.80569,0.84525],[0.10738,0.81381,0.83484],[0.10357,0.82177,0.82437],[0.10026,0.82955,0.81389],[0.09750,0.83714,0.80342],[0.09532,0.84455,0.79299],[0.09377,0.85175,0.78264],[0.09287,0.85875,0.77240],[0.09267,0.86554,0.76230],[0.09320,0.87211,0.75237],[0.09451,0.87844,0.74265],[0.09662,0.88454,0.73316],[0.09958,0.89040,0.72393],[0.10342,0.89600,0.71500],[0.10815,0.90142,0.70599],[0.11374,0.90673,0.69651],[0.12014,0.91193,0.68660],[0.12733,0.91701,0.67627],[0.13526,0.92197,0.66556],[0.14391,0.92680,0.65448],[0.15323,0.93151,0.64308],[0.16319,0.93609,0.63137],[0.17377,0.94053,0.61938],[0.18491,0.94484,0.60713],[0.19659,0.94901,0.59466],[0.20877,0.95304,0.58199],[0.22142,0.95692,0.56914],[0.23449,0.96065,0.55614],[0.24797,0.96423,0.54303],[0.26180,0.96765,0.52981],[0.27597,0.97092,0.51653],[0.29042,0.97403,0.50321],[0.30513,0.97697,0.48987],[0.32006,0.97974,0.47654],[0.33517,0.98234,0.46325],[0.35043,0.98477,0.45002],[0.36581,0.98702,0.43688],[0.38127,0.98909,0.42386],[0.39678,0.99098,0.41098],[0.41229,0.99268,0.39826],[0.42778,0.99419,0.38575],[0.44321,0.99551,0.37345],[0.45854,0.99663,0.36140],[0.47375,0.99755,0.34963],[0.48879,0.99828,0.33816],[0.50362,0.99879,0.32701],[0.51822,0.99910,0.31622],[0.53255,0.99919,0.30581],[0.54658,0.99907,0.29581],[0.56026,0.99873,0.28623],[0.57357,0.99817,0.27712],[0.58646,0.99739,0.26849],[0.59891,0.99638,0.26038],[0.61088,0.99514,0.25280],[0.62233,0.99366,0.24579],[0.63323,0.99195,0.23937],[0.64362,0.98999,0.23356],[0.65394,0.98775,0.22835],[0.66428,0.98524,0.22370],[0.67462,0.98246,0.21960],[0.68494,0.97941,0.21602],[0.69525,0.97610,0.21294],[0.70553,0.97255,0.21032],[0.71577,0.96875,0.20815],[0.72596,0.96470,0.20640],[0.73610,0.96043,0.20504],[0.74617,0.95593,0.20406],[0.75617,0.95121,0.20343],[0.76608,0.94627,0.20311],[0.77591,0.94113,0.20310],[0.78563,0.93579,0.20336],[0.79524,0.93025,0.20386],[0.80473,0.92452,0.20459],[0.81410,0.91861,0.20552],[0.82333,0.91253,0.20663],[0.83241,0.90627,0.20788],[0.84133,0.89986,0.20926],[0.85010,0.89328,0.21074],[0.85868,0.88655,0.21230],[0.86709,0.87968,0.21391],[0.87530,0.87267,0.21555],[0.88331,0.86553,0.21719],[0.89112,0.85826,0.21880],[0.89870,0.85087,0.22038],[0.90605,0.84337,0.22188],[0.91317,0.83576,0.22328],[0.92004,0.82806,0.22456],[0.92666,0.82025,0.22570],[0.93301,0.81236,0.22667],[0.93909,0.80439,0.22744],[0.94489,0.79634,0.22800],[0.95039,0.78823,0.22831],[0.95560,0.78005,0.22836],[0.96049,0.77181,0.22811],[0.96507,0.76352,0.22754],[0.96931,0.75519,0.22663],[0.97323,0.74682,0.22536],[0.97679,0.73842,0.22369],[0.98000,0.73000,0.22161],[0.98289,0.72140,0.21918],[0.98549,0.71250,0.21650],[0.98781,0.70330,0.21358],[0.98986,0.69382,0.21043],[0.99163,0.68408,0.20706],[0.99314,0.67408,0.20348],[0.99438,0.66386,0.19971],[0.99535,0.65341,0.19577],[0.99607,0.64277,0.19165],[0.99654,0.63193,0.18738],[0.99675,0.62093,0.18297],[0.99672,0.60977,0.17842],[0.99644,0.59846,0.17376],[0.99593,0.58703,0.16899],[0.99517,0.57549,0.16412],[0.99419,0.56386,0.15918],[0.99297,0.55214,0.15417],[0.99153,0.54036,0.14910],[0.98987,0.52854,0.14398],[0.98799,0.51667,0.13883],[0.98590,0.50479,0.13367],[0.98360,0.49291,0.12849],[0.98108,0.48104,0.12332],[0.97837,0.46920,0.11817],[0.97545,0.45740,0.11305],[0.97234,0.44565,0.10797],[0.96904,0.43399,0.10294],[0.96555,0.42241,0.09798],[0.96187,0.41093,0.09310],[0.95801,0.39958,0.08831],[0.95398,0.38836,0.08362],[0.94977,0.37729,0.07905],[0.94538,0.36638,0.07461],[0.94084,0.35566,0.07031],[0.93612,0.34513,0.06616],[0.93125,0.33482,0.06218],[0.92623,0.32473,0.05837],[0.92105,0.31489,0.05475],[0.91572,0.30530,0.05134],[0.91024,0.29599,0.04814],[0.90463,0.28696,0.04516],[0.89888,0.27824,0.04243],[0.89298,0.26981,0.03993],[0.88691,0.26152,0.03753],[0.88066,0.25334,0.03521],[0.87422,0.24526,0.03297],[0.86760,0.23730,0.03082],[0.86079,0.22945,0.02875],[0.85380,0.22170,0.02677],[0.84662,0.21407,0.02487],[0.83926,0.20654,0.02305],[0.83172,0.19912,0.02131],[0.82399,0.19182,0.01966],[0.81608,0.18462,0.01809],[0.80799,0.17753,0.01660],[0.79971,0.17055,0.01520],[0.79125,0.16368,0.01387],[0.78260,0.15693,0.01264],[0.77377,0.15028,0.01148],[0.76476,0.14374,0.01041],[0.75556,0.13731,0.00942],[0.74617,0.13098,0.00851],[0.73661,0.12477,0.00769],[0.72686,0.11867,0.00695],[0.71692,0.11268,0.00629],[0.70680,0.10680,0.00571],[0.69650,0.10102,0.00522],[0.68602,0.09536,0.00481],[0.67535,0.08980,0.00449],[0.66449,0.08436,0.00424],[0.65345,0.07902,0.00408],[0.64223,0.07380,0.00401],[0.63082,0.06868,0.00401],[0.61923,0.06367,0.00410],[0.60746,0.05878,0.00427],[0.59550,0.05399,0.00453],[0.58336,0.04931,0.00486],[0.57103,0.04474,0.00529],[0.55852,0.04028,0.00579],[0.54583,0.03593,0.00638],[0.53295,0.03169,0.00705],[0.51989,0.02756,0.00780],[0.50664,0.02354,0.00863],[0.49321,0.01963,0.00955],[0.47960,0.01583,0.01055]]
        cm.register_cmap('turbo', cmap=ListedColormap(turbo_colormap_data))
        import time
        import random

        clf = LinearSVC(random_state=23)

        # load class balanced scorer
        from sklearn.metrics import make_scorer
        scorer = make_scorer(Feature_Stats.accuracy_score_class_balanced)
        
        #----- Set the hyperparameters ------
        
        # Which plot to show?
        which_plot = 'datamat' # options --> ('datamat', 'dist')
        
        # Compute the matrix? If matrix already computed, then put false
        
        n_clust_max = 0.5
        n_clust_step = 0.1
        
        # Threshold values
        n_clust_array = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5 ])
        n_clust_array = np.around(n_clust_array,3)
        n_clust_steps = len(n_clust_array)

        # Number of Top features
        n_topOps_array = np.array([100])
        
        #-------------------------------------
        
        result_mat = []
        reduced = np.empty((len(self.tasks), n_clust_steps))

        if PARAMS['calculate_mat']:
            for n_topOpsInd, n_topOps in enumerate(n_topOps_array):

                print "\nNow taking %i topOps." % n_topOps
                for clust_ind, n_clust in enumerate(n_clust_array):
                    
                    t = time.time()
                    print '\n%f cluster threshold' % n_clust
                    left_out_task_acc = []
                    for task_ind, task in enumerate(self.tasks): # leave-one-task-out -- leave one task and calculate performance, avg over all such perf.
                        
                        # leave task_ind 
                        # calc performance with left task
                        self.collect_stats_good_op_ids(leave_task = task_ind)
                        # re-select the top ops
                        self.n_good_perf_ops = n_topOps
                        self.select_good_perf_ops()

                        # -- intitialise the redundancy method with the calculated results
                        self.init_redundancy_method_problem_space()
                        # re-calculate the correlation matrix between top ops
                        self.redundancy_method.calc_similarity()
                        # -- create n_clust clusters
                        print(n_clust)
                        self.redundancy_method.calc_hierch_cluster(t=n_clust)#, criterion='distance') 

                        # -- select single features for each cluster
                        self.select_good_perf_cluster_center_ops()

                        # decide on number of folds
                        un, counts = np.unique(task.labels, return_counts=True)
                        max_folds = 10
                        min_folds = 2
                        folds = np.min([max_folds, np.max([min_folds, np.min(counts)])])

                        # -- do cross-validated scoring
                        # only cluster centers
                        thisClusterData = task.data[:, np.isin(task.op_ids,self.good_perf_cluster_center_op_ids)]

                        if np.size(thisClusterData) == 0:
                            score_this_task_cluster_ops = np.full((1,2), np.nan)
                        else:
                            score_this_task_cluster_ops = cross_val_score(clf, task.data[:, np.isin(task.op_ids,
                                                                        self.good_perf_cluster_center_op_ids)],
                                                                        task.labels, cv=folds, scoring=scorer)

                        left_out_task_acc.append( np.mean( score_this_task_cluster_ops ) ) # Mean of cv accuracy
                        reduced[task_ind][clust_ind] = len(self.good_perf_cluster_center_op_ids)
                        #cv_acc = np.append(cv_acc, score_this_task_cluster_ops) # append all cv accuracies -- 2D array
                    result_mat.append(left_out_task_acc) # no of clusters (y-axis) x no of tasks (x-axis)
                    
            np.save('result_mat.npy', result_mat) # save
            np.save('reduced.npy', reduced) # save
        else: 
            # Load the computed data
            result_mat = np.load('result_mat.npy') # load
            reduced = np.load('reduced.npy') # load

        # SCATTER PLOT:
        if (which_plot == 'scatter') or True:
            colors = cm.rainbow(np.linspace(0, 1, len(n_topOps_array) ))
            mpl.pyplot.figure(figsize=(5,7))
            for i in range(len(n_topOps_array)): # different colors
                # scatter plot
                print(reduced.shape)
                print(result_mat.shape)
                mpl.pyplot.scatter(reduced[i,:], result_mat[:,i], color = colors[i] , alpha = 0.8 )
                # error bar
                #mpl.pyplot.errorbar(reduced[i,:], result_mat[i,:], yerr = std_result_mat[i,:], fmt = 'o')
            mpl.pyplot.legend(labels=n_topOps_array.astype('int32') )
            mpl.pyplot.xlabel('no. of features')
            mpl.pyplot.ylabel('validated accuracy')
            mpl.pyplot.savefig("{}/scatter.svg".format(PARAMS['figure_dir']),dpi=400, bbox_inches='tight', pad_inches=0, transparent=True)
            # mpl.pyplot.show()
        if (which_plot == 'datamat') or True:
            
            # DATA PLOT
            result_mat = np.array(result_mat).T
            sorted_ind = result_mat[:,1].argsort()[::-1]
            result_mat = result_mat[sorted_ind]
            mpl.pyplot.figure(figsize=(5,7))
            mpl.pyplot.matshow( result_mat, cmap = 'OrRd', fignum=None)
            # mpl.pyplot.imshow(result_mat,cmap='OrRd')
            #mpl.pyplot.colorbar()
            clb = mpl.pyplot.colorbar(shrink=0.8)
            clb.ax.set_title('accuracy') # before = 'avg task accuracy'
            mpl.pyplot.xticks(np.arange(0,len(n_clust_array)), n_clust_array, rotation = 45 ) 
            mpl.pyplot.yticks(np.arange(0,len(self.tasks)), np.array(self.task_names)[sorted_ind] )
            # -- Loop over data dimensions and create text annotations
            for i in range(len(self.tasks)):
                for j in range(len(n_clust_array)):
                    xpos,ypos,textval,colorval = j, i, int(reduced[i, j]),float(result_mat[i, j])
                    if colorval<=0.425:
                        text = mpl.pyplot.text(xpos,ypos,textval, ha="center", va="center", color="k")
                    else:           
                        text = mpl.pyplot.text(xpos,ypos,textval, ha="center", va="center", color="w")
            mpl.pyplot.title('Left-out-task performance matrix (Top 100 features taken)\n', fontsize= 16, horizontalalignment='center')
            mpl.pyplot.xlabel('\nthreshold applied')
            mpl.pyplot.ylabel('Left-out-task')
            #mpl.pyplot.show()
            mpl.pyplot.savefig("{}/datamat.svg".format(PARAMS['figure_dir']),dpi=400, bbox_inches='tight', pad_inches=0, transparent=True)
            
        if 'plot'==PARAMS['complete_average_logic']:
            try:
                average_performance = np.loadtxt("peformance_mat_fullMeanStd_topMeanStd_clusterMeanStd_givenSplit_nonBalanced_new710_average.txt")
                complete_performance = np.loadtxt("peformance_mat_fullMeanStd_topMeanStd_clusterMeanStd_givenSplit_nonBalanced_new710_complete.txt")

                mpl.pyplot.figure()
                mpl.pyplot.plot((0, 1.5), (0, 1.5), '--', color=np.array((1, 1, 1)) * 0.7)
                mpl.pyplot.errorbar(average_performance[:, 4], complete_performance[:, 4],
                                                xerr=average_performance[:, 5], yerr=complete_performance[:, 5], fmt='o',
                                                color='r', ecolor='r')

                mpl.pyplot.xlabel('Accuracy of catchaMouse16')
                mpl.pyplot.ylabel('Accuracy of complete linkage centroids') # catchaMouse16
                print("Mean performance of the average clustering: {}".format(np.mean(average_performance[:, 4])))
                print("Mean performance of the complete clustering: {}".format(np.mean(complete_performance[:, 4])))
                mpl.pyplot.savefig("{}/performance_comparison.svg".format(PARAMS['figure_dir']),dpi=400, bbox_inches='tight', pad_inches=0, transparent=True)
            except IOError:
                warnings.warn("Trying to compare average and complete clustering. Expected files not found. Have you run both the average and the complete clustering?",UserWarning)
                print("Ignoring this plot for this run... Check that the both performance_mat files exist")
        if True:  # Originally Else: 
            
            # DISTRIBUTION PLOT ('dist')
            import seaborn as sns
            result_mat = np.array(result_mat)#.T
            print(result_mat.shape)
            colors = cm.rainbow(np.linspace(0, 1, len(n_clust_array) ))
            mpl.pyplot.figure(figsize=(7,5))
            sns.violinplot(data=result_mat, color ="0.8")
            sns.stripplot(data=result_mat, jitter=True, zorder=1)
            mpl.pyplot.title("'{}' Distribution Plot".format(PARAMS['linkage_method']))
            mpl.pyplot.xticks(range(n_clust_steps),n_clust_array)
            mpl.pyplot.xlabel('thresholds')
            mpl.pyplot.ylabel('Accuracy')
            #mpl.pyplot.show()
            mpl.pyplot.savefig("{}/dist.svg".format(PARAMS['figure_dir']),dpi=400, bbox_inches='tight', pad_inches=0.25, transparent=True)

if __name__ == '__main__':
    basePath = locations.rootDir() + '/'

    # select runtype by analysis done:
    # - null for repeated runs with shuffled labels
    # - otherwise only one run with valid labels
    # classifier type
    # - dectree,
    # - svm, or
    # - linear
    if 'dectree' in PARAMS['runtype']:
        if 'null' in PARAMS['runtype']:
            ranking_method = Feature_Stats.Null_Decision_Tree()
        else:
            null_folder = basePath + 'results/intermediate_results_' + PARAMS['runtype'] + '_null/'
            if not os.path.exists(null_folder):
                os.makedirs(null_folder)
            null_pattern = null_folder + 'task_{:s}_tot_stats_all_runs.txt'
            ranking_method = Feature_Stats.Decision_Tree(null_pattern)
    elif 'svm' in PARAMS['runtype']:
        if 'null' in PARAMS['runtype']:
            ranking_method = Feature_Stats.Null_Linear_Classifier()
        else:
            ranking_method = Feature_Stats.Linear_Classifier()
    else:
        ranking_method = Feature_Stats.Linear_Classifier()
        PARAMS['runtype'] = PARAMS['runtype'] + '_dectree'
        raise Warning('classifier not specified! Using dectree')

    # First check if hpc input directory exists, otherwise use local one
    inputDir = basePath + 'input_data/'+datatype+'/'

    intermediateResultsDir = basePath + 'results/intermediate_results_' + PARAMS['runtype'] + '/'

    # create directories if not there
    outputDir = basePath + 'output/' + PARAMS['runtype'] + '/'
    if not os.path.exists(inputDir):
        os.makedirs(inputDir)
    if not os.path.exists(intermediateResultsDir):
        os.makedirs(intermediateResultsDir)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    print "runtype = {}, datatype = {}, inputDir = {}".format(PARAMS['runtype'],datatype,inputDir)
    #path_pattern = inputDir + 'HCTSA_{:s}_N.mat'
    old_matlab = False
    label_regex_pattern = '(?:[^\,]*\,){0}([^,]*)'  # FIRST VALUE

    path_pattern_task_attrib = intermediateResultsDir + 'task_{:s}_{:s}'
    plot_out_path = outputDir + 'out_figure.eps'
    result_txt_outpath = outputDir + 'result_txt.txt'
    masking_method = 'NaN'

    # other parameters for feature classification accuracy normalisation, combination, selection
    select_good_perf_ops_norm = 'mean-norm' # 'zscore' # 'median-diff' # 'none' #
    select_good_perf_ops_method = 'sort_asc'
    select_good_perf_ops_combination = 'mean' # 'pos_sum' #
    similarity_method = 'corr'#'abscorr' # 'abscorr', 'corr', 'cos', 'euc'
    compare_space = 'problem_stats'
    min_calc_tasks = np.ceil(float(len(PARAMS['task_names'])) * 0.8) # np.ceil(float(len(task_names)) / float(1.25))

    # -----------------------------------------------------------------
    # -- Initialise Class instances -----------------------------------
    # -----------------------------------------------------------------

    input_method = Data_Input.Datafile_Input(inputDir,masking_method,label_regex_pattern)
    redundancy_method = Reducing_Redundancy.Reducing_Redundancy(similarity_method = similarity_method,compare_space = compare_space)

    workflow = Workflow(PARAMS['task_names'],input_method,ranking_method,
                        # combine_tasks_method = combine_tasks_method,combine_tasks_norm = combine_tasks_norm,
                        select_good_perf_ops_method = select_good_perf_ops_method, select_good_perf_ops_norm = select_good_perf_ops_norm,
                        select_good_perf_ops_combination=select_good_perf_ops_combination, redundancy_method = redundancy_method,
                        n_good_perf_ops = PARAMS['n_good_perf_ops'])

    # -----------------------------------------------------------------
    # -- Do the statistic calculations --------------------------------
    # -----------------------------------------------------------------

    # -- calculate the statistics
    if PARAMS['compute_features']:
        workflow.read_data(old_matlab=old_matlab)
        workflow.calculate_stats('tot_stats', path_pattern_task_attrib)
    else:
        # workflow.read_data(is_read_feature_data=False, old_matlab=old_matlab) # only performances, no feature-outputs
        workflow.read_data(old_matlab=old_matlab) # read full data for the classification comparison in the end.
        workflow.load_task_attribute('tot_stats', path_pattern_task_attrib)

    print 'loaded everything'

    # -- find the features which are calculated for at least min_calc_tasks tasks
    workflow.find_good_op_ids(min_calc_tasks)
    # -- exclude operations with 'raw' keyword
    workflow.exclude_good_ops_keyword('raw')
    # -- Collect all combined stats for each task and take stats for good (non-NaN) operations only
    workflow.collect_stats_good_op_ids(leave_task = -1) # leave no task
    # -- Select a subset of well performing operations (z-score across tasks, take n best)
    workflow.select_good_perf_ops()
    # workflow.select_good_perf_ops_sort_asc_input_params_to_file()
    #print ( ( workflow.stats_good_op_comb - np.nanmean(workflow.stats_good_op_comb) ) / np.nanstd(workflow.stats_good_op_comb) ) 
    # workflow.test()
    print( workflow.stats_good_op_comb.shape )
    print(len(workflow.tasks))
    #workflow.testing_parameters()
    #quit()

    # # -- IMPORTANT depends on knowledge on nulls
    # # -- mask p-values of operations with too few outputs
    # workflow.mask_pvals_too_few_unique_nulls()
    # # -- Combine the p-values of all the tasks
    # workflow.combine_task_pvals(min_p=0.05)
    # # -- Correct for multiple hyptothesis testing
    # workflow.correct_pvals_multiple_testing()
    # # # -- Select a subset of well performing operations (p-value-based)
    # # workflow.select_good_pval_ops_sort_asc()

    # -----------------------------------------------------------------
    # -- Do the redundancy calculations -------------------------------
    # -----------------------------------------------------------------
    # -- intitialise the redundancy method with the calculated results
    workflow.init_redundancy_method_problem_space()
    # -- calculate the correlation matrix saved in workflow.redundancy_method.similarity_array
    workflow.redundancy_method.calc_similarity()
    # -- calculate the linkage, the cluster indices and the clustering in self.corr_linkage,self.cluster_inds,self.cluster_op_id_list,respectively
    workflow.redundancy_method.calc_hierch_cluster(t = PARAMS['max_dist_cluster'])
    #
    # # -- single features for each cluster
    workflow.select_good_perf_cluster_center_ops()
    #quit()
    # -----------------------------------------------------------------
    # -- Classification perf with feature subsets ---------------------
    # -----------------------------------------------------------------

    # workflow.UMAP_all_topOps_clusters()
    # workflow.classify_good_perf_ops_vs_good_ops()
    workflow.classify_good_perf_ops_vs_super_vs_good_ops()
    # workflow.classify_N_clusters()
    # workflow.classify_good_perf_ops_vs_good_ops_givenSplit()
    # workflow.classify_selected_ops([0011, 0012, 0134, 0135, 0241, 1121, 7543, 3477, 1406, 1585, 1965, 0310, 2997, 3264, 3294, 4492, 3467, 3604, 4036, 4156, 4421, 3010])
    # workflow.classify_selectedOps_givenSplit()
    # workflow.classify_selected_ops_internalSet()
    # workflow.greedy_selectedOps()

    #workflow.classify_random_features()
    
    # -----------------------------------------------------------------
    # -- Testing param of Hierarchical clustering algorithm  ----------
    # -----------------------------------------------------------------
    # quit()
    workflow.testing_parameters()
    
    # quit()

    # -- show performance matrix of catch22-features only
    # workflow.show_catch22_perfmat()
    #quit()

    # -----------------------------------------------------------------
    # -- Do the plotting ----------------------------------------------
    # -----------------------------------------------------------------
    # -- initialise the plotting class

    plotting = Plotting.Plotting(workflow,max_dist_cluster = PARAMS['max_dist_cluster'])

    if False:
        # -- Plot the statistics array
        plotting.plot_stat_array()
    else:
        # -- Plot the similarity array
        plotting.plot_similarity_array()

    # mpl.pyplot.savefig(plot_out_path)

    # -----------------------------------------------------------------
    # -- Output the results to text file-------------------------------
    # -----------------------------------------------------------------
    op_id_name_map = plotting.map_op_id_name_mult_task(workflow.tasks)
    # -- write not reduced top performing features indicating the respective clusters they belong to
    # -- number of problems for which each good performing feature has been calculated
    measures = np.zeros((3,len(workflow.good_op_ids)))
    # -- op_ids
    measures[0,:] =  workflow.good_op_ids
    print(measures[0,:].shape)
    # -- number of problems calculated
    measures[1,:] = (~workflow.stats_good_op.mask).sum(axis=0)
    # -- z scored u-stat
    # measures[2,:] = fap.normalise_masked_array(workflow.stats_good_op_comb, axis= 0,norm_type = 'zscore')[0]
    # meanErrors = np.nanmean(np.ma.filled(workflow.stats_good_op, np.nan),0)
    # measures[2, :] = (meanErrors - np.nanmean(meanErrors))/np.nanstd(meanErrors)
    measures[2, :] = (workflow.stats_good_op_comb - np.nanmean(workflow.stats_good_op_comb))/np.nanstd(workflow.stats_good_op_comb)
    # # -- the normalised combined error as selected
    # measures[2, :] = -workflow.stats_good_op_comb

    # -- write textfile containing the information as shown in the plot
    workflow.redundancy_method.write_cluster_file(result_txt_outpath,op_id_name_map,measures)


    # -----------------------------------------------------------------
    # -- show the plot as last task of the script
    # -----------------------------------------------------------------
    mpl.pyplot.savefig("{}/cluster.svg".format(PARAMS['figure_dir']),dpi=400, bbox_inches='tight', pad_inches=0.25, transparent=True)
    #mpl.pyplot.show()
