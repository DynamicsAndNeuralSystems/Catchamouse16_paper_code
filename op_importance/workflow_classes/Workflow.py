import numpy as np
import matplotlib as mpl
#mpl.use("Agg")
import Task
import Data_Input
import Feature_Stats
import Reducing_Redundancy
import Plotting
import os
import sys

import collections
import modules.misc.PK_helper as hlp
import modules.feature_importance.PK_feat_array_proc as fap

import scipy.stats
import statsmodels
import statsmodels.sandbox.stats.multicomp

from pathos.multiprocessing import ThreadPool as Pool # ProcessingPool

class Workflow:

    def __init__(self,task_names,input_method,stats_method,redundancy_method,combine_tasks_method = 'mean',
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

    def collect_stats_good_op_ids(self):
        """
        Collect all combined stats for each task and take stats for good operations only
        """
        #stats_good_op_ma = np.empty((data.shape[0],np.array(self.good_op_ids).shape[0]))
        stats_good_op_tmp = []
        pvals_good_op_tmp = []
        #stats_good_op_ma[:] = np.NaN
        for task in self.tasks:
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
        stats_out = statsmodels.sandbox.stats.multicomp.multipletests(self.pvals_good_op_comb, alpha=0.05, method='h')

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

        self.op_ids = np.unique(op_ids_tasks)

        c = collections.Counter(op_ids_tasks)
        for key in c.keys():
            if c[key] >= threshold:
                self.good_op_ids.append(key)
        self.good_op_ids = np.array(self.good_op_ids)

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
        with open('/Users/carl/PycharmProjects/op_importance/bad_operations_and_tasks_they_fail_on.txt', 'w') as f:
            for item in sorted(bad_op_dict.items(), key=lambda t: len(t[1]), reverse=True):
                f.write("%s: %i bad tasks\n" % (item[0], len(item[1])))
                for bad_task in item[1]:
                        f.write("%s," % bad_task)
                f.write("\n\n")


    def create_op_keyword_and_name_dict(self):

        op_keyword_dict = {}
        op_name_dict = {}
        op_master_id_dict = {}

        m_op_name_dict = {}
        m_op_op_id_array = np.array([[], []]).T

        # some tasks don't cover all operations. To make sure you get all ops, iterate through tasks to create a
        # complete list of operation ids, names and
        for task in self.tasks:

            # read operation props
            op_master_id_this_task = np.array(task.op['master_id'])
            op_id_this_task = np.array(task.op['id'])
            op_names_this_task = np.array(task.op['code_string'])
            op_keywords_this_task = np.array(task.op['keywords'])

            # create a continuous array of op, m_op pairs to know for each m_op, what are the ops, not only the other
            # way around
            m_op_op_id_array = np.row_stack((m_op_op_id_array, np.column_stack((op_master_id_this_task, op_id_this_task))))

            # create a local dictionary to combine to a global one
            op_keyword_dict_this_task = {id : op_keywords_this_task[ind] for ind, id in enumerate(op_id_this_task)}
            op_name_dict_this_task = {id: op_names_this_task[ind] for ind, id in enumerate(op_id_this_task)}
            op_master_id_dict_this_task = {id: op_master_id_this_task[ind] for ind, id in enumerate(op_id_this_task)}

            # combine dictionaries across tasks
            op_keyword_dict = dict(op_keyword_dict, **op_keyword_dict_this_task)
            op_name_dict = dict(op_name_dict, **op_name_dict_this_task)
            op_master_id_dict = dict(op_master_id_dict, **op_master_id_dict_this_task)

            # read master operation dictionary to associate master_id with a name
            m_op_this_task = np.array(task.m_op['id'])
            m_op_name_this_task = np.array(task.m_op['name'])

            m_op_name_dict_this_task = {id: m_op_name_this_task[ind] for ind, id in enumerate(m_op_this_task)}

            m_op_name_dict = dict(m_op_name_dict, **m_op_name_dict_this_task)


        self.op_name_dict = op_name_dict
        self.op_keyword_dict = op_keyword_dict
        self.op_master_id_dict = op_master_id_dict

        self.m_op_name_dict = m_op_name_dict
        self.m_op_op_id_array = np.unique(m_op_op_id_array, axis=0)

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
        b = [self.op_name_dict[op_id] for op_id in self.good_op_ids]

        sortedOpInds = np.lexsort((b, a))[::-1]  # Sort by a then by b

        # sortedOpInds = np.argsort(self.pvals_good_op_comb)[::-1]
        ind = 0
        while self.pvals_good_op_comb[sortedOpInds[ind]] > 0.01:
            print '%1.5f %s' % (self.pvals_good_op_comb[sortedOpInds[ind]], self.op_name_dict[self.good_op_ids[sortedOpInds[ind]]])
            ind += 1

        # get the master op ids for those two failing classes
        bad_op_m_op_ids = [self.op_master_id_dict[op_id] for op_id in bad_op_ids]
        non_sign_op_m_op_ids = [self.op_master_id_dict[op_id] for op_id in non_sign_op_ids]

        # count master op ids in whole data set and bad and insignificant
        m_id_count_all = collections.Counter(self.op_master_id_dict.values())
        m_id_count_bad = collections.Counter(bad_op_m_op_ids)
        m_id_count_non_sign = collections.Counter(non_sign_op_m_op_ids)

        # calculate the share of not cool features per master operation id
        m_id_bad_share = {id: float(m_id_count_bad[id]) / m_id_count_all[id] for id in m_id_count_bad.keys()}
        m_id_non_sign_share = {id: float(m_id_count_non_sign[id]) / m_id_count_all[id] for id in m_id_count_non_sign.keys()}

        # sort and print
        print 'Bad master operations'
        for w in sorted(m_id_bad_share, key=m_id_bad_share.get, reverse=True):
            print '(%i/%i) %1.3f %s' % (m_id_count_bad[w], m_id_count_all[w], m_id_bad_share[w], self.m_op_name_dict[w])
            # print '(', m_id_count_bad[w], '/', m_id_count_all[w], ')', m_id_bad_share[w], ', ', self.m_op_name_dict[w]

        print '\nInsignificant master operations'
        for w in sorted(m_id_non_sign_share, key=m_id_non_sign_share.get, reverse=True):
            print '(%i/%i) %1.3f %s' % (m_id_count_non_sign[w], m_id_count_all[w], m_id_non_sign_share[w], self.m_op_name_dict[w])

            # # print the single operations, too
            # op_ids_this_m_op = self.m_op_op_id_array[self.m_op_op_id_array[:,0]==w,1]
            # for op_id_this_m_op in op_ids_this_m_op:
            #     print '%s' % self.op_name_dict[op_id_this_m_op]
            # print '(', m_id_count_non_sign[w], '/', m_id_count_all[w], ')', m_id_non_sign_share[w], ', ', self.m_op_name_dict[w]


        # # write to file
        # with open('/Users/carl/PycharmProjects/op_importance/bad_operations_and_tasks_they_fail_on.txt', 'w') as f:
        #     for item in sorted(bad_op_dict.items(), key=lambda t: len(t[1]), reverse=True):
        #         f.write("%s: %i bad tasks\n" % (item[0], len(item[1])))
        #         for bad_task in item[1]:
        #                 f.write("%s," % bad_task)
        #         f.write("\n\n")

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

    def select_good_perf_ops_sort_asc(self):
        """
        Select a subset of well performing operations
        """

        if self.select_good_perf_ops_norm in ['z-score','zscore'] :
            all_classes_good_norm = fap.normalise_masked_array(self.stats_good_op,axis = 1,norm_type = 'zscore')[0]

        elif self.select_good_perf_ops_norm == 'mean-norm':
            all_classes_good_mean = np.ma.masked_invalid(np.ma.mean(self.stats_good_op,axis = 1))
            all_classes_good_norm = (self.stats_good_op.T / all_classes_good_mean).T

        elif self.select_good_perf_ops_norm == 'median-diff':
            all_classes_good_norm = fap.normalise_masked_array(self.stats_good_op, axis=1, norm_type='median-diff')[0]
        else:
            all_classes_good_norm =  self.stats_good_op

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

        sort_ind_tmp = np.argsort(all_classes_norm_comb)

        if self.n_good_perf_ops == None:
            self.stats_good_perf_op_comb  = self.stats_good_op_comb[sort_ind_tmp]
            self.good_perf_op_ids =  self.good_op_ids[sort_ind_tmp]
        else:
            self.stats_good_perf_op_comb  = self.stats_good_op_comb[sort_ind_tmp][:self.n_good_perf_ops]
            self.good_perf_op_ids =  self.good_op_ids[sort_ind_tmp][:self.n_good_perf_ops]

    def select_good_perf_ops_sort_asc_input_params_to_file(self, norm='zscore', comb='mean'):
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
            all_classes_good_norm_pos[all_classes_good_norm_pos > 0] = 0 # it's error, so >0 -> bad
            stats_good_op_comb = all_classes_good_norm_pos.sum(axis=0)
        elif comb == 'max':
            stats_good_op_comb = all_classes_good_norm.max(axis=0)
        elif comb == 'min':
            stats_good_op_comb = all_classes_good_norm.max(axis=0)
        else:
            raise NameError('No valid performance combination identifier.')

        # sort combined
        sort_ind_tmp = np.argsort(stats_good_op_comb)

        # sort according to performance
        stats_good_perf_op_comb  = stats_good_op_comb[sort_ind_tmp]
        good_perf_op_ids =  self.good_op_ids[sort_ind_tmp]

        # write to file
        filename = 'topTops_' + norm + '_' + comb  + '.txt'

        np.savetxt('/Users/carl/PycharmProjects/op_importance/topOps/' + filename, np.column_stack((good_perf_op_ids, stats_good_op_comb)))

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
        mpl.pyplot.savefig('/Users/carl/PycharmProjects/op_importance/results/figures/norm='+norm+' comb='+comb+'.png')
        mpl.pyplot.close(f)


    def plot_null_distributions(self, p_min):

        from scipy import stats
        from matplotlib.pyplot import cm

        # get all null stats
        null_stats_all_tasks = []
        print 'loading null stats'
        for task_ind, task in enumerate(self.tasks):
            null_stats_all_tasks.append(self.stats_method.get_null_stats(task.name))
            print 'null stats for task %s loaded. (%i/%i)' % (task.name, task_ind, len(self.tasks))

        # print non significant ops
        sortedOpInds = np.argsort(self.pvals_good_op_comb)[::-1]
        ind = 0
        while self.pvals_good_op_comb[sortedOpInds[ind]] >= p_min:
            titleString = '%1.5f %s' % (
            self.pvals_good_op_comb[sortedOpInds[ind]], self.op_name_dict[self.good_op_ids[sortedOpInds[ind]]])

            pvals_all_tasks = self.pvals_good_op[:,sortedOpInds[ind]]
            sortedTaskInds = np.argsort(pvals_all_tasks)[::-1]
            for taskInd in sortedTaskInds:
                print '%1.5f %s' % (pvals_all_tasks[taskInd], self.task_names[taskInd])

            f, ax = mpl.pyplot.subplots(1)
            color = iter(cm.rainbow(np.linspace(0, 1, len(self.tasks))))

            for task_ind, task in enumerate(self.tasks):

                # if pvals_all_tasks[task_ind] < 0.9:
                #     continue

                null_stats = null_stats_all_tasks[task_ind]

                null_stats_this_op = null_stats[task.op_ids == self.good_op_ids[sortedOpInds[ind]],:]

                c = next(color)
                mpl.pyplot.hist(np.squeeze(null_stats_this_op), color=c)

                # density = stats.kde.gaussian_kde(null_stats_good_ops[~np.isnan(null_stats_good_ops)])
                # x = np.arange(0., 1, .001)
                # mpl.pyplot.plot(x, density(x), color=c)

                stats_this_op = self.stats_good_op[task_ind, sortedOpInds[ind]]
                ax.axvline(stats_this_op, color=c)

                print 'hm'

            #
            # for task_ind, task in enumerate(self.tasks):
            #     null_stats = self.stats_method.get_null_stats(task.name)
            #
            #     null_stats_good_ops = null_stats[np.isin(task.op_ids, self.good_op_ids),:]
            #
            #     density = stats.kde.gaussian_kde(null_stats_good_ops[~np.isnan(null_stats_good_ops)])
            #     x = np.arange(0., 1, .001)
            #
            #     c = next(color)
            #
            #     mpl.pyplot.plot(x, density(x), color=c)
            #
            #     stats_this_op = self.stats_good_op[task_ind, sortedOpInds[ind]]
            #     ax.axvline(stats_this_op, color=c)
            #
            # mpl.pyplot.xlabel('error')
            # mpl.pyplot.ylabel('frequency')



            ind += 1



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

        if self.select_good_perf_ops_norm in ['z-score','zscore'] :
            all_classes_good_norm = fap.normalise_masked_array(self.stats_good_op,axis = 1,norm_type = 'zscore')[0]

        elif self.select_good_perf_ops_norm == 'mean-norm':
            all_classes_good_mean = np.ma.masked_invalid(np.ma.mean(self.stats_good_op,axis = 1))
            all_classes_good_norm = (self.stats_good_op.T / all_classes_good_mean).T

        else:
            all_classes_good_norm =  self.stats_good_op

        # sort operations once, then filter this index list to contain only indices of the cluster at hand
        sorted_op_ids = self.good_op_ids[np.argsort(all_classes_good_norm.mean(axis=0))]

        cluster_center_op_ids = []
        for i, cluster_op_ids in enumerate(self.redundancy_method.cluster_op_id_list):

            cluster_center_op_ids.append(sorted_op_ids[np.isin(sorted_op_ids, cluster_op_ids)][0])

        self.good_perf_cluster_center_op_ids  = np.array(cluster_center_op_ids)

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
            score_this_task_top_ops = cross_val_score(clf, task.data[:, np.isin(task.op_ids, self.good_perf_op_ids)], task.labels, cv=folds) # , scoring=scorer)

            # only cluster centers
            score_this_task_cluster_ops = cross_val_score(clf, task.data[:, np.isin(task.op_ids, self.good_perf_cluster_center_op_ids)],
                                                      task.labels, cv=folds) # , scoring=scorer)

            # whole matrix
            score_this_task_whole = cross_val_score(clf, task.data, task.labels, cv=folds) # , scoring=scorer)

            # plot immediately
            p1 = mpl.pyplot.errorbar(np.mean(score_this_task_whole), np.mean(score_this_task_top_ops),
                                xerr=np.std(score_this_task_whole), yerr=np.std(score_this_task_top_ops), fmt='o', color='b', ecolor='b')
            p2 = mpl.pyplot.errorbar(np.mean(score_this_task_whole), np.mean(score_this_task_cluster_ops),
                                xerr=np.std(score_this_task_whole), yerr=np.std(score_this_task_cluster_ops), fmt='o',
                                color='r', ecolor='r')

            # save scores
            perfmat[task_ind, 0] = np.mean(score_this_task_whole)
            perfmat[task_ind, 1] = np.std(score_this_task_whole)
            perfmat[task_ind, 2] = np.mean(score_this_task_top_ops)
            perfmat[task_ind, 3] = np.std(score_this_task_top_ops)
            perfmat[task_ind, 4] = np.mean(score_this_task_cluster_ops)
            perfmat[task_ind, 5] = np.std(score_this_task_cluster_ops)

            print 'Done. Took %1.1f minutes.' % ((time.time() - t)/60)

            np.savetxt('/Users/carl/PycharmProjects/op_importance/peformance_mat_fullMeanStd_topMeanStd_clusterMeanStd_pos_sum.txt', perfmat)

        mpl.pyplot.legend((p1, p2), ('500 top ops', 'only cluster centers'))
        # mpl.pyplot.xlim((0, 1))
        # mpl.pyplot.ylim((0, 1))
        mpl.pyplot.xlabel('performance on whole feature set')
        mpl.pyplot.ylabel('performance only selected features')
        mpl.pyplot.ylabel('class imbalance corrected performance')
        mpl.pyplot.show()

if __name__ == '__main__':

    # -----------------------------------------------------------------
    # -- Set Parameters -----------------------------------------------
    # -----------------------------------------------------------------
    if len(sys.argv) > 1:
        runtype = sys.argv[1]
    else:
        runtype = 'dectree_maxmin' # 'dectree_maxmin'

    if len(sys.argv) > 2:
        task_names = sys.argv[2].split(",")
    else:
        # task_names = ["50words", "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "CBF", "Car",
        #               "ChlorineConcentration", "CinC_ECG_torso", "Coffee", "Computers", "Cricket_X", "Cricket_Y",
        #               "Cricket_Z", "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect",
        #               "DistalPhalanxTW", "ECG200", "ECG5000", "ECGFiveDays", "Earthquakes", "ElectricDevices", "FISH",
        #               "FaceAll", "FaceFour", "FacesUCR", "FordA", "FordB", "Gun_Point", "Ham", "HandOutlines",
        #               "Haptics", "Herring", "InlineSkate", "InsectWingbeatSound", "ItalyPowerDemand",
        #               "LargeKitchenAppliances", "Lighting2", "Lighting7", "MALLAT", "Meat", "MedicalImages",
        #               "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MoteStrain",
        #               "NonInvasiveFatalECG_Thorax1", "NonInvasiveFatalECG_Thorax2", "OSULeaf", "OliveOil",
        #               "PhalangesOutlinesCorrect", "Phoneme", "Plane", "ProximalPhalanxOutlineAgeGroup",
        #               "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "RefrigerationDevices", "ScreenType",
        #               "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SonyAIBORobotSurface",
        #               "SonyAIBORobotSurfaceII", "StarLightCurves", "Strawberry", "SwedishLeaf", "Symbols",
        #               "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG", "Two_Patterns",
        #               "UWaveGestureLibraryAll", "Wine", "WordsSynonyms", "Worms", "WormsTwoClass", "synthetic_control",
        #               "uWaveGestureLibrary_X", "uWaveGestureLibrary_Y", "uWaveGestureLibrary_Z", "wafer", "yoga"]
        # task_names = ["Adiac", "ArrowHead", "Beef"] # ["BirdChicken", "50words"] # ["Wine","50words"]
        # PHILS TASKS: task_names = ['MedicalImages', 'Cricket_X', 'InlineSkate', 'ECG200', 'WordsSynonyms', 'uWaveGestureLibrary_X', 'Two_Patterns', 'yoga', 'Symbols', 'uWaveGestureLibrary_Z', 'SonyAIBORobotSurfaceII', 'Cricket_Y', 'Gun_Point', 'OliveOil', 'Lighting7', 'NonInvasiveFatalECG _Thorax1', 'Haptics', 'Adiac', 'ChlorineConcentration', 'synthetic_control', 'OSULeaf', 'DiatomSizeReduction', 'SonyAIBORobotSurface', 'MALLAT', 'uWaveGestureLibrary_Y', 'CBF', 'ECGFiveDays', 'Lighting2', 'FISH', 'FacesUCR', 'FaceFour', 'Trace', 'Coffee', '50words', 'MoteStrain', 'wafer', 'Cricket_Z', 'SwedishLeaf']
        # UCR 2018:
        task_names = ["AALTDChallenge", "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "CBF", "Car",
                      "ChlorineConcentration", "CinCECGtorso", "Coffee", "Computers", "CricketX", "CricketY", "CricketZ",
                      "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect",
                      "DistalPhalanxTW", "ECG200", "ECG5000", "ECGFiveDays", "ECGMeditation", "Earthquakes",
                      "ElectricDeviceOn", "ElectricDevices", "EpilepsyX", "EthanolLevel", "FaceAll", "FaceFour", "FacesUCR",
                      "FiftyWords", "Fish", "FordA", "FordB", "GunPoint", "Ham", "HandOutlines", "Haptics", "HeartbeatBIDMC",
                      "Herring", "InlineSkate", "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances",
                      "Lightning2", "Lightning7", "Mallat", "Meat", "MedicalImages", "MiddlePhalanxOutlineAgeGroup",
                      "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MoteStrain", "NonInvasiveFatalECGThorax1",
                      "NonInvasiveFatalECGThorax2", "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2", "OSULeaf",
                      "OliveOil", "PhalangesOutlinesCorrect", "Phoneme", "Plane", "ProximalPhalanxOutlineAgeGroup",
                      "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "RefrigerationDevices", "ScreenType",
                      "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SonyAIBORobotSurface1",
                      "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf", "Symbols",
                      "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG", "TwoPatterns",
                      "UWaveGestureLibraryAll", "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ",
                      "Wafer", "Wine", "WordSynonyms", "Worms", "WormsTwoClass", "Yoga"]
        # # selection of tasks as in old UCR
        # task_names = ["Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "CBF", "Car",
        #               "ChlorineConcentration", "CinCECGtorso", "Coffee", "Computers", "CricketX", "CricketY",
        #               "CricketZ",
        #               "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect",
        #               "DistalPhalanxTW", "ECG200", "ECG5000", "ECGFiveDays", "Earthquakes",
        #               "ElectricDevices", "FaceAll", "FaceFour",
        #               "FacesUCR",
        #               "FiftyWords", "Fish", "FordA", "FordB", "GunPoint", "Ham", "HandOutlines", "Haptics",
        #               "Herring", "InlineSkate", "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances",
        #               "Lightning2", "Lightning7", "Mallat", "Meat", "MedicalImages", "MiddlePhalanxOutlineAgeGroup",
        #               "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MoteStrain", "NonInvasiveFatalECGThorax1",
        #               "NonInvasiveFatalECGThorax2",
        #               "OSULeaf",
        #               "OliveOil", "PhalangesOutlinesCorrect", "Phoneme", "Plane", "ProximalPhalanxOutlineAgeGroup",
        #               "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "RefrigerationDevices", "ScreenType",
        #               "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SonyAIBORobotSurface1",
        #               "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf", "Symbols",
        #               "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG", "TwoPatterns",
        #               "UWaveGestureLibraryAll", "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ",
        #               "Wafer", "Wine", "WordSynonyms", "Worms", "WormsTwoClass", "Yoga"]

    n_good_perf_ops = 500
    compute_features = False # True #
    max_dist_cluster = 0.2

    if 'maxmin' in runtype:
        datatype = 'maxmin'
    elif 'scaledrobustsigmoid' in runtype:
        datatype = 'scaledrobustsigmoid'
    else:
        datatype = 'maxmin'
        runtype = runtype + '_maxmin'
        raise Warning('normalisation not specified! Using maxmin')

    basePath = '/Users/carl/PycharmProjects/op_importance/'

    if 'dectree' in runtype:
        if 'null' in runtype:
            ranking_method = Feature_Stats.Null_Decision_Tree()
        else:
            null_folder = basePath + 'results/intermediate_results_' + runtype + '_null/'
            if not os.path.exists(null_folder):
                os.makedirs(null_folder)
            null_pattern = null_folder + 'task_{:s}_tot_stats_all_runs.txt'
            ranking_method = Feature_Stats.Decision_Tree(null_pattern)
    elif 'svm' in runtype:
        if 'null' in runtype:
            ranking_method = Feature_Stats.Null_Linear_Classifier()
        else:
            ranking_method = Feature_Stats.Linear_Classifier()
    else:
        ranking_method = Feature_Stats.Linear_Classifier()
        runtype = runtype + '_dectree'
        raise Warning('classifier not specified! Using dectree')

    # First check if hpc input directory exists, otherwise use local one
    inputDir = basePath + 'input_data/'+datatype+'/'

    intermediateResultsDir = basePath + 'results/intermediate_results_' + runtype + '/'

    # outputDir = '../output/'+runtype+'/'
    outputDir = basePath + 'output/' + runtype + '/'
    if not os.path.exists(inputDir):
        os.makedirs(inputDir)
    if not os.path.exists(intermediateResultsDir):
        os.makedirs(intermediateResultsDir)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)


    print "runtype = {}, datatype = {}, inputDir = {}".format(runtype,datatype,inputDir)

    path_pattern = inputDir + 'HCTSA_{:s}_N.mat'
    old_matlab = False
    label_regex_pattern = '(?:[^\,]*\,){0}([^,]*)'  # FIRST VALUE

    #path_pattern = '../phil_matlab/HCTSA_{:s}_N_70_100_reduced.mat'
    #old_matlab = True
    #label_regex_pattern = label_regex_pattern = '.*,(.*)$' # LAST VALUE

    path_pattern_task_attrib = intermediateResultsDir + 'task_{:s}_{:s}'
    plot_out_path = outputDir + 'out_figure.png'
    result_txt_outpath = outputDir + 'result_txt.txt'
    masking_method = 'NaN'

    combine_pair_method = 'mean'
    combine_tasks_method = 'mean'
    combine_tasks_norm = None
    select_good_perf_ops_norm = 'mean-norm' # 'zscore' # 'median-diff' # 'none' #
    select_good_perf_ops_method = 'sort_asc'
    select_good_perf_ops_combination = 'mean' # 'pos_sum' #
    similarity_method = 'correlation'
    compare_space = 'problem_stats'
    min_calc_tasks = np.ceil(float(len(task_names)) * 0.8) # np.ceil(float(len(task_names)) / float(1.25))

    # -----------------------------------------------------------------
    # -- Initialise Class instances -----------------------------------
    # -----------------------------------------------------------------

    input_method = Data_Input.Datafile_Input(path_pattern,masking_method,label_regex_pattern)
    redundancy_method = Reducing_Redundancy.Reducing_Redundancy(similarity_method = similarity_method,compare_space = compare_space)

    workflow = Workflow(task_names,input_method,ranking_method,
                        combine_tasks_method = combine_tasks_method,combine_tasks_norm = combine_tasks_norm,
                        select_good_perf_ops_method = select_good_perf_ops_method, select_good_perf_ops_norm = select_good_perf_ops_norm,
                        select_good_perf_ops_combination=select_good_perf_ops_combination, redundancy_method = redundancy_method,
                        n_good_perf_ops = n_good_perf_ops)

    # -----------------------------------------------------------------
    # -- Do the statistic calculations --------------------------------
    # -----------------------------------------------------------------

    # -- calculate the statistics
    if compute_features:
        workflow.read_data(old_matlab=old_matlab)
        workflow.calculate_stats('tot_stats', path_pattern_task_attrib)
    else:
        workflow.read_data(is_read_feature_data=False, old_matlab=old_matlab)
        # # todo: restore to data loading without the matrix. This is just for the classification comparison in the end.
        # workflow.read_data(old_matlab=old_matlab)
        workflow.load_task_attribute('tot_stats', path_pattern_task_attrib)

    # -- find the features which are calculated for at least min_calc_tasks tasks
    workflow.find_good_op_ids(min_calc_tasks)
    # -- Collect all combined stats for each task and take stats for good (non-NaN) operations only
    workflow.collect_stats_good_op_ids()
    # -- Combine the stats of all the tasks (mean of classification error, not p-value)
    workflow.combine_tasks()
    # -- Select a subset of well performing operations (z-score across tasks, take n best)
    workflow.select_good_perf_ops()

    # # -- just for analysis, print out the operation ids and performances
    # for comb in ['mean', 'min', 'pos_sum']:
    #     for norm in ['z-score', 'mean-norm', 'median-diff', 'none']:
    #         # workflow.plot_perf_histograms(norm, comb)
    #         workflow.select_good_perf_ops_sort_asc_input_params_to_file(norm, comb)
    # # mpl.pyplot.show()

    # # plot the correlation between all (not only top) features
    # mpl.pyplot.imshow(np.ma.corrcoef(workflow.stats_good_op))
    # cb = mpl.pyplot.colorbar()
    # cb.set_label('Pearson correlation')
    # mpl.pyplot.yticks(np.arange(len(workflow.task_names)), workflow.task_names, fontsize=8)
    #
    # # mpl.pyplot.imshow(np.ma.corrcoef(workflow.stats_good_op.T))
    # # cb = mpl.pyplot.colorbar()
    # # cb.set_label('Pearson correlation')
    # mpl.pyplot.show()

    # # -- just for analysis, print which operations failed on what tasks
    # workflow.list_bad_op_ids_and_tasks()

    # # -- checking if distribution of classification errors are normal
    # from scipy import stats
    # #
    # # mpl.pyplot.figure()
    # # for i in range(np.size(workflow.stats_good_op, 0)-1):
    # #     data = workflow.stats_good_op[i,:]
    # #     density = stats.kde.gaussian_kde(data[~np.isnan(data)])
    # #     x = np.arange(0., 1, .001)
    # #     mpl.pyplot.plot(x, density(x))
    # # mpl.pyplot.xlabel('error')
    # # mpl.pyplot.ylabel('frequency')
    # # mpl.pyplot.title('non-normalised')
    # #
    # # all_classes_good_norm = fap.normalise_masked_array(workflow.stats_good_op, axis=1, norm_type='zscore')[0]
    # # mpl.pyplot.figure()
    # # for i in range(np.size(workflow.stats_good_op, 0) - 1):
    # #     data = all_classes_good_norm[i, :]
    # #     density = stats.kde.gaussian_kde(data[~np.isnan(data)])
    # #     x = np.arange(-4, 4, .001)
    # #     mpl.pyplot.plot(x, density(x))
    # # mpl.pyplot.xlabel('error')
    # # mpl.pyplot.ylabel('frequency')
    # # mpl.pyplot.title('z-scored')
    # #
    # # all_classes_good_mean = np.ma.masked_invalid(np.ma.mean(workflow.stats_good_op, axis=1))
    # # all_classes_good_norm = (workflow.stats_good_op.T / all_classes_good_mean).T
    # # mpl.pyplot.figure()
    # # for i in range(np.size(workflow.stats_good_op, 0) - 1):
    # #     data = all_classes_good_norm[i, :]
    # #     density = stats.kde.gaussian_kde(data[~np.isnan(data)])
    # #     x = np.arange(0., 10, .001)
    # #     mpl.pyplot.plot(x, density(x))
    # # mpl.pyplot.xlabel('error')
    # # mpl.pyplot.ylabel('frequency')
    # # mpl.pyplot.title('divided by mean')
    # #
    #
    # print 'started the distribution plot'
    # all_classes_good_norm = fap.normalise_masked_array(workflow.stats_good_op, axis=1, norm_type='zscore_median')[0]
    # mpl.pyplot.figure()
    # for i in range(np.size(workflow.stats_good_op, 0) - 1):
    #     data = all_classes_good_norm[i, :]
    #     density = stats.kde.gaussian_kde(data[~np.isnan(data)])
    #     x = np.arange(-4., 4, .001)
    #     mpl.pyplot.plot(x, density(x))
    # mpl.pyplot.xlabel('error')
    # mpl.pyplot.ylabel('frequency')
    # mpl.pyplot.title('zscore with median')
    #
    # all_classes_good_norm = fap.normalise_masked_array(workflow.stats_good_op, axis=1, norm_type='median-div')[0]
    # mpl.pyplot.figure()
    # for i in range(np.size(workflow.stats_good_op, 0) - 1):
    #     data = all_classes_good_norm[i, :]
    #     density = stats.kde.gaussian_kde(data[~np.isnan(data)])
    #     x = np.arange(-0., 3, .001)
    #     mpl.pyplot.plot(x, density(x))
    # mpl.pyplot.xlabel('error')
    # mpl.pyplot.ylabel('frequency')
    # mpl.pyplot.title('divided by median')
    #
    # all_classes_good_norm = fap.normalise_masked_array(workflow.stats_good_op, axis=1, norm_type='median-diff')[0]
    # mpl.pyplot.figure()
    # for i in range(np.size(workflow.stats_good_op, 0) - 1):
    #     data = all_classes_good_norm[i, :]
    #     density = stats.kde.gaussian_kde(data[~np.isnan(data)])
    #     x = np.arange(-3., 3, .001)
    #     mpl.pyplot.plot(x, density(x))
    # mpl.pyplot.xlabel('error')
    # mpl.pyplot.ylabel('frequency')
    # mpl.pyplot.title('median subtracted')

    # from scipy import stats
    # # all_classes_good_norm = fap.normalise_masked_array(workflow.stats_good_op, axis=1, norm_type='median-diff')[0]
    # mpl.pyplot.figure()
    # for i in range(np.size(workflow.stats_good_op, 0) - 1):
    #     data = workflow.stats_good_op[i, :]
    #     density = stats.kde.gaussian_kde(data[~np.isnan(data)])
    #     x = np.arange(-3., 3, .001)
    #     mpl.pyplot.plot(x, density(x))
    # mpl.pyplot.xlabel('error')
    # mpl.pyplot.ylabel('frequency')
    # mpl.pyplot.title('raw')
    #
    # mpl.pyplot.show()

    # -- Combine the p-values of all the tasks
    workflow.combine_task_pvals(min_p=0.001)
    # -- Correct for multiple hyptothesis testing
    workflow.correct_pvals_multiple_testing()
    # # -- Select a subset of well performing operations (p-value-based)
    # workflow.select_good_pval_ops_sort_asc()

    # # -- see how the different sets of operations overlap (p-value vs. mean error selection)
    # pval_good_perf_op_ids_set = set(workflow.pval_good_perf_op_ids)
    # good_perf_op_ids_set = set(workflow.good_perf_op_ids)
    # good_id_intersection = pval_good_perf_op_ids_set.intersection(good_perf_op_ids_set)
    # print "%i/%i overlap between p-value and mean performance selected ops." % (len(good_id_intersection), n_good_perf_ops)

    # # -- see if features classify inconsistently
    # all_classes_good_norm = fap.normalise_masked_array(workflow.stats_good_op, axis=1, norm_type='zscore')[0]
    # stdOfError = np.std(all_classes_good_norm, 0)
    # zscoredStdOfError = (stdOfError - np.mean(stdOfError)) / np.std(stdOfError)
    # inconsistent_op_ids = workflow.good_op_ids[zscoredStdOfError > 1]
    #
    # mpl.pyplot.hist(
    #     (np.std(all_classes_good_norm[:, np.isin(workflow.good_op_ids, workflow.good_perf_op_ids)], axis=0)),
    #     bins=50), mpl.pyplot.xlabel('std of normalised error for 500 good features (mean perf)'), mpl.pyplot.ylabel(
    #     'frequency')
    # mpl.pyplot.scatter(np.std(all_classes_good_norm, axis=0),
    #                    np.mean(all_classes_good_norm, axis=0)), \
    # mpl.pyplot.xlabel('std of normalised error')
    # mpl.pyplot.ylabel('mean of normalised error')
    # inconsistent_but_good_op_ids = pval_good_perf_op_ids_set.intersection(set(inconsistent_op_ids))

    # -- creat a dictionary of information about each feature to analyse the failing ones
    workflow.create_op_keyword_and_name_dict()
    workflow.list_bad_and_non_significant_ops()
    # workflow.plot_null_distributions(1)

    # -----------------------------------------------------------------
    # -- Do the redundancy calculations -------------------------------
    # -----------------------------------------------------------------

    # -- intitialise the redundancy method with the calculated results
    workflow.init_redundancy_method_problem_space()
    # -- calculate the correlation matrix saved in workflow.redundancy_method.similarity_array
    workflow.redundancy_method.calc_similarity()
    # -- calculate the linkage, the cluster indices and the clustering in self.corr_linkage,self.cluster_inds,self.cluster_op_id_list,respectively
    workflow.redundancy_method.calc_hierch_cluster(t = max_dist_cluster)

    # -- single features for each cluster
    workflow.select_good_perf_cluster_center_ops()

    # -----------------------------------------------------------------
    # -- Classification perf with feature subsets ---------------------
    # -----------------------------------------------------------------

    # workflow.classify_good_perf_ops_vs_good_ops()

    # -----------------------------------------------------------------
    # -- Do the plotting ----------------------------------------------
    # -----------------------------------------------------------------
    # -- initialise the plotting class
    plotting = Plotting.Plotting(workflow,max_dist_cluster = max_dist_cluster)

    if False:
        # -- Plot the statistics array
        plotting.plot_stat_array()
    else:
        # -- Plot the similarity array
        plotting.plot_similarity_array()

    mpl.pyplot.savefig(plot_out_path)

    # -----------------------------------------------------------------
    # -- Output the results to text file-------------------------------
    # -----------------------------------------------------------------
    op_id_name_map = plotting.map_op_id_name_mult_task(workflow.tasks)
    # -- write not reduced top performing features indicating the respective clusters they belong to
    # -- number of problems for which each good performing feature has been calculated
    measures = np.zeros((3,len(workflow.good_op_ids)))
    # -- op_ids
    measures[0,:] =  workflow.good_op_ids
    # -- number of problems calculated
    measures[1,:] = (~workflow.stats_good_op.mask).sum(axis=0)
    # -- z scored u-stat
    measures[2,:] = fap.normalise_masked_array(workflow.stats_good_op_comb, axis= 0,norm_type = 'zscore')[0]

    # -- write textfile containing the information as shown in the plot
    workflow.redundancy_method.write_cluster_file(result_txt_outpath,op_id_name_map,measures)


    # -----------------------------------------------------------------
    # -- show the plot as last task of the script
    # -----------------------------------------------------------------
    mpl.pyplot.show()

    # -- write not reduced top performing features to text file
#     with open(result_txt_outpath,'wb') as out_result_txt_file:
#         for op_id,op_name,op_U in zip(workflow.good_perf_op_ids,
#                                       hlp.ind_map_subset(op_id_name_map[0],op_id_name_map[1], workflow.good_perf_op_ids),
#                                       workflow.stats_good_perf_op_comb):
#             out_result_txt_file.write("{:d} {:s} {:f}\n".format(op_id,op_name,op_U))
