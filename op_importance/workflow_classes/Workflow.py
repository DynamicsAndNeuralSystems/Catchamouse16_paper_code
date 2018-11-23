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
        with open('/Users/carl/PycharmProjects/op_importance/bad_operations_and_tasks_they_fail_on.txt', 'w') as f:
            for item in sorted(bad_op_dict.items(), key=lambda t: len(t[1]), reverse=True):
                f.write("%s: %i bad tasks\n" % (item[0], len(item[1])))
                for bad_task in item[1]:
                        f.write("%s," % bad_task)
                f.write("\n\n")


    # def create_op_keyword_and_name_dict(self):
    #
    #     op_keyword_dict = {}
    #     op_name_dict = {}
    #     op_master_id_dict = {}
    #
    #     m_op_name_dict = {}
    #     m_op_op_id_array = np.array([[], []]).T
    #
    #     # some tasks don't cover all operations. To make sure you get all ops, iterate through tasks to create a
    #     # complete list of operation ids, names and
    #     for task in self.tasks:
    #
    #         # read operation props
    #         op_master_id_this_task = np.array(task.op['master_id'])
    #         op_id_this_task = np.array(task.op['id'])
    #         op_names_this_task = np.array(task.op['code_string'])
    #         op_keywords_this_task = np.array(task.op['keywords'])
    #
    #         # create a continuous array of op, m_op pairs to know for each m_op, what are the ops, not only the other
    #         # way around
    #         m_op_op_id_array = np.row_stack((m_op_op_id_array, np.column_stack((op_master_id_this_task, op_id_this_task))))
    #
    #         # create a local dictionary to combine to a global one
    #         op_keyword_dict_this_task = {ind : op_keywords_this_task[ind] for ind, id in enumerate(op_id_this_task)}
    #         op_name_dict_this_task = {ind: op_names_this_task[ind] for ind, id in enumerate(op_id_this_task)}
    #         op_master_id_dict_this_task = {ind: op_master_id_this_task[ind] for ind, id in enumerate(op_id_this_task)}
    #
    #         # combine dictionaries across tasks
    #         op_keyword_dict = dict(op_keyword_dict, **op_keyword_dict_this_task)
    #         op_name_dict = dict(op_name_dict, **op_name_dict_this_task)
    #         op_master_id_dict = dict(op_master_id_dict, **op_master_id_dict_this_task)
    #
    #         print 'After task %s with %i operations, we now have %i operations overall.\n' % (task.name, len(op_id_this_task), len(op_name_dict))
    #
    #         # read master operation dictionary to associate master_id with a name
    #         m_op_this_task = np.array(task.m_op['id'])
    #         m_op_name_this_task = np.array(task.m_op['name'])
    #
    #         m_op_name_dict_this_task = {id: m_op_name_this_task[ind] for ind, id in enumerate(m_op_this_task)}
    #
    #         m_op_name_dict = dict(m_op_name_dict, **m_op_name_dict_this_task)
    #
    #
    #     self.op_IDs_HCTSA = op_name_dict.keys()
    #     self.op_name_dict = op_name_dict
    #     self.op_keyword_dict = op_keyword_dict
    #     self.op_master_id_dict = op_master_id_dict
    #
    #     self.m_op_name_dict = m_op_name_dict
    #     self.m_op_op_id_array = np.unique(m_op_op_id_array, axis=0)

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

        # f, axarr = mpl.pyplot.subplots(2,2, sharex='col', sharey='row')
        #
        # ax = axarr[1][0]
        # ax.imshow(self.pvals_good_op[:, sortedOpInds[0:maxInd]], aspect='auto')
        # ax.set_yticks(np.arange(len(self.task_names)))
        # ax.set_yticklabels(self.task_names, fontdict={'fontsize':3})
        # ax.set_xticks(np.arange(maxInd))
        # ax.set_xticklabels(op_IDs_HCTSA[self.good_op_ids[sortedOpInds[0:maxInd]]], fontdict={'fontsize':6}, rotation='90')
        #
        # ax = axarr[0][0]
        # ax.plot(range(maxInd), self.pvals_good_op_comb[sortedOpInds[0:maxInd]], label='Fisher')
        # ax.errorbar(range(maxInd), np.ma.mean(self.pvals_good_op[:,sortedOpInds[0:maxInd]],0), np.ma.std(self.pvals_good_op[:,sortedOpInds[0:maxInd]],0), label='mean and std')
        # #ax.set_yscale('log')
        # ax.set_ylabel('p-Value')
        # ax.legend()
        # ax.set_title('least significant')
        #
        # nEnd = 100
        # ax = axarr[1][1]
        # ax.imshow(self.pvals_good_op[:, sortedOpInds[-nEnd-1:-1]], aspect='auto', )
        # ax.set_yticks(np.arange(len(self.task_names)))
        # ax.set_yticklabels(self.task_names, fontdict={'fontsize': 3})
        #
        # ax = axarr[0][1]
        # ax.plot(range(nEnd), self.pvals_good_op_comb[sortedOpInds[-nEnd-1:-1]], label='Fisher')
        # ax.errorbar(range(nEnd), np.ma.mean(self.pvals_good_op[:, sortedOpInds[-nEnd-1:-1]], 0),
        #             np.ma.std(self.pvals_good_op[:, sortedOpInds[-nEnd-1:-1]], 0), label='mean')
        # #ax.set_yscale('log')
        # #ax.set_yticks([])
        # ax.legend()
        # ax.set_title('most significant')
        #
        # mpl.pyplot.show()


        # # get the master op ids for those two failing classes
        # bad_op_m_op_ids = [self.op_master_id_dict[op_id] for op_id in bad_op_ids]
        # non_sign_op_m_op_ids = [self.op_master_id_dict[op_id] for op_id in non_sign_op_ids]
        #
        # # count master op ids in whole data set and bad and insignificant
        # m_id_count_all = collections.Counter(self.op_master_id_dict.values())
        # m_id_count_bad = collections.Counter(bad_op_m_op_ids)
        # m_id_count_non_sign = collections.Counter(non_sign_op_m_op_ids)
        #
        # # calculate the share of not cool features per master operation id
        # m_id_bad_share = {id: float(m_id_count_bad[id]) / m_id_count_all[id] for id in m_id_count_bad.keys()}
        # m_id_non_sign_share = {id: float(m_id_count_non_sign[id]) / m_id_count_all[id] for id in m_id_count_non_sign.keys()}
        #
        # # sort and print
        # print 'Bad master operations'
        # for w in sorted(m_id_bad_share, key=m_id_bad_share.get, reverse=True):
        #     print '(%i/%i) %1.3f %s' % (m_id_count_bad[w], m_id_count_all[w], m_id_bad_share[w], self.m_op_name_dict[w])
        #     # print '(', m_id_count_bad[w], '/', m_id_count_all[w], ')', m_id_bad_share[w], ', ', self.m_op_name_dict[w]
        #
        # print '\nInsignificant master operations'
        # for w in sorted(m_id_non_sign_share, key=m_id_non_sign_share.get, reverse=True):
        #     print '(%i/%i) %1.3f %s' % (m_id_count_non_sign[w], m_id_count_all[w], m_id_non_sign_share[w], self.m_op_name_dict[w])
        #
        #     # # print the single operations, too
        #     # op_ids_this_m_op = self.m_op_op_id_array[self.m_op_op_id_array[:,0]==w,1]
        #     # for op_id_this_m_op in op_ids_this_m_op:
        #     #     print '%s' % self.op_name_dict[op_id_this_m_op]
        #     # print '(', m_id_count_non_sign[w], '/', m_id_count_all[w], ')', m_id_non_sign_share[w], ', ', self.m_op_name_dict[w]
        #
        #
        # # # write to file
        # # with open('/Users/carl/PycharmProjects/op_importance/bad_operations_and_tasks_they_fail_on.txt', 'w') as f:
        # #     for item in sorted(bad_op_dict.items(), key=lambda t: len(t[1]), reverse=True):
        # #         f.write("%s: %i bad tasks\n" % (item[0], len(item[1])))
        # #         for bad_task in item[1]:
        # #                 f.write("%s," % bad_task)
        # #         f.write("\n\n")

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
        # np.savetxt('/Users/carl/PycharmProjects/op_importance/performance_all_ops_tasks_normed' + filenameSuffix,
        #            1 - all_classes_good_norm)
        # np.savetxt('/Users/carl/PycharmProjects/op_importance/performance_all_ops_tasks_raw' + filenameSuffix,
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

    def mask_pvals_too_few_unique_outputs(self):

        # todo: this replaces the mask of the p-values for simplicity. Remove from workflow later!!

        masked_pvals_all_tasks = self.pvals_good_op

        good_pval_mask = np.ma.getmask(masked_pvals_all_tasks)

        for task_ind, task in enumerate(self.tasks):

            # load number of unique values per operations from file
            n_uniques = np.loadtxt('/Users/carl/PycharmProjects/op_importance/uniqueValues/' + task.name + '.txt')

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
            n_uniques = np.load('/Users/carl/PycharmProjects/op_importance/results/intermediate_results_dectree_maxmin_unique_nulls_npy/task_' + task.name + '_tot_stats_all_runs.npy')

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

        nullFigDir = '/Users/carl/PycharmProjects/op_importance/nullHists/'
        if not os.path.exists(nullFigDir):
            os.makedirs(nullFigDir)

        # get all null stats
        null_stats_all_tasks = []
        print 'loading null stats'
        for task_ind, task in enumerate(self.tasks):
            # null_stats_all_tasks.append(self.stats_method.get_null_stats(task.name))
            null_stats_all_tasks.append(np.load('/Users/carl/PycharmProjects/op_importance/results/intermediate_results_dectree_maxmin_null_npy/task_'+task.name+'_tot_stats_all_runs.npy'))
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

    def kstest_null_distributions(self):

        from scipy import stats
        from matplotlib.pyplot import cm

        saveDir = '/Users/carl/PycharmProjects/op_importance/'

        # get all null stats
        null_stats_all_tasks = []
        print 'loading null stats'
        for task_ind, task in enumerate(self.tasks):
            # null_stats_all_tasks.append(self.stats_method.get_null_stats(task.name))
            null_stats_all_tasks.append(np.load('/Users/carl/PycharmProjects/op_importance/results/intermediate_results_dectree_maxmin_null_npy/task_'+task.name+'_tot_stats_all_runs.npy'))
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

        print 'hm'

        # f, ax = mpl.pyplot.subplots(1)
        # mpl.pyplot.hist(np.squeeze(null_stats_this_op), 100,
        #                 label="%s, p=%1.3f, N=%i" % (task.name, pvals_all_tasks_filled[task_ind], len(task.labels)))
        #
        # stats_this_op = self.stats_good_op[task_ind, sortedOpInds[ind]]
        # ax.axvline(stats_this_op, linestyle=':', color=c)
        #
        # # mpl.pyplot.legend()
        # mpl.pyplot.xlabel('error')
        # mpl.pyplot.ylabel('frequency')

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
            print '%i, %s' % (id, self.good_op_names[self.good_op_ids == id])

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

        np.savetxt('/Users/carl/PycharmProjects/op_importance/peformance_mat_fullMeanStd_topMeanStd_clusterMeanStd_new710.txt', perfmat)

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

        np.savetxt('/Users/carl/PycharmProjects/op_importance/peformance_canonical_linear.txt', perfmat)

    def classify_good_perf_ops_vs_super_vs_good_ops(self):

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


        perfmat = np.zeros((len(self.tasks), 8))
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

            # only super ops
            score_this_task_super_ops = cross_val_score(clf, task.data[:, np.isin(task.op_ids,
                                                                                    self.super_perf_op_ids)],
                                                          task.labels, cv=folds, scoring=scorer)

            # whole matrix
            score_this_task_whole = cross_val_score(clf, task.data, task.labels, cv=folds, scoring=scorer)

            # plot immediately
            p1 = mpl.pyplot.errorbar(np.mean(score_this_task_whole), np.mean(score_this_task_top_ops),
                                xerr=np.std(score_this_task_whole), yerr=np.std(score_this_task_top_ops), fmt='o', color='b', ecolor='b')
            p2 = mpl.pyplot.errorbar(np.mean(score_this_task_whole), np.mean(score_this_task_cluster_ops),
                                xerr=np.std(score_this_task_whole), yerr=np.std(score_this_task_cluster_ops), fmt='o',
                                color='r', ecolor='r')
            p3 = mpl.pyplot.errorbar(np.mean(score_this_task_whole), np.mean(score_this_task_super_ops),
                                     xerr=np.std(score_this_task_whole), yerr=np.std(score_this_task_super_ops),
                                     fmt='o',
                                     color='g', ecolor='g')

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

        np.savetxt('/Users/carl/PycharmProjects/op_importance/peformance_mat_fullMeanStd_topMeanStd_clusterMeanStd.txt', perfmat)

        mpl.pyplot.legend((p1, p2, p3), ('500 top ops', 'only cluster centers', 'super operations'))
        # mpl.pyplot.xlim((0, 1))
        # mpl.pyplot.ylim((0, 1))
        mpl.pyplot.xlabel('performance on whole feature set')
        mpl.pyplot.ylabel('performance only selected features')
        mpl.pyplot.show()

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
            # lowDim.savefig('/Users/carl/PycharmProjects/op_importance/UMAP/' + titleString + '.png')

            # top ops
            embedding = reducer.fit_transform(task.data[:, np.isin(task.op_ids, self.good_perf_op_ids)])
            df = pd.DataFrame(data=embedding, columns=('umap-1', 'umap-2'))
            df['label'] = task.labels
            lowDim = sns.lmplot(x='umap-1', y='umap-2', data=df, fit_reg=False, markers='.',
                                hue='label', legend=True, legend_out=True, palette='Set2')
            titleString = '%s UMAP %i top ops' % (task.name, len(self.good_perf_op_ids))
            mpl.pyplot.title(titleString)
            mpl.pyplot.tight_layout()

            lowDim.savefig('/Users/carl/PycharmProjects/op_importance/UMAP/' + titleString + '.png')

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

            lowDim.savefig('/Users/carl/PycharmProjects/op_importance/UMAP/' + titleString + '.png')

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

        np.savetxt('/Users/carl/PycharmProjects/op_importance/peformance_mat_fullMeanStd_topMeanStd_clusterMeanStd_givenSplit_nonBalanced_new710.txt', perfmat)

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

        np.savetxt('/Users/carl/PycharmProjects/op_importance/peformance_mat_canonical_givenSplit_nonBalanced_linear.txt', perfmat)

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
            '/Users/carl/PycharmProjects/op_importance/peformance_mat_n_clusters_new_variableNTopOps_noRaw.txt',
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

            mpl.pyplot.savefig('/Users/carl/PycharmProjects/op_importance/errorHistogramsHighPerformers/' + op_name + ',' + taskName + '.png')

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
            '/Users/carl/PycharmProjects/op_importance/errorHistogramsHighPerformers/' + op_name + ',_allTasks.png')

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
        mpl.pyplot.imshow(perfMatCatch22Norm,
                          vmin=-absMax, vmax=absMax,
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
            '/Users/carl/PycharmProjects/op_importance/catch22_performance_zscore_clustered.pdf')
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
            '/Users/carl/PycharmProjects/op_importance/catch22_performance_corr.pdf')

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
        # old 2015 data
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
        # # insignificance tasks
        # task_names = ["CBF", "Lightning7"] # , "ECGMeditation", "LargeKitchenAppliances", "Lightning2", "MedicalImages"]
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
        # task_names = ["Adiac", 'ArrowHead']
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

    n_good_perf_ops = 710 # 811 #
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
    plot_out_path = outputDir + 'out_figure.eps'
    result_txt_outpath = outputDir + 'result_txt.txt'
    masking_method = 'NaN'

    # combine_pair_method = 'mean'
    # combine_tasks_method = 'mean'
    # combine_tasks_norm = None
    select_good_perf_ops_norm = 'mean-norm' # 'zscore' # 'median-diff' # 'none' #
    select_good_perf_ops_method = 'sort_asc'
    select_good_perf_ops_combination = 'mean' # 'pos_sum' #
    similarity_method = 'abscorr' # 'abscorr', 'corr', 'cos', 'euc'
    compare_space = 'problem_stats'
    min_calc_tasks = np.ceil(float(len(task_names)) * 0.8) # np.ceil(float(len(task_names)) / float(1.25))

    # -----------------------------------------------------------------
    # -- Initialise Class instances -----------------------------------
    # -----------------------------------------------------------------

    input_method = Data_Input.Datafile_Input(path_pattern,masking_method,label_regex_pattern)
    redundancy_method = Reducing_Redundancy.Reducing_Redundancy(similarity_method = similarity_method,compare_space = compare_space)

    workflow = Workflow(task_names,input_method,ranking_method,
                        # combine_tasks_method = combine_tasks_method,combine_tasks_norm = combine_tasks_norm,
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
        # workflow.read_data(is_read_feature_data=False, old_matlab=old_matlab) # only performances, no feature-outputs
        workflow.read_data(old_matlab=old_matlab) # read full data for the classification comparison in the end.
        workflow.load_task_attribute('tot_stats', path_pattern_task_attrib)

    print 'loaded everything'

    # # -- exclude operations with 'raw' keyword
    # workflow.exclude_ops_keyword_per_task('raw')
    # -- find the features which are calculated for at least min_calc_tasks tasks
    workflow.find_good_op_ids(min_calc_tasks)
    # -- exclude operations with 'raw' keyword
    workflow.exclude_good_ops_keyword('raw')
    # -- Collect all combined stats for each task and take stats for good (non-NaN) operations only
    workflow.collect_stats_good_op_ids()
    # combination now done in selection step.
    # # -- Combine the stats of all the tasks (mean of classification error, not p-value)
    # workflow.combine_tasks()
    # -- Select a subset of well performing operations (z-score across tasks, take n best)
    workflow.select_good_perf_ops()

    # np.savetxt('/Users/carl/PycharmProjects/op_importance/performance_all_ops_tasks_normed_mean_noRaw.txt', workflow.stats_good_op_norm)
    # quit()

    # np.savetxt('/Users/carl/PycharmProjects/op_importance/good_op_ids.txt', workflow.good_op_ids)
    # quit()

    # lostOps = []
    # for task in workflow.tasks:
    #     # print "%s: %i/7658" % (task.name, len(task.op_ids))
    #     lostOpsThisDataset = 7658 - 766 - len(task.op_ids)
    #     lostOps.append(lostOpsThisDataset)
    #
    #     print "%s: %i" % (task.name, lostOpsThisDataset)
    # print "lost operations in normalisation: min %i, max %i, mean %i, std %i" %(np.min(lostOps), np.max(lostOps), np.mean(lostOps), np.std(lostOps))
    # quit()

    # workflow.kstest_null_distributions()
    # quit()

    # print 'ID, name'
    # for i, opId in enumerate(workflow.good_op_ids):
    #     print opId, ', ', workflow.good_op_names[i]
    # quit()

    # workflow.show_high_performer_dist('DN_HistogramMode_10')
    # quit()

    # np.savetxt('/Users/carl/PycharmProjects/op_importance/statsGoodPerfOps_noFailureAllowed.txt', workflow.stats_good_op[:, np.isin(workflow.good_op_ids, workflow.good_perf_op_ids)])

    # # -- just for analysis, print out the operation ids and performances
    # for accuracyError in [True, False]:
    #     for comb in ['mean', 'min', 'pos_sum']:
    #         for norm in ['z-score', 'mean-norm', 'median-norm', 'median-diff', 'none']:
    #             # workflow.plot_perf_histograms(norm, comb)
    #             workflow.select_good_perf_ops_sort_asc_input_params_to_file(norm, comb, accuracyError)
    # quit()
    # # mpl.pyplot.show()

    # # plot the correlation between tasks
    # mpl.pyplot.imshow(np.ma.corrcoef(workflow.stats_good_op), cmap=mpl.pyplot.cm.get_cmap('jet', 10))
    # cb = mpl.pyplot.colorbar()
    # mpl.pyplot.clim(0, 1);
    # cb.set_label('Pearson correlation')
    # mpl.pyplot.yticks(np.arange(len(workflow.task_names)), workflow.task_names, fontsize=8)
    # mpl.pyplot.savefig('/Users/carl/PycharmProjects/op_importance/Fig2A_taskCorrelation.eps')
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

    # -- mask p-values of operations with too few outputs
    # workflow.mask_pvals_too_few_unique_outputs()
    workflow.mask_pvals_too_few_unique_nulls()
    # -- Combine the p-values of all the tasks
    workflow.combine_task_pvals(min_p=0.05)
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
    # workflow.create_op_keyword_and_name_dict()
    # workflow.list_bad_and_non_significant_ops()
    # quit()
    # workflow.plot_null_distributions(0.01)
    # workflow.plot_one_null_distribution(5702, "SmallKitchenAppliances")

    # -----------------------------------------------------------------
    # -- Do the redundancy calculations -------------------------------
    # -----------------------------------------------------------------

    # -- intitialise the redundancy method with the calculated results
    workflow.init_redundancy_method_problem_space()
    # -- calculate the correlation matrix saved in workflow.redundancy_method.similarity_array
    workflow.redundancy_method.calc_similarity()
    # -- calculate the linkage, the cluster indices and the clustering in self.corr_linkage,self.cluster_inds,self.cluster_op_id_list,respectively
    workflow.redundancy_method.calc_hierch_cluster(t = max_dist_cluster)
    #
    # # -- single features for each cluster
    workflow.select_good_perf_cluster_center_ops()

    # # just for testing, show which super-operations ends up in which cluster:
    # for i, cluster_op_ids in enumerate(workflow.redundancy_method.cluster_op_id_list):
    #     superIndicator = np.isin(workflow.super_perf_op_ids, cluster_op_ids)
    #
    #     print 'cluster ' + str(i) + ' :'
    #     for name in workflow.good_op_names[np.isin(workflow.good_op_ids, workflow.super_perf_op_ids[superIndicator])]:
    #         print name

    # -----------------------------------------------------------------
    # -- Classification perf with feature subsets ---------------------
    # -----------------------------------------------------------------

    # workflow.UMAP_all_topOps_clusters()
    # workflow.classify_good_perf_ops_vs_good_ops()
    # workflow.classify_good_perf_ops_vs_super_vs_good_ops()
    # workflow.classify_N_clusters()
    # workflow.classify_good_perf_ops_vs_good_ops_givenSplit()
    # workflow.classify_selected_ops([0011, 0012, 0134, 0135, 0241, 1121, 7543, 3477, 1406, 1585, 1965, 0310, 2997, 3264, 3294, 4492, 3467, 3604, 4036, 4156, 4421, 3010])
    # workflow.classify_selectedOps_givenSplit()
    # quit()

    # # -- show performance matrix of catch22-features only
    # workflow.show_catch22_perfmat()

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
    # measures[2,:] = fap.normalise_masked_array(workflow.stats_good_op_comb, axis= 0,norm_type = 'zscore')[0]
    # meanErrors = np.nanmean(np.ma.filled(workflow.stats_good_op, np.nan),0)
    # measures[2, :] = (meanErrors - np.nanmean(meanErrors))/np.nanstd(meanErrors)
    measures[2, :] = (workflow.stats_good_op_comb - np.nanmean(workflow.stats_good_op_comb))/np.nanstd(workflow.stats_good_op_comb)
    # # -- the normalised combined error as selected
    # measures[2, :] = workflow.stats_good_op_comb

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
