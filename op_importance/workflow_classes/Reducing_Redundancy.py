import modules.misc.PK_helper as hlp
import modules.feature_importance.PK_ident_top_op as idtop
import scipy.cluster.hierarchy as hierarchy
import numpy as np

class Reducing_Redundancy:
    def __init__(self,similarity_method,compare_space):
        """
        Constructor
        Parameters:
        -----------
        similarity_method : string
            String describing the method used to calculate the distance array
        compare_space : string
            String describing in which space the distance calculation is going to happen (e.g. problem_stats,feature_vals)
        """
        if similarity_method == 'abscorr':
            self.calc_similarity = self.calc_abs_corr
        elif similarity_method == 'corr':
            self.calc_similarity = self.calc_corr
        elif similarity_method == 'cos':
            self.calc_similarity = self.calc_cos
        elif similarity_method == 'euc':
            self.calc_similarity = self.calc_euc
        else:
            raise NameError('No valid distance metric chosen.')

        self.compare_space = compare_space   
        self.good_perf_op_ids = None
        self.ops_base_perf_vals = None
        self.similarity_array = None
        self.cluster_inds = None # Indices to which cluster an entry of the similarity array belongs
        self.cluster_op_id_list= None # List of lists containing the op_id for the operations in the clusters
        self.linkage = None # The linkage calculated for the similarity matrix
        self.similarity_array_op_ids = None # Operation ids of the correlation array
        
    def set_parameters(self,ops_base_vals,good_op_ids,good_perf_op_ids):
        """
        Set and compute the parameters needed to calculate the distance array
        Parameters:
        -----------
        ops_base_vals : nd array
            Array containing the values on which the similarity of the operations will be calculated
        good_op_ids : ndarray
            The op_ids of the columns in  ops_base_vals
        good_perf_op_ids : ndarray
            The op_ids of the features we are interested in
        """
        self.good_perf_op_ids = good_perf_op_ids
        self.good_op_ids = good_op_ids
        # -- This discards the potential large ops_base_vals and also good_op_ids after exiting the constructor
        self.ops_base_perf_vals = self.reduce_to_good_perf_ops(ops_base_vals,self.good_perf_op_ids,good_op_ids)
        

    def reduce_to_good_perf_ops(self,ops_base_vals,good_perf_op_ids,good_op_ids):
        """
        Reduce the ops_base_vals by keeping only the columns corresponding to the op_ids in self.good_perf_op_ids
        Parameters:
        -----------
        ops_base_vals : nd array
            Array containing the values on which the similarity of the operations will be calculated
        good_op_ids : ndarray
            The op_ids of the columns in  ops_base_vals
        good_perf_op_ids : ndarray
            The op_ids of the features we are interested in
        Returns:
        --------
        ops_base_perf_vals : ndarray
            ops_base_vals reduced to contain only operations with ids given in good_perf_op_ids with the same ordering.
        """
        good_perf_ind = hlp.ismember(good_perf_op_ids,good_op_ids)
        ops_base_perf_vals = ops_base_vals[:,good_perf_ind]
        return ops_base_perf_vals
    
    def calc_abs_corr(self):
        """
        Calculate the distance matrix using a correlation approach for every column in self.ops_base_perf_vals
        """
        # -- no normalisation in here as the best performing features have been picked already, potentially using normalisation
        self.similarity_array,sort_ind,_ = idtop.calc_perform_corr_mat(self.ops_base_perf_vals,norm=None, type='abscorr',
                                                              max_feat = self.ops_base_perf_vals.shape[1])
        self.similarity_array_op_ids = self.good_perf_op_ids[sort_ind]

    def calc_corr(self):
        """
        Calculate the distance matrix using a correlation approach for every column in self.ops_base_perf_vals
        """
        # -- no normalisation in here as the best performing features have been picked already, potentially using normalisation
        self.similarity_array,sort_ind,_ = idtop.calc_perform_corr_mat(self.ops_base_perf_vals,norm=None, type='abscorr',
                                                              max_feat = self.ops_base_perf_vals.shape[1])
        self.similarity_array_op_ids = self.good_perf_op_ids[sort_ind]

    def calc_cos(self):
        """
        Calculate the distance matrix using a correlation approach for every column in self.ops_base_perf_vals
        """
        # -- no normalisation in here as the best performing features have been picked already, potentially using normalisation
        self.similarity_array,sort_ind,_ = idtop.calc_perform_corr_mat(self.ops_base_perf_vals,norm=None, type='cos',
                                                              max_feat = self.ops_base_perf_vals.shape[1])
        self.similarity_array_op_ids = self.good_perf_op_ids[sort_ind]

    def calc_euc(self):
        """
        Calculate the distance matrix using a correlation approach for every column in self.ops_base_perf_vals
        """
        # -- no normalisation in here as the best performing features have been picked already, potentially using normalisation
        self.similarity_array, sort_ind, _ = idtop.calc_perform_corr_mat(self.ops_base_perf_vals, norm=None,
                                                                         type='euc',
                                                                         max_feat=self.ops_base_perf_vals.shape[1])
        self.similarity_array_op_ids = self.good_perf_op_ids[sort_ind]

    def calc_hierch_cluster(self,t = 0.2, criterion='distance' ):
        """
        Calculate the clustering using hierachical clustering
        Parameters:
        -----------
        t : float
            The threshold to apply when forming flat clusters.
        criterion : str, optional
            The criterion to use in forming flat clusters. 
        """
        self.linkage = idtop.calc_linkage(self.similarity_array)[0]
        self.cluster_inds = hierarchy.fcluster(self.linkage, t = t, criterion=criterion)
        # -- map index to op_id and create list of lists representing clusters
        self.cluster_op_id_list = [[] for x in xrange(self.cluster_inds.max())]
        for i,cluster_ind in enumerate(self.cluster_inds):
            self.cluster_op_id_list[cluster_ind-1].append(self.similarity_array_op_ids[i])
            
    def write_cluster_file(self,out_path,op_id_name_map,measures):
        """ Write text file containing operation names arranged in their respective clusters
        Parameters:
        -----------
        out_path : string
            path to which the results are written
        op_id_name_map : list of lists
            A list of lists where the first sub-list corresponds to the operation id and the 
            second sub-list to the operation name as returned e.g. by PLOTTING.map_op_id_name_mult_task()
        measures : list of lists
            A list of lists where the first sub-list corresponds to the operation id the second sub-list the 
            number of problems for which the operation has been calculated successfully and the third is
            the z-scored average u-stat for each operation.
        """
        with open(out_path,'w') as out_file:
            out_file.write('------------------------------------------------------------------\n')
            out_file.write('--- clusters of operations----------------------------------------\n')
            out_file.write('--- name, n problems calculated, normalised combined error\n')
            out_file.write('------------------------------------------------------------------\n')
            for cluster in self.cluster_op_id_list:

                names = ['' for i in cluster]
                n_calcs = np.zeros(len(cluster))
                norm_ustats = np.zeros(len(cluster))

                for opInd, op in enumerate(cluster):
                    ind_tmp = np.nonzero(measures[0]==op)[0]
                    names[opInd] = op_id_name_map[1][op_id_name_map[0].index(op)]
                    n_calcs[opInd] = measures[1][ind_tmp]
                    norm_ustats[opInd] = measures[2][ind_tmp]

                sortInds = np.argsort(norm_ustats)

                for ind in sortInds:
                    out_file.write('{:s},{:d},{:1.2f}\n'.format(names[ind],int(n_calcs[ind]),norm_ustats[ind]))

                out_file.write('------------------------------------------------------------------\n')
        
        
        
class Correlation_Dist:
    def __init__(self):
        pass
