'''
Created on 5 Nov 2015

@author: philip
'''
import numpy as np
import scipy.spatial.distance as spdst
import scipy.cluster.hierarchy as hierarchy

def calc_linkage(abs_corr_array, linkage_method):#'average'):# complete
    """
    Calculate the linkage for a absolute correlation array
    Parameters:
    -----------
    abs_corr_array : ndarray
        array containing the correlation matrix
    linkage_method : str,optional
        The linkage algorithm to use.
    Returns:
    --------
    link_arr : ndarray
        The hierarchical clustering encoded as a linkage matrix.
    abs_corr_dist_arr : ndarray
        The distance array calculated from abs_corr_array
    """
    # -- transform the correlation matrix into distance measure
    abs_corr_dist_arr = np.around(1 - abs_corr_array,7)
    # -- XXX The correlation sometime is larger than 1. not sure if that is a negligible or has to
    # -- be sorted out
    abs_corr_dist_arr[(abs_corr_dist_arr < 0)] = 0
    #np.fill_diagonal(abs_corr_dist_arr, 0)
    # -- transform the correlation matrix into condensed distance matrix
    dist_corr = spdst.squareform(abs_corr_dist_arr)
    
    # -- calculate the linkage
    link_arr = hierarchy.linkage(dist_corr, method=linkage_method)

    return link_arr,abs_corr_dist_arr

def calc_perform_corr_mat(all_classes_avg_good,norm = None, type='abscorr', max_feat = 200):
    """
    Calculate the correlation matrix of the performance for top features. If norm != None it uses a normed
    version of the all_classes_avg_good array for estimating the best features. It uses the non-normed originla
    version of the array to calculate the correlation array, though.
    XXX There is an issue if too many entries in all_classes_avg_good are masked which can result in correlation
    coeffitients larger than one in their absolute value.
    Parameters:
    -----------
    all_classes_avg_good : masked ndarray
        Masked array containing the average statistics for each problem(row = problem)
        for each good (preselected) operation (column = operation)
    norm : str,optional
        The name of the normalisation if any. Options are 'z-score' or 'none'
    max_feat : int
        Max number of feature for which the correlation is calculated
    Retruns:
    --------
    abs_corr_array : nd_array
        Array containing the correlation matrix for the top max_feat features. Entries are sorted by sort_ind.
    sort_ind : ndarray
        indices that would sort the rows of all_classes_avg_good.
    all_classes_avg_good_norm : masked ndarray
        Array similar to all_classes_avg_good but normed by 'norm'.
    """
    if norm in ['z-score','zscore'] :
        all_classes_avg_good = np.ma.masked_invalid(all_classes_avg_good)
        all_classes_avg_good_norm = ((all_classes_avg_good.T - np.ma.mean(all_classes_avg_good,axis=1)) / np.ma.std(all_classes_avg_good,axis=1)).T
    elif norm == 'mean-norm':
        all_classes_avg_good_mean = np.ma.masked_invalid(np.ma.mean(all_classes_avg_good,axis = 1))
        all_classes_avg_good_norm = (all_classes_avg_good.T / all_classes_avg_good_mean).T
    else:
        all_classes_avg_good = np.ma.masked_invalid(all_classes_avg_good)
        all_classes_avg_good_norm = all_classes_avg_good

    #all_classes_avg_good_norm = np.ma.masked_invalid(all_classes_avg_good_norm)
    sort_ind = np.ma.argsort(all_classes_avg_good_norm.mean(axis=0))
    acag_n_sort_red = all_classes_avg_good[:,sort_ind[:max_feat]]

    def replace_nan_with_mean(a):


        # obtain mean of columns as you need, nanmean is just convenient.
        col_mean = np.nanmean(a, axis=0)

        # find indicies that you need to replace
        inds = np.where(np.isnan(a))

        # place column means in the indices. Align the arrays using take
        acag_n_sort_red[inds] = np.take(col_mean, inds[1])

        return a


    # -- calculate the correlation
    if type == 'corr':

        corr_array = np.ma.corrcoef(acag_n_sort_red, rowvar=0)

    elif type=='spearmanr':
        from scipy.stats import spearmanr
        corr_array, _ = spearmanr(acag_n_sort_red, axis=0)

        # If using masked arrays, ensure proper handling of masked values
        corr_array = np.ma.array(corr_array, mask=np.isnan(corr_array))

    elif type == 'abscorr':

        corr_array = np.abs(np.ma.corrcoef(acag_n_sort_red, rowvar=0))

    elif type == 'cos':

        # there's no built-in pdist function that handles missing values/ nans. So replace nans by column means
        # (mean over tasks of feature performance)
        acag_n_sort_red_noNaN = replace_nan_with_mean(acag_n_sort_red)

        corr_array = 1 - spdst.squareform(spdst.pdist(acag_n_sort_red_noNaN.T, 'cosine'))

    elif type == 'euc':

        # there's no built-in pdist function that handles missing values/ nans. So replace nans by column means
        # (mean over tasks of feature performance)
        acag_n_sort_red_noNaN = replace_nan_with_mean(acag_n_sort_red)

        corr_array = 1 - spdst.squareform(spdst.pdist(acag_n_sort_red_noNaN.T, 'euclidean'))

    else:

        raise NameError('Unknown distance type.')

    if np.ma.max(corr_array ) > 1.1:
        raise ValueError('Too many masked values in "all_classes_avg_good". \n np.ma.corrcoeff() returned invalid values')

    return corr_array,sort_ind,all_classes_avg_good_norm
