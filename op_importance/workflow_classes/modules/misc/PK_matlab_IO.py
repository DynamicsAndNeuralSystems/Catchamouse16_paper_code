'''
Created on 8 Jun 2015

@author: philip knaute
------------------------------------------------------------------------------
Copyright (C) 2015, Philip Knaute <philiphorst.project@gmail.com>,

This work is licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of
this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send
a letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
California, 94041, USA.
------------------------------------------------------------------------------
'''
import csv
import scipy.io as sio

def read_calc_times(mat_file_path):
    """ Read the average calculation times for each operation from a HCTSA_loc.mat file
    Parameters:
    -----------
    mat_file_path : string
        Path to the HCTSA_loc.mat file
    Returns:
    --------
    op_ids_times_lst : list
        List of two lists. First being the operation ids and second the average times over all 
        calculated timeseries for each respective operation.
    """    
    mat_file = sio.loadmat(mat_file_path)
    op_id = lambda i : int(mat_file['Operations'][i][0][3][0])
    
    calc_times = mat_file['TS_CalcTime'].sum(axis=0)/mat_file['TS_CalcTime'].shape[0]
    op_ids = [op_id(i) for i in range(mat_file['Operations'].shape[0])]
    return [op_ids,calc_times.tolist()]

def read_from_mat_file(inputDir,task_name,hctsa_struct_names,is_from_old_matlab = False):
    """
    read .mat files into appropriate python data structures
    
    Parameters:
    ----------
    inputDir : string
        path to the directory where the mat/csv files are kept
    task_name : string
        the name of the classification task to be imported
    hctsa_struct_names : list
        List of strings of identifiers for which structures are to be read from the mat_file. 
        Possible values are : 'TimeSeries','Operations','TS_DataMat'
    is_from_old_matlab : bool
        If the HCTSA_loc.mat files are saved from an older version of the comp engine. The order of entries is different.
    Returns:
    --------
    retval : tuple
        Tuple of the imported values in the order given by hctsa_struct_names
    """

    LTv7_3 = False # lesser than mat v7.3
    # Input for false case not implemented yet
    

    # try:
    #     mat_file = sio.loadmat(mat_file_path)
    # except NotImplementedError:
    #     LTv7_3 = False
    #     mat_file = h5py.File(mat_file_path,'r')
    # except:
    #     ValueError('Could not read the file!')
    
    retval = tuple()
    for item in hctsa_struct_names:
        if item == 'TimeSeries':
            path_pattern = inputDir+'{:s}_TSInfo.txt'
            ts_info_path = path_pattern.format(task_name)
            timeseries = dict()
            if is_from_old_matlab:
                # lambda function used to populate the dictionary with the appropriate data lists
                ts_id = lambda i : int(ts_info['TimeSeries'][i][0][0][0]) #id
                ts_filename = lambda i : str(ts_info['TimeSeries'][i][0][1][0]) #name
                ts_kw = lambda i : str(ts_info['TimeSeries'][i][0][2][0]) #keyword
                ts_n_samples = lambda i : int(ts_info['TimeSeries'][i][0][3][0]) #length
                # data is not included in the returned dictionary as it seem a waste of space
                #ts_data = lambda i : mat_file['TimeSeries'][i][0][4]
                for extractor,key in zip([ts_id,ts_filename ,ts_kw,ts_n_samples],['id','filename','keywords','n_samples']):
                    timeseries[key] =[extractor(i) for i in range(mat_file['TimeSeries'].shape[0])]
            elif not is_from_old_matlab and LTv7_3:
                # -- currently there seems to be a bug in the creation of those files. Need to
                #    read them differently
                # lambda function used to populate the dictionary with the appropriate data lists
                #ts_id = lambda i : int(mat_file['TimeSeries'][i][0][0][0])

                # ----------- this is the original version, I (Carl) somehow need to switch the first two dimensions ----
                # ts_filename = lambda i : str(mat_file['TimeSeries'][0][i][0][0])
                # ts_kw = lambda i : str(mat_file['TimeSeries'][0][i][1][0])
                # ts_n_samples = lambda i : int(mat_file['TimeSeries'][0][i][2][0])
                # # -- data is not included in the returned dictionary as it seem a waste of space
                # #ts_data = lambda i : mat_file['TimeSeries'][i][0][3]
                # for extractor,key in zip([ts_filename ,ts_kw,ts_n_samples],['filename','keywords','n_samples']):
                #     timeseries[key] =[extractor(i) for i in range(mat_file['TimeSeries'].shape[1])]

                # ------------ to this
                ts_filename = lambda i: str(mat_file['TimeSeries'][i][0][0][0])
                ts_kw = lambda i: str(mat_file['TimeSeries'][i][0][1][0])
                ts_n_samples = lambda i: int(mat_file['TimeSeries'][i][0][2][0])
                # -- data is not included in the returned dictionary as it seem a waste of space
                # ts_data = lambda i : mat_file['TimeSeries'][i][0][3]
                for extractor, key in zip([ts_filename, ts_kw, ts_n_samples], ['filename', 'keywords', 'n_samples']):
                    timeseries[key] = [extractor(i) for i in range(mat_file['TimeSeries'].shape[0])]
            else:
                with open(ts_info_path,'r') as f:
                    reader = csv.reader(f)
                    ts_info = []
                    for row in reader:
                        ts_info.append(row)
                    del ts_info[0]
                ts_filename = lambda i: str(ts_info[i][0])
                ts_kw = lambda i: str(ts_info[i][1])
                ts_n_samples = lambda i: int(ts_info[i][2])

                for extractor, key in zip([ts_filename, ts_kw, ts_n_samples], ['filename', 'keywords', 'n_samples']):
                    timeseries[key] = [extractor(i) for i in range(len(ts_info))]
                
            retval = retval + (timeseries,)
            
        if item == 'Operations':
            path_pattern = inputDir+'{:s}_Operations.txt'
            op_path = path_pattern.format(task_name)
            operations = dict()
            if is_from_old_matlab:
                op_id = lambda i : int(mat_file['Operations'][i][0][0][0])
                op_name = lambda i : str(mat_file['Operations'][i][0][1][0])
                op_kw = lambda i : str(mat_file['Operations'][i][0][2][0])
                op_code = lambda i : str(mat_file['Operations'][i][0][3][0])
                op_mopid = lambda i : int(mat_file['Operations'][i][0][4][0])
                for extractor,key in zip([op_id,op_name ,op_kw,op_code,op_mopid],['id','name','keywords','code_string','master_id']):
                    operations[key] =[extractor(i) for i in range(mat_file['Operations'].shape[0])]               
            elif not is_from_old_matlab and LTv7_3:
                # lambda function used to populate the dictionary with the appropriate data lists
                op_id = lambda i : int(mat_file['Operations'][i][0][3][0])
                op_name = lambda i : str(mat_file['Operations'][i][0][1][0])
                op_kw = lambda i : str(mat_file['Operations'][i][0][2][0])
                op_code = lambda i : str(mat_file['Operations'][i][0][0][0])
                op_mopid = lambda i : int(mat_file['Operations'][i][0][4][0])
                for extractor,key in zip([op_id,op_name ,op_kw,op_code,op_mopid],['id','name','keywords','code_string','master_id']):
                    operations[key] =[extractor(i) for i in range(mat_file['Operations'].shape[0])]
            else:
                with open(op_path,'r') as f:
                    reader = csv.reader(f)
                    op = []
                    for row in reader:
                        op.append(row)
                    del op[0]
                op_id = lambda i : int(op[i][3])
                op_name = lambda i : str(op[i][1])
                op_kw = lambda i : str(op[i][2])
                op_code = lambda i : str(op[i][0])
                op_mopid = lambda i : int(op[i][4])
                for extractor,key in zip([op_id,op_name ,op_kw,op_code,op_mopid],['id','name','keywords','code_string','master_id']):
                    operations[key] =[extractor(i) for i in range(len(op))]
            retval = retval + (operations,)

        if item == 'TS_DataMat':
            if LTv7_3:
                raise ValueError("Not Implemented yet")
            else:
                path_pattern = inputDir + '{:s}_TSData.mat'
                mat_file_path = path_pattern.format(task_name)
                mat_file = sio.loadmat(mat_file_path)
                retval = retval + (mat_file['data'],)

        if item == 'MasterOperations':
            path_pattern = inputDir+'{:s}_MasterOperations.txt'
            m_op_path = path_pattern.format(task_name)
            m_operations = dict()
            if is_from_old_matlab:
                raise NameError('Don''t know how to get MasterOperations from old Matlab version.')
            elif not is_from_old_matlab and LTv7_3:
                # lambda function used to populate the dictionary with the appropriate data lists
                m_op_id = lambda i: int(mat_file['MasterOperations'][i][0][2][0][0])
                m_op_name = lambda i: str(mat_file['MasterOperations'][i][0][1][0])
                for extractor, key in zip([m_op_id, m_op_name],
                                      ['id', 'name']):
                    m_operations[key] = [extractor(i) for i in range(mat_file['MasterOperations'].shape[0])]
            else:
                with open(m_op_path,'r') as f:
                    reader = csv.reader(f)
                    m_op = []
                    for row in reader:
                        m_op.append(row)
                    del m_op[0]
                m_op_id = lambda i: int(m_op[i][2])
                m_op_name = lambda i: str(m_op[i][1])
                for extractor, key in zip([m_op_id, m_op_name],
                                        ['id', 'name']):
                    m_operations[key] = [extractor(i) for i in range(len(m_op))]
            retval = retval + (m_operations,)

    return retval
