import numpy as np
import os

class Task:
    
    def __init__(self,name,input_method,stat_method=None):
        """
        Constructor.
        Parameters:
        -----------
        name : str
            The name of the current task
        input_method : Data_Input
            The data input method used to read the data from disk. 
        stat_method : Feature_Stats
            The mehtod used to calculate the statistics
        """
        # -- calculation methods
        self.input_method = input_method
        self.stat_method = stat_method
        # -- task identifier
        self.name = name
        
        # -- data placeholders
        self.labels = np.array([])
        self.data = np.ma.masked_array([])
        self.op_ids = np.array([])
        self.pair_stats = np.ma.masked_array([])
        self.tot_stats = np.ma.masked_array([])
        self.tot_stats_all_runs = np.ma.masked_array([])
        self.tot_stats_p_vals = np.ma.masked_array([])
        self.op = dict()
        self.ts = dict()

    def calc_stats(self,is_keep_data = False):
        """
        Calculate the statistics using the method given by self.stat_method. Pairwise and task total. 
        Parameters:
        ----------
        is_keep_data : bool
            Is the feature data to be kept after calculating the statistics or discarded (to save RAM space)?
            
        """
        print "Calculating stats for Task: {}".format(self.name)

        if self.stat_method.is_pairwise_stat:
            self.pair_stats = self.stat_method.calc_pairs(self.labels, self.data)
            # -- combine the stats of the label pairs to one pooled stat for each feature
            self.tot_stats = self.stat_method.combine_pair(self.pair_stats)
        else:
            self.tot_stats_all_runs, self.tot_stats, self.tot_stats_p_vals = self.stat_method.calc_tots(self.labels,self.data,self.name)


        # -- free data if not required anymore to safe RAM space
        if not is_keep_data:
            self.data = None

        
    def read_data(self,is_read_feature_data = True, old_matlab = False):
        """
        Read the data using the input method given by self.input_method.
        Paramters:
        ----------
        is_read_feature_data : bool
            Is the feature data to be read or not
        """
        #self.data, keywords_ts, self.op_ids = self.input_method.input_task(self.name,is_read_feature_data = is_read_feature_data)
        # self.data, self.ts, self.op = self.input_method.input_task(self.name,is_read_feature_data=is_read_feature_data, old_matlab=old_matlab)
        self.data, self.ts, self.op, self.m_op = self.input_method.input_task_master(self.name,is_read_feature_data=is_read_feature_data, old_matlab=old_matlab)

        self.op_ids = np.array(self.op['id'])
        self.keywords_op = self.op['keywords']
        self.keywords_ts = self.ts['keywords']
        self.labels = self.input_method.extract_labels(self.keywords_ts)
    
    def load_attribute(self,attribute_name,in_path_pattern):
        """
        Load an attribute of the instance from a file
        Parameters:
        -----------
        attribute_name : string
            The name of the attribute of Task to be loaded
        out_path-pattern : string
            A string containing the pattern for the path pointing to the input file. 
            Formatted as in_path_pattern.format(self.name,attribute_name) + file extension
        """
       
        if attribute_name == 'tot_stats':
            self.tot_stats = np.loadtxt(in_path_pattern.format(self.name,attribute_name)+'.txt')
            self.tot_stats_all_runs = np.loadtxt(in_path_pattern.format(self.name,attribute_name)+'_all_runs.txt')
            pvals_file = in_path_pattern.format(self.name,attribute_name)+'_p_vals.txt'
            if os.path.exists(pvals_file):
                self.tot_stats_p_vals = np.loadtxt(pvals_file)


    def save_attribute(self,attribute_name,out_path_pattern):    
        """
        Save an attribute of the instance to a file
        Parameters:
        -----------
        attribute_name : string
            The name of the attribute of Task to be saved
        out_path-pattern : string
            A string containing the pattern for the path pointing to the output file. 
            Formatted as out_path_pattern.format(self.name,attribute_name) + file extension
        """

        if attribute_name == 'tot_stats':
            print "Saving {}".format(out_path_pattern)
            np.savetxt(out_path_pattern.format(self.name,attribute_name)+'.txt',self.tot_stats)
            np.savetxt(out_path_pattern.format(self.name,attribute_name)+'_all_runs.txt',self.tot_stats_all_runs)
            if self.tot_stats_p_vals.size != 0:
                np.savetxt(out_path_pattern.format(self.name,attribute_name)+'_p_vals.txt',self.tot_stats_p_vals)

           
