hemisphere = {'Right','Left'};
conditions = {'CAMK','excitatory','SHAM','PVCre'};
numConditions = length(conditions);

dirs = [];
for h = 1:2
    theHemisphere = hemisphere{h};
    if strcmp(theHemisphere,'Right')
        matFile = 'HCTSA_CalculatedData/HCTSA_data_CerCtx/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_N.mat';
    elseif strcmp(theHemisphere,'Left')
        matFile = 'HCTSA_CalculatedData/HCTSA_data_CerCtx/LeftCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_N.mat';
    else
        error("Hemisphere was set to %s which is not 'r' or 'l' for right and left brain hemispheres, respectively",theHemisphere);
    end
    for i = 1:numConditions
        for j = i+1:numConditions
            % (make new directory)
            writedir = sprintf('op_importance/input_data/maxmin/%s_%s_%s',theHemisphere,conditions{i},conditions{j});
            mkdir(writedir);
            % (specify output filename to put into new directory)

            [IDs,notIDs] = TS_GetIDs({conditions{i},conditions{j}},matFile,'ts','Keywords');
            TS_Subset(matFile,IDs,[],1,append(writedir,'/hctsa_datamatrix.mat'));
            % TS_LabelGroups(outputFileName,{conditions{i},conditions{j}});
            % (if python code uses the .Group column of the TimeSeries Table, the above LabelGroups will be needed to e.g., label the two conditions as 0 and 1)
            oldfolder = cd(writedir);
            OutputToCSV(append(writedir,'/hctsa_datamatrix.mat'),true,true);
            cd(oldfolder);
        end
    end
end
