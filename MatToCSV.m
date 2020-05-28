function [] = MatToCSV(name)
% name = 'Dataset1'; 
% RightCtx -- Dataset1, LeftCtx -- Dataset2, Control -- Dataset3

normalizedData = load('HCTSA_N.mat');
writetable(normalizedData.Operations,strcat(name,'_Operations.txt'));
writetable(normalizedData.MasterOperations,strcat(name,'_MasterOperations.txt'));

for i = 1:size(normalizedData.TimeSeries,1)
    normalizedData.TimeSeries.Data{i,1}=i;
end

data = normalizedData.TS_DataMat;
save(strcat(name,'_TSData.mat'),'data');
writetable(normalizedData.TimeSeries,strcat(name,'_TSInfo.txt'));

end