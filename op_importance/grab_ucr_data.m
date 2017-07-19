cd('UCR_TS_Archive_2015');

% Get a list of all files and folders in this folder.
topFiles = dir;
dirFlags = [topFiles.isdir];
subFolders = topFiles(dirFlags);
% Print folder names to command window.

topDir = pwd;

timeSeriesData = {};
keywords = {};
labels = {};
for i = 1 : length(subFolders)
    if strcmp(subFolders(i).name,'.') || strcmp(subFolders(i).name,'..')
        continue;
    end
    fprintf('Moving into %s\n',subFolders(i).name)
    cd(subFolders(i).name);
    allfs = dir;
    dirFlags = [allfs.isdir];
    fs = allfs(~dirFlags);
    for j = 1:length(fs)
        fprintf('Processing %s\n',fs(j).name);
        data = load(fs(j).name);
        for k = 1:size(data,1)
            ts = data(k,2:end);
            class = int2str(data(k,1));
            timeSeriesData{end+1} = ts;
            keywords{end+1} = [class,',',subFolders(i).name,',',fs(j).name,',ts_num',int2str(k)];
            labels{end+1} = [fs(j).name,'_',int2str(k),'_class_',class];
        end
    end
    cd(topDir);
end

timeSeriesData = timeSeriesData';
keywords = keywords';
labels = labels';
save('INP_test.mat','timeSeriesData','keywords','labels');
%TS_init('INP_test.mat',~,~,[0,0,0]);
