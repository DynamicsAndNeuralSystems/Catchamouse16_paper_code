function checkZScore()

    folder = '../UCR_2018_rawHCTSA_shaved';

    curDir = pwd;
    cd(folder);

    files = dir;
    files([files.isdir])=[];

    for i = 1:length(files)
        thisFile = files(i);
        load(thisFile.name, 'TimeSeries');

        nTS = length(TimeSeries);
        means = nan(nTS,1);
        stds = nan(nTS,1);
        for j = 1:nTS
            means(j) = mean(TimeSeries(j).Data);
            stds(j) = std(TimeSeries(j).Data);
        end
        fprintf("mean mean %1.3f, mean std %1.3f (%s)\n", nanmean(means), nanmean(stds), thisFile.name);
    end