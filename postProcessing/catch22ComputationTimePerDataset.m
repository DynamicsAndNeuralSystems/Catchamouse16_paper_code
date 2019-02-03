function catch22ComputationTimePerDataset()

    C_outdir = '/Users/carl/PycharmProjects/catch22/C/featureOutputs/';

    %% load C out
    cd(C_outdir);

    files = dir('*.txt');
    filenames = {files.name};

    for fileInd = 1:numel(filenames)

        filename = filenames{fileInd}; 
        f = fopen(filename, 'r');

        filenameSplit = split(filename, '_');
        filenameStripped = filenameSplit{1};
        if fileInd == 1
            outStruct.name = filenameStripped;
            outStruct.values = [];
        elseif ~strcmp(filenameStripped, outStruct(end).name)
            structTemp.name = filenameStripped;
            structTemp.values = [];
            
            outStruct = [outStruct, structTemp];
        end
        
        
        C = textscan(f, '%f %s %f', 'Delimiter',',');

        outStruct(end).values = [outStruct(end).values, sum(C{3})];
        
        fclose(f);
    end

    
    save('/Users/carl/PycharmProjects/op_importance/intermediateAnalysisResults/CcomputationTimes.mat', 'outStruct');
end