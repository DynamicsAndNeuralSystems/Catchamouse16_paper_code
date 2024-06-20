function [varA, varB] = templateLoadFunction()
%TEMPLATELOADFUNCTION A template autoLoad function. Adjust for each dataset,
% and place in an adjacent to the data files.
    thisDir = strrep(which(mfilename), [mfilename, '.m'], '');
    savefileName = fullfile(thisDir, [mfilename, '.mat']);

    if isfile(savefileName)
        load(savefileName)
        return
    end

% Load any variables from adjacent data files. This will depend substantially between datasets
    loadfileName = fullfile(thisDir, 'testData.mat');
    load(loadfileName); % Loads varA, varB, varC

% Perform any reshaping or pre-processing
    varA = (varA.*2)';
    varB = varB./varC;

% Save the data for efficient loading in the future
    %save(savefileName, 'varA', 'varB',  '-v7.3') % Uncomment once load function is finalised
end
