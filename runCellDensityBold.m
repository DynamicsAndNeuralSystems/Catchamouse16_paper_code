[currentDir, ~, ~] = fileparts(mfilename('fullpath'));

cd(fullfile(currentDir,"cellDensityBOLD/catchaMouse16_paper"));
add_all_subfolders()
generate_figures();