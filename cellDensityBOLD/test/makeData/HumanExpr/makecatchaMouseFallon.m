cdh()

cd('../../Data/HumanfMRI/')

% You would have run makeHCTSAFallon.m, so you'll have 100 hctsa files in this folder.
% Let's work with those.
load('./Data/subs100.mat')
mkdir('./catchaMouseHCTSAFiles')
for i = 1:height(subs100)
    data = hctsa2catchaMouse(['./HCTSA_', num2str(i), '.mat']);
    save(['./catchaMouseHCTSAFiles/HCTSA_', num2str(i), '.mat'], '-struct', 'data')
end
