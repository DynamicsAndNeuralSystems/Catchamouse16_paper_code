cdh()

dummyDREADD = autoLoad('../Data/DREADD/RightCtx_HCTSA_CAMK_Excitatory_PVCre_SHAM_ts2-BL_v1.mat');
dataDensity = autoLoad('../Data/Results/AllFeatures_100Subjects/Layers/joined_layer_data.mat');
dummyDensity = dataRef(dataDensity, 'excitatory', 'Isocortex');
Nfeatures = 2;
denVar = [0.1];

% We want two classes of dreadd, 'SHAM' and 'DREADD'. 
TS_DataMat = dummyDREADD.TS_DataMat;
Nobs = 130;
%Nfeatures = size(TS_DataMat, 2);
sep = ones(1, Nfeatures);
sep(1, end) = 0;
X1 = randn(Nobs, Nfeatures) - sep; % Variance 1 centred at -1
X2 = randn(Nobs, Nfeatures) + sep; % Variance 1 centred at +1
Y1 = repmat({'SHAM'}, Nobs, 1);
Y2 = repmat({'DREADD'}, Nobs, 1);

% We'll leave the hctsa feature names there, since anything else would be
% just as arbitrary. Same with the time series.
dummyDREADD.TS_DataMat = [X1; X2];
while height(dummyDREADD.TimeSeries) < Nobs.*2
    dummyDREADD.TimeSeries = [dummyDREADD.TimeSeries; dummyDREADD.TimeSeries];
end
dummyDREADD.TimeSeries = dummyDREADD.TimeSeries(1:Nobs.*2, :);
dummyDREADD.TimeSeries.Group = [Y1; Y2];
dummyDREADD.groupNames = {'SHAM', 'DREADD'};
dummyDREADD.TimeSeries.Keywords = [Y1; Y2];
dummyDREADD.Operations = dummyDREADD.Operations(1:Nfeatures, :);
dummyDREADD.Operations.Name = arrayfun(@num2str, 1:Nfeatures, 'un', 0)';

% Onto densities. Same as for DREADD; replacing density value sbut leaving
% labels and names in place, for now, even though they lose meaning
Nreg = length(dummyDensity.Inputs.density);
% Now, we want to sample from a line, but centred at 0.5's with a decent spread and a small amount of
% noise on the featue values
%dummyDensity.Inputs.density = randn(Nreg, 1) + 0.5; % Centre 0.5, SD 1
TS_DataMat = dummyDREADD.TS_DataMat;
%Nfeatures = size(TS_DataMat, 2);
dummyDensity.Inputs.density = randn(Nreg, 1).*0.5; 
dummyDensity.TS_DataMat = dummyDensity.Inputs.density + denVar.*randn(Nreg, Nfeatures); % So feature values are just densities with a little noise
dummyDensity.Operations = dummyDREADD.Operations;

save('../Data/DREADD/dummyDREADD.mat', '-struct', 'dummyDREADD')
save('../Data/DREADD/dummyDensity.mat', 'dummyDensity')