function f = nfPlotNullComparison(dataDREADD, dataDensity, ops, classKeys, model, params, FsCriterion, wsCutoff, fileName)
%NFPLOTNULLCOMPARISON
    %rng(32)
    if ischar(dataDREADD)
        dataDREADD = autoLoad(dataDREADD);
    end
    if ischar(dataDensity)
        dataDensity = autoLoad(dataDensity);
    end
    if nargin < 3 || isempty(ops)
        ops = 'all';
    end
    if nargin < 4 || isempty(classKeys)
        classKeys = {'Sham', 'Excitatory'};
    end
    if nargin < 5 || isempty(model)
        model = {'LDA', 'SVM', 'ranksum', 'ranksum_logp'};
    end
    if nargin < 6
        params = [];
    end
    if nargin < 7 || isempty(FsCriterion)
        FsCriterion = 'misclassification';
    end
    if nargin < 8 || isempty(wsCutoff)
        wsCutoff = 0;
    end
    if nargin < 9
        fileName = [];
    end
    
    
    if ~isempty(fileName) && isfile(fileName)
        fprintf('Found precomputed nulls in %s, so using these...\n', fileName)
        load(fileName, 'nullDir', 'nullDen')
    else
        nReps = 500;
        % Make direction null
        fprintf('\nSampling %i direction nulls:\n', nReps)
        nullDir = nfNullDistribution(dataDREADD, dataDensity, ops, classKeys, 0, nReps);

        % Make density null
        fprintf('\nSampling %i density nulls:\n', nReps)
        nullDen = nfNullyDensity(dataDREADD, dataDensity, ops, classKeys, 0, nReps);
    end
    
    % Make the classifier thresholds
    for i = 1:length(model)
        [rhos(i), ~, ~, ~, ~, ~, ps(i)] = nfCompare(dataDREADD, dataDensity, ops, classKeys, model{i}, params, FsCriterion, wsCutoff);
    end
    
    % Plot
    f = figure('color', 'w');
    customHistogram(nullDen, 50, [], 1, iwantcolor('gray', 1).*0.5)
    if ~isstruct(nullDir)
        customHistogram(nullDir, 50, 'k');
    end
    colors = iwantcolor('cellDensityDREADD');
    smodel = cellfun(@(x) strrep(x, '_', '\_'), model', 'un', 0);
    for i = 1:length(model)
        % Then add the three p-value estimates
        % pa is the random density density more extreme (one sided) than rho
        % pb is the random direction density " " " " " "
        % pc is the p value from corr()
        if isstruct(nullDir)
            customHistogram(nullDir.(model{i}), 25, [], [], colors(i, :))
            pb = sum(sign(rhos(i)).*nullDir.(model{i}) > abs(rhos(i)))./length(nullDir.(model{i}));
        else
            pb = sum(sign(rhos(i)).*nullDir > abs(rhos(i)))./length(nullDir);
        end
        pa = sum(sign(rhos(i)).*nullDen > abs(rhos(i)))./length(nullDen);
        
        pc = ps(i);
        if iscell(model{i})
            model{i} = [model{i}{:}];
        end
        xline(rhos(i), '-', {smodel{i}, sprintf('p_a = %.0e, p_b = %.0e, p_c = %.0e', pa, pb, pc)}, 'Color', colors(i, :), 'LineWidth', 2.5);
        %xline(rhos(i), '-', sprintf('p_a = %.0e, p_b = %.0e, p_c = %.0e', pa, pb, pc),...
              %'Color', colors(i, :), 'LineWidth', 2.5, 'LabelHorizontalAlignment', 'left');
    end
    xlabel('\rho')
    ylabel('Frequency')
    ax = gca;
    ax.Box = 'on';
    legend({'Random Densities', 'Random Directions'}, 'Location', 'NorthWest')
end
