function [timeSeriesData, refTable] = importVoxelTimeseries(timeSeriesData, mask, flipDims)

    % You have one file containg a matrix with timeseries at every voxel,
    % and a mask to select these timeseries. Want a format these can be
    % stored in that is compatible with hctsa computations.
    % Will need to:
    %   - Arrange the timeseries matrix so that it is consistent with the mask
    %   - Reshape the timeseries to a 2d matrix, hctsa compatible
    %   - Make an accompanying table that has the same order as the matix,
    %   but has: 
    %       - Filename or other subject identifier
    %       - Mask coordinates, i, j, k
    %       - Mask value i.e. the structure ID
    % Will preference the orientation of the coordinate system of the mask.
    %
    % The timeSeriesData dimensions will be permuted to match the mask
    % dimensions, but you must specify if you want to flip the matrix along
    % any of these dimensions. This will be applied AFTER dimension
    % matching. e.g. [1 2] flips along dimensions 1 as well as 2.
    
    if nargin < 3
        flipDims = [];
    end
    if ischar(timeSeriesData)
        filename = timeSeriesData;
        timeSeriesData = niftiread(filename);
    end
    
    if ischar(mask)
        mask = h5read(mask, '/mask');
    end
    
%% Match the timeseries array to the mask
    tdims = size(timeSeriesData);
    
    % Check the mask and the timeSeriesData at least have the same unordered dimensions
    if ~all(sort(tdims(1:3)) == sort(size(mask)))
        error('The timeSeriesData and the mask are not consistent')
    end
    
    % Permute the dimensions of timeSeriesData if neccessary
    if ~all(tdims(1:3) == size(mask))
        [~, ~, permdim] = intersect(size(mask), tdims(1:3), 'stable'); % Get the indices that reorder the dimensions of timeSereisData into those of mask
        timeSeriesData = permute(timeSeriesData, [permdim', 4]);
    end
    
    % Then flip if requested
   for i = 1:length(flipDims)
       timeSeriesData = flip(timeSeriesData, flipDims(i));
   end
    
%% Flatten
    [R, C, P] = ndgrid(1:size(mask, 1), 1:size(mask, 2), 1:size(mask, 3));
    timeSeriesData = reshape(timeSeriesData, [], size(timeSeriesData, 4));
    mask = reshape(mask, [], 1);
    R = reshape(R, [], 1);
    C = reshape(C, [], 1);
    P = reshape(P, [], 1);

%% Remove timeseries where the mask is 0
    idxs = mask ~= 0;
    mask = mask(idxs);
    timeSeriesData = timeSeriesData(idxs, :);
    R = R(idxs);
    C = C(idxs);
    P = P(idxs);

%% Make the reference table
    indices = [R, C, P];
    refTable = table(mask, indices, 'VariableNames', {'structID', 'MaskIndices'});

end

