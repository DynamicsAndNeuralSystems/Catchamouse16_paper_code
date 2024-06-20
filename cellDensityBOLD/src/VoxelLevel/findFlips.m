function findFlips(matA, matB, flipDims)
% Show the two input matrices with the same dimensions to determine whcih
% need to be flipped, for imprtVoxelTimeseries. FlipDims filips the
% dimension of matB as specified
    if ~all(sort(size(matA)) == sort(size(matB)))
        error('The sorted dimensions of the two input matrices are inconsistent')
    end
    % Match dimensions of matB to those of matA
    if ~all(size(matB) == size(matA))
        [~, ~, permdim] = intersect(size(matA), size(matB), 'stable');
        matB = permute(matB, permdim);
    end
    
    for i = 1:length(flipDims)
       matB = flip(matB, flipDims(i));
    end
    
    matA(matA < 1) = 0;
    matB(matB < 1) = 0;
    volumeViewer(logical(matA))
    volumeViewer(matB)
end

