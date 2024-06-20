function animateVoxelTimeseries(timeSeriesData,filtertol)
%ANIMATEVOXELTIMESERIES Animate the given time series
    if nargin < 2 
        filtertol = [];
    end
    if ischar(timeSeriesData)
        timeSeriesData = niftiread(timeSeriesData);
    end
    if length(size(timeSeriesData)) > 3
        timeSeriesData = timeSeriesData(:, :, 20, :);
    end
    if ~isempty(filtertol)
        tsMean = mean(abs(timeSeriesData), 4);
        tolIdxs = repmat(tsMean < filtertol, 1, 1, 1, size(timeSeriesData, 4));
        timeSeriesData(tolIdxs) = 0;
    end
    f = gcf;
    set(gcf, 'color', 'w')
    cmap = cbrewer('div', 'RdBu', 256);
    colormap(cmap)
    %amap = [0; ones(255, 1)];
    v = VideoWriter('VoxelAnimation.avi', 'Uncompressed AVI');
    open(v)
    cmin = min(timeSeriesData(:));
    cmax = max(timeSeriesData(:));
    for t = 1:size(timeSeriesData, 4)%'AlphaMap', amap,
        %volshow(timeSeriesData(:, :, :, t), 'ColorMap', cmap,  'BackgroundColor', 'w', 'renderer', 'VolumeRendering')
        imagesc(timeSeriesData(:, :, t))
        caxis([cmin, cmax])
        drawnow
        writeVideo(v, getframe(f))
    end
    close(v)
end

