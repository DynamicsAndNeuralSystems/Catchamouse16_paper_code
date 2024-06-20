function layer_densityCorrelationMatrix(data)
    f = gcf;
    set(f, 'color', 'w')
    for i = 1:size(data, 1)
        subplot(size(data, 1), 1, i)
        densityCorrelationMatrix(data(i).Data, [], 1);
        ax = gca;
        ax.FontSize = 5;
        ax.Colorbar.Label.FontSize = 7;
        title(sprintf('%s) %s', char(i + 96), data(i).Layer), 'FontSize', 7)
        axis image
    end
    f.Units = 'pixels';
    f.Position = [711 1 409 1150];
end

