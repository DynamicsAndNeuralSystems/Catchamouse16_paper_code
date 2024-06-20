function moveAxisExponent(ax)

    yExp = [];
    xExp = [];
    
    useLaTex = strcmp(ax.YAxis.Label.Interpreter, 'latex');
    
    if ~isempty(ax.YLabel.String)
        yExp = double(ax.YRuler.Exponent);
        if useLaTex
            ax.YLabel.String{end} = sprintf('%s $$(\\times 10^{%i})$$', ax.YLabel.String{end}, yExp);
            ax.YRuler.Exponent = 0;  
        else
            ax.YLabel.String{end} = sprintf('%s (\\times 10^{%i})', ax.YLabel.String{end}, yExp);
            ax.YRuler.Exponent = 0;
        end
    end
    
    useLaTex = strcmp(ax.XAxis.Label.Interpreter, 'latex');
    
    if ~isempty(ax.XLabel.String)
        xExp = double(ax.XRuler.Exponent);
        if useLaTex
            ax.XLabel.String{end} = sprintf('%s $$(\\times 10^{%i})$$', ax.XLabel.String, xExp);
            ax.XRuler.Exponent = 0; 
        else
            ax.XLabel.String{end} = sprintf('%s (\\times 10^{%i})', ax.XLabel.String{end}, xExp);
            ax.XRuler.Exponent = 0;
        end
    end
    
    % Then go and scale the DATA, rather than the axes labels
    for c = 1:length(ax.Children)
        if isprop(ax.Children(c), 'YData') && ~isempty(yExp)
            ax.Children(c).YData = ax.Children(c).YData./(10.^yExp);
        end
        if isprop(ax.Children(c), 'XData') && ~isempty(xExp)
            ax.Children(c).XData = ax.Children(c).XData./(10.^xExp);
        end
    end
    
end

