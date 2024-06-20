function PlotScatter(xData,yData,structInfo)

%f = figure('color','w');
set(gcf, 'color', 'w')
hold('on');

numPoints = height(structInfo);
for i = 1:numPoints
    plot(xData(i),yData(i),'ok','MarkerFaceColor',rgbconv(structInfo.color_hex_triplet{i}));
end
