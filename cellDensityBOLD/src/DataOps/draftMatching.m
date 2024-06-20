
% We need to match names to acronyms:
structInfoNames = regexprep(structInfo.name,',','');
[~,ia,ib] = intersect(lower(structInfoNames),lower(dataTable.Regions),'stable');
dataTable = dataTable(ib,:);
dataTable.acronym = structInfo.acronym(ia);
fprintf(1,'%u names match to set of %u structures\n',...
                height(dataTable),height(structInfo));

for i = 1:height(dataTable)
    fprintf(1,'%s (%s)\n',dataTable.Regions{i},dataTable.acronym{i});
end
