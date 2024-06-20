function tbl = joinDataTables(tblA, tblB, tblAcol, tblBcol)
% Performs an outer join on the two tables using the specified columns,
% accounting for duplicate columns and commas in text and merging the two
% keys. 
    
    if ischar(tblA)
        tblA = readtable(tblA);
    end
    if ischar(tblB)
        tblB = readtable(tblB);
    end
    tblA.(tblAcol) = strrep(tblA.(tblAcol),',','');
    tblB.(tblBcol) = strrep(tblB.(tblBcol),',','');
        
    
    tbl = outerjoin(tblA, tblB, 'LeftKeys', tblAcol, 'RightKeys', tblBcol);
    
    tbl(ismissing(tbl.(tblAcol)), :).(tblAcol) = tbl(ismissing(tbl.(tblAcol)), :).(tblBcol);
    tbl(ismissing(tbl.(tblBcol)), :).(tblBcol) = tbl(ismissing(tbl.(tblBcol)), :).(tblAcol);
    
    tbl = removevars(tbl, tblBcol);
    
end

