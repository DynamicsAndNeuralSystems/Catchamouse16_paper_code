function cdh(loc)
% Change the directory to HERE. Most often will go at the start of a script.
    if nargin < 1 || isempty(loc)
        loc = dbstack('-completenames'); % Get the path of the calling file
    else
        loc = which(loc); % You have named a file. Find out where it is
    end
    try
        if length(loc) > 1 && ~ischar(loc)
            loc = loc(end, :).file;
        end
        loc = regexp(loc, ['.*(?=\', filesep, '\w+)'], 'match');
        if isempty(loc)
            warning('File not found. Doin'' nothin''')
        else
            evalin('base', ['cd(''', loc{1}, ''')'])
        end
    catch
        warning('Could not enter this directory. Doin'' nothin''')
    end
end
