% Remember home directory
MyPBSHome = pwd;

% Load paths for the HCTSA package
cd /home/ma/p/pknaute/work/git_repos/hctsa/
startup

% Move Matlab back to the working PBS directory
cd(MyPBSHome)

% Set parameters for run:
parallelize = 0; % set to 1 to parallelize computations over available CPUs using Matlab's Parellel Computing Toolbox?
DoLog = 0; % set to 1 to log results to a .log file? (usually not necessary)
tsidmin = xxTSIDMINxx; % calculate from this ts_id...
tsidmax = xxTSIDMAXxx; % to this ts_id
WriteWhat = 'null'; % retrieve and write back missing (NULL) entries in the database

% Define tsidr, the range of time series to calculate (one at a time):
tsidr = (tsidmin:tsidmax);

% Retrieve a vector of m_ids to calculate subject to additional conditions
% Here we remove operations with labels 'shit', 'kalafutvisscher' and 'waveletTB'
fprintf(1,'time_01 ###   %s  ###   \n',datestr(now))
opids = SQL_getids('ops',1,{},{'tisean'});
fprintf(1,'time_02 ###   %s  ###   \n',datestr(now))
% Range of m_ids retrieved at each iteration:
opidr = [min(opids), max(opids)];

% ------------------------------------------------------------------------------
%% Now start calculating!
% ------------------------------------------------------------------------------
% Provide a quick message:
fprintf(1,['About to calculate across ts_ids %u--%u and m_ids %u--%u over a total of '  ...
    		 '%u iterations\n'],tsidr(1),tsidr(end),opidr(1),opidr(end),1);

%for i = 1:length(tsidr) % Loop over single time series
%fprintf(1,'\n\n%s\nWe''re looking at ts_id %u to ts_id %u and %u m_ids, from %u--%u\n\n\n', ...
%                        	datestr(now),tsidr(1),tsidr(end),length(opids),opidr(1),opidr(2))

% We loop over:
% (i) Running TSQ_prepared to retrieve data from the database -> HCTSA_loc.mat
% (ii) Using TSQ_brawn to calculate missing entries
% (iii) Running TSQ_agglomerate to write results back into the database
try
    fprintf(1,'time_03 ###   %s  ###   \n',datestr(now))
	DidWrite = TSQ_prepared(tsidr,opids,WriteWhat); % Collect the null entries in the database
    fprintf(1,'time_04 ###   %s  ###   \n',datestr(now))
    	if DidWrite % Only calculate if TSQ_prepared found time series to retrieve:
        fprintf(1,'time_05 ###   %s  ###   \n',datestr(now))
		TSQ_brawn(DoLog,parallelize); % computes the operations and time series retrieved
		fprintf(1,'time_06 ###   %s  ###   \n',datestr(now))
        TSQ_agglomerate(WriteWhat,DoLog); % stores the results back to the database
        fprintf(1,'time_07 ###   %s  ###   \n',datestr(now))
    	else
		fprintf(1,'No time series retrieved for ts_id = %u\n',tsidr(i));
    	end
catch
	disp(['Calculation failed.\n'])
end
%end

% Clean up by removing the local file HCTSA_loc.mat (if it exists)
if exist('HCTSA_loc.mat')~=0, delete('HCTSA_loc.mat'); end
fprintf(1,'All calculations for ts_ids from %u to %u completed at %s\n',tsidmin,tsidmax,datestr(now));
