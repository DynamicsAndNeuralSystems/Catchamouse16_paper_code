clear variables;

dbc = SQL_opendatabase;


SQL_retrieve(1:50,'all','all')

SQL_closedatabase(dbc);

TS_normalize()
delete('HCTSA.mat')

clear variables
load('HCTSA_N.mat')
save('HCTSA_ts1_N_70_100_reduced.mat','-v7')
delete('HCTSA_N.mat')

clear variables;

dbc = SQL_opendatabase;

SQL_retrieve(1000:1050,'all','all')

SQL_closedatabase(dbc);

TS_normalize()
delete('HCTSA.mat')

clear variables
load('HCTSA_N.mat')
save('HCTSA_ts2_N_70_100_reduced.mat','-v7')
delete('HCTSA_N.mat')
