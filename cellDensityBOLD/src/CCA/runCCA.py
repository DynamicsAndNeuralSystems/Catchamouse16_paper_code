import rcca

def runCCA(X, Y, theReg, theNumCC):
    cca = rcca.CCA(reg=theReg, numCC=theNumCC)
    cca.train([X, Y])
    cca.compute_ev([X, Y])
    return cca 

#def runCCACrossValidate(X, Y, theRegs='', theNumCCs=''):
#    if theRegs == '':
#        theRegs = [0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#    if theNumCCs == '':
#        theNumCCs = [1, 2, 3, 4, 5]
#    cca = rcca.CCACrossValidate(regs=theRegs, numCCs=theNumCCs)
#    cca.train([X, Y])
#    cca.compute_ev([X, Y])
#    return cca 
