# UROP Summer
# 6/20/17
# Generalized matrix multiplication
# Inhomogeneous Testing

import itertools
import numpy as np
import random as rd
import time
import math
from string import ascii_lowercase as al

# looks like a set of tuples
# adds like a list
# multiplies like Cartesian coordinates
# is each element of a labeledTensor
class solutionTuple(object):
    def __init__(self,sol=None):
        if type(sol) == list:
            self.sol = [[i for i in sol]]
        else:
            self.sol = [[sol]] # first value yay!

    def __str__(self):
        tortn = "{("
        for sol in self.sol:
            for i in range(len(sol)):
                if sol[i] is None:
                    tortn += str(sol[i])
                else: tortn += str('%.3f'%(sol[i]))
                if i != len(sol)-1:
                    tortn += ", "
            tortn += ")"
            if self.sol.index(sol) != len(self.sol)-1:
                tortn += ", ("

        tortn += "}"
        return tortn

    def getSol(self):
        return self.sol

    def setSol(self,newsol):
        if type(sol) == list:
            self.sol = [[i for i in sol]]
        else:
            self.sol = [[sol]]

    def add(self, tup2): # rightwards add - set concatenation
        j = 0
        if self.sol[0][0] is None:
            self.sol = [tup2.getSol()[0]]
            j = 1

        for i in range(j, len(tup2.getSol())):
            self.sol.append(tup2.getSol()[i])

    def delete(self, tup): # remove tup from self.sol
        self.sol.remove(tup.getSol()[0])
        
    def mult(self, tup2): # rightwards multiply - Cartesian multiplication
        if (not (self.sol[0][0] is None) and not (tup2.getSol()[0][0] is None)):
            newself = []
            bothlists = [self.sol, tup2.getSol()]
            for e in itertools.product(*bothlists):
                e = list(e)
                e = [item for sublist in e for item in sublist]
                newself.append(e)
            # print (newself)
            toRtn = solutionTuple(newself[0])
            for i in range(1, len(newself)):
                toRtn.add(solutionTuple(newself[i]))
            return toRtn
            # return solutionTuple(newself[0]) # don't want to modify the tensor
        return solutionTuple()

    def simpleMult(self, tup2): # rightwards multiply - only first element of Cartesian product
        if (not (self.sol[0][0] is None) and not (tup2.getSol()[0][0] is None)):
            newself = []
            bothlists = [self.sol, tup2.getSol()]
            for e in itertools.product(*bothlists):
                e = list(e)
                e = [item for sublist in e for item in sublist]
                newself.append(e)
            return solutionTuple(newself[0]) # don't want to modify the tensor
        return solutionTuple()

class labeledTensor(object):
    def __init__(self, mat, dims, resos):
        # assert (len(list(mat.shape)) == len(dims)) # the dims can actually match
        assert (len(dims) == len(resos)) # same number of variables
        self.tensor = mat # numpy matrix
        self.dims = dims # dimension names in order
        self.resolutions = resos # resolutions of each dimension in order

    def getTensor(self):
        return self.tensor

    def getDims(self):
        return self.dims

    def getResos(self):
        return self.resolutions

    def extractDim(self,a,b,dim): # extract possible values of a dimension
        assert(dim in self.dims)
        n = self.getReso(dim)
        vals = [((n-1-i)*a+i*b)/(n-1) for i in range(n)]
        return vals

    def getReso(self,dim): # get resolution of dimension
        return self.resolutions[self.dims.index(dim)]

# n possible solution values y for each variable
# NOTE: n IS NOT THE NUMBER OF VARIABLES
# from steady state program
# params contains other useful variables, like the current spatial coordinates
# val, val1, val2 etc. contain potential steady state solution values
# Because of the nature of the discretized equation, let
# val = n_j, val1 = n_{j+1}, val2 = n_{j+2}
def makeU(a,b,n,params):
    U = []
    dim = [((n-1-i)*a+i*b)/(n-1) for i in range(n)]
    for val1 in dim:
        row = [] #new row for variable 
        for val in dim:
            element = []
            for val2 in dim:
                if steadyStateTest([val1,val,val2],params,dim): # not hardcoded anymore!
                    element.append(solutionTuple(val))
                else: element.append(solutionTuple())
            row.append(element)
        U.append(row)
    return U, dim

# If all your U's have the same range and dimension
# (In other words, without the labels, all the U's are identical)
# This method resolves some of the hardcoding issues
# numMats: number of matrices
# a, b, n, params as defined in makeU, although params contains all U's
# varNames: list of all variable names in all the U's
# dimU: the dimension of each U
# paramlen is the length of params for each U
# difparams is a boolean indicating whether each U's params are different or not
def makeAllU(numMats,a,b,n,params,varnames,dimU,paramlen=0,difparams=False,):
    assert (numMats + dimU - 1 == len(varnames)) # we have exactly enough varnames
    Us = []
    if not(difparams):
        templateTensor, dim1 = makeU(a,b,n,params)
    for i in range(numMats):
        if difparams:
            templateTensor, dim1 = makeU(a,b,n,params[i*paramlen,(i+1)*paramlen])
        Us.append(labeledTensor(templateTensor, varnames[i:(i+dimU)], [n]*dimU))

    return Us, dim1

# rounding to resolution - from steady state program
def roundtores(num, dim): #resolution n
    c = dim[0]; m = dim[1] - dim[0] # b_i = m*i + c
    i = (num - c)/m # solve for i
    roundi = round(i+.00001) # round i to the nearest integer, round up
    return m*roundi+c

# Thank you Stack Overflow
def removeDupsOrder(vars):
    seen = set()
    seen_add = seen.add
    return [x for x in vars if not (x in seen or seen_add(x))]

# Discrete steady state tester
# Used in making the U matrix
def steadyStateTest(orderedvars,params,dim):
    dif = (dim[1] - dim[0])/2 # L_inf norm
    h = .05
    factor = math.pow(h,-2)

    # heat equation
    # assume all difs for all variables are the same, by system symmetry/doesn't make sense otherwise
    ui = orderedvars[1]; uleft = orderedvars[0]; uright = orderedvars[2]
    '''vs = []
    #for i in range(8):
        # vs.append(uleft+math.pow(-1,i)*dif + uright+math.pow(-1,i//2)*dif - 2*(ui+math.pow(-1,i//4)*dif))
    if abs(ui - max(dim)) <= .0001 and abs(uleft - max(dim)) <= .0001:
        for i in range(8):
            vs.append(uleft+math.pow(-1,i//4)*dif + uright + math.pow(-1, i) * dif - 2 * (ui+math.pow(-1, i//2)*dif))
    elif abs(uleft - max(dim)) <= .0001:
        for i in range(4):
            vs.append(uleft+math.pow(-1,i//2)*dif + uright + math.pow(-1, i) * dif - 2 * (ui-dif))
    elif abs(ui - max(dim)) <= .0001:
        for i in range(4):
            vs.append(uleft-dif + uright+math.pow(-1,i)*dif - 2*(ui+math.pow(-1,i//2)*dif))
    else:
        for i in range(2):
            vs.append(uleft-dif + uright+math.pow(-1,i)*dif - 2*(ui-dif))

    return len(set(np.sign(vs))) > 1 # multiple signs? There was a 0 in the subcube. One sign? The plane doesn't intersect'''

    # Non-functional W-J code, although I think the math is right.
    '''p = float(params[0]); delta = float(params[1])
    if ((p < 0) or (delta < 0)) and ((ui == 0) or (uleft == 0)): # check for negative powers of 0
        return False
    if (abs(p - round(p)) > .0001 and uleft < 0) or (abs(delta - round(delta)) > .0001 and (ui < 0 or uleft < 0)):
        return False

    source = math.pow(uleft,p)
    diffusion = (math.pow(ui,delta)*(uright - ui) - math.pow(uleft,delta)*(ui - uleft)) * factor
    v = source + diffusion
    try:
        grad = [factor*math.pow(ui,delta),
            factor*(delta*math.pow(ui,delta-1)*(uright-ui) - math.pow(n,delta) - math.pow(uleft,delta)),
            factor*(delta*math.pow(uleft,delta-1)*(ui-uleft)) + p*math.pow(uleft,p-1)]
    except:
        return False

    mag = 0 # magnitude
    for i in grad:
        mag += math.pow(i,2)
    mag = math.pow(mag,.5)
    maxchange = mag * dif # the dot of the gradient with itself is mag^2, divided by mag and multiplied by dif
    return (abs(v) < maxchange)'''

    # Functional W-J code
    assert(len(orderedvars) == 3)
    p = float(params[0]); delta = float(params[1])
    if ((p < 0) or (delta < 0)) and ((ui == 0) or (uleft == 0)): # check for negative powers of 0
        return False
    if (abs(p - round(p)) > .0001 and uleft < 0) or (abs(delta - round(delta)) > .0001 and (ui < 0 or uleft < 0)):
        return False

    source = math.pow(uleft,p)
    diffusion = (math.pow(ui,delta)*(uright - ui) - math.pow(uleft,delta)*(ui - uleft)) * factor
    return (abs(roundtores(source + diffusion, dim)) < .00001)

# Generalized matrix multiplication
# A times B (labelled tensors)
def matMult(A, B, exposed, simple):
    # Preparation
    matA = A.tensor; matB = B.tensor # what we're actually multiplying
    avars = A.dims; bvars = B.dims;
    aresos = A.resolutions; bresos = B.resolutions
    allvar = removeDupsOrder(avars + bvars) # in order, removed duplicates
    indexers = [i for i in allvar if not (i in exposed)]
    
    allresos = []
    for v in allvar:
        if v in avars:
            allresos.append(aresos[avars.index(v)])
        elif v in bvars:
            allresos.append(bresos[bvars.index(v)])

    exposedi = [allvar.index(e) for e in exposed]
    indexi = [allvar.index(i) for i in indexers]
    exresos = [allresos[e] for e in exposedi]
    inresos = [allresos[i] for i in indexi]

    # multipliable lists with indexers' resolutions
    cartinds = []
    for ir in inresos:
        cartinds.append([i for i in range(ir)])

    rawshape = []
    for r in exresos:
        rawshape.append(r)

    shape = tuple(rawshape)
    Usol = np.empty(shape)
    Usol = Usol.tolist()

    # for each set of possible boundary conditions taking from A and B:
    for bci in np.ndenumerate(Usol):
        bc = list(list(bci)[0]) #extract tuple of bc indices
        # these indices are the indices of Usol we care about for now

        # start a new solutionTuple a
        el = solutionTuple()

        # for each possible set of indexer values
        for inds in itertools.product(*cartinds): # indexers!
            # extract the right values to multiply
            subela = matA; subelb = matB
            for v in avars:
                if v in exposed:
                    # the second exposed variable corresponds to the
                    # second index in the boundary conditions, etc.
                    i = exposed.index(v) # take advantage of ordering
                    subela = subela[bc[i]]
                elif v in indexers:
                    i = indexers.index(v)
                    subela = subela[inds[i]]

            for v in bvars:
                if v in exposed:
                    j = exposed.index(v)
                    subelb = subelb[bc[j]]
                elif v in indexers:
                    j = indexers.index(v)
                    subelb = subelb[inds[j]]

            # print (str(subela))
            # print (str(subelb))

            # both subela and subelb should be solution tuples
            # A[indexer values].mult(B)[indexer values]
            # a.add(A[indexer values])
            if simple:
                prod = subela.simpleMult(subelb)
            else: prod = subela.mult(subelb)
            if (not (subela.getSol()[0][0] is None) and not (subelb.getSol()[0][0] is None)):
                el.add(prod)

        UpSol = Usol
        for n in range(len(bc)-1):
            UpSol = UpSol[bc[n]]

        UpSol[bc[len(bc)-1]] = el # by parameter passing, USol gets it

    Usol = np.array(Usol)
    return labeledTensor(Usol, exposed, exresos)

# Determining whether two solutions are identical
# Algorithm: difference between two solutions is less than
# 1/2 * (number of variables) * (dif)^2 where dif is
# the difference between two adjacent bins (possible solution values)
def reduceSolutions(USol, dim, numMats):
    dif = dim[1] - dim[0]
    maxdev = .5 * (numMats) * math.pow(dif,2)
    for e in USol.getTensor().flatten():
        sol = e.getSol()

        uniques = []; i = 0; notuniques = []
        for s in sol:
            if (i == 0):
                uniques.append(s)
            else:
                isUnique = True
                for k in range(len(uniques)):
                    dev = 0
                    for j in range(len(s)):
                        dev += math.pow(s[j] - uniques[k][j], 2)

                    dev *= .5
                    if (dev <= maxdev): # not a unique solution
                        isUnique = False
                        notuniques.append(solutionTuple(s))
                        break

                if isUnique:
                    uniques.append(s)
                             
            i += 1

        # remove all not unique solutions
        for rep in notuniques:
            e.delete(rep)
        
# print the entire matrix
def printall(USol, dim1, bins):
    i = 0
    for e in USol.getTensor().flatten():
        leftc = i//bins; rightc = i%bins
        left = '%.3f'%(dim1[leftc])
        right = '%.3f'%(dim1[rightc])
        print ("Value for bc's (" + str(left) + ", " +
               str(right) + "): " + str(e))
        i += 1

# print only all the solutions
# Assumes 2 dimensional
def printSols(USol, dim1, bins, dimSol):
    assert bins > 0
    count = 0; i = 0; countsol = 0;
    for e in USol.getTensor().flatten():
        if e.getSol()[0][0] is not None: # solutions exist
            count += 1
            indices = []; vals = []
            for j in range(dimSol):
                indices.append(i//int(math.pow(bins,j)) % bins)
                vals.append('%.2f'%dim1[indices[-1]])

            countsol += str(e).count('(')
            print ("Value for bc's " + str(vals[::-1]) + ": " + str(e))
        i += 1

    print ("There were " + str(count) + " sets of boundary conditions with solutions")
    print ("There were " + str(countsol) + " solutions")

# print only the boundary conditions with solutions
# do not care for the number of solutions
# Assumes 2 dimensional
def printBCs(USol, dim1, bins):
    count = 0; i = 0
    for e in USol.getTensor().flatten():
        if e.getSol()[0][0] is not None: # solutions exist
            count += 1
            leftc = i//bins; rightc = i%bins
            left = '%.3f'%(dim1[leftc])
            right = '%.3f'%(dim1[rightc])
            print ("Value for bc's (" + str(left) + ", " +
                  str(right) + ")")
        i += 1

    print ("There were " + str(count) + " sets of boundary conditions with solutions")

def main():
    # Actual testing time
    # params = [p, delta] fulfilling p = delta + 1

    numMats = 7; dimU = 3
    params = [2,1]
    bins = 40
    simple = False

    alused = al[8:] + al[0:8]
    bound = numMats + dimU - 1 # how many variable names we need - 1
    assert (bound >= 0)
    varnames = list(alused[0:bound]); i = 0
    while (bound > 26): # we want distinct variable names
        bound -= 26
        newvars = [(varnames[j] + str(i)) for j in range(min(bound,26))]
        varnames += newvars
        i += 1

    Us, dim1 = makeAllU(numMats,0,5,bins,params,varnames,dimU)
    print (str(dim1))

    # debugging the steady state tester
    # print (steadyStateTest([0,5,.128],[2,1],dim1))
    # print (steadyStateTest([0,.385,.513],[2,1],dim1))
    # print (steadyStateTest([0,.769,.128],[2,1],dim1))

    # return

    # U12 = matMult(Us[0],Us[1],['i','k','l'])
    # print ("mult 1 done")
    # U123 = matMult(U12,Us[2],['i','l','m'])
    # print ("mult 2 done")
    # U14 = matMult(U123,Us[3],['i','m','n'])
    # print ("mult 3 done")
    # U15 = matMult(U14,Us[4],['i','n','o'])
    # print ("mult 4 done")
    # U16 = matMult(U15,Us[5],['i','o','p'])
    # print ("mult 5 done")
    # USol = matMult(U16,Us[6],['i','q'])

    prod = [];
    for i in range(len(Us)-2):
        if prod: # if not empty
            rightDims = Us[i+1].getDims()[-2:]
            assert len(rightDims) == 2
            allDims = ['i'] + rightDims
            prod = matMult(prod,Us[i+1], allDims, simple)
        else:
            prod = matMult(Us[0],Us[1],['i','k','l'], simple)

        print ("mult " + str(i+1) + " done")

    if prod: # not empty
        prod = matMult(prod,Us[-1],[varnames[0], varnames[-1]], simple) # the final multiplication
    else: prod = matMult(Us[0],Us[1],['i','k','l'], simple)
    reduceSolutions(prod, dim1, numMats)

    # printall(USol, dim1)
    dimSol = 2
    printSols(prod, dim1, bins, dimSol)

start_time = time.time()
main()
print(" SECONDS IN RUN TIME -------- : %s" % (time.time() - start_time))
