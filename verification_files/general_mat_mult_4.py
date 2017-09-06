# UROP Summer
# 6/9/17
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
            self.sol = [sol]
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

    def add(self, tup2): # rightwards add if unique
        j = 0
        if self.sol[0][0] is None:
            self.sol = [tup2.getSol()[0]]
            j = 1

        for i in range(j, len(tup2.getSol())):
            if not(tup2.getSol()[i] in self.sol):
                self.sol.append(tup2.getSol()[i])

    def delete(self, tup): # remove tup from self.sol
        self.sol.remove(tup.getSol()[0])
        if len(self.sol) == 0:
            self.sol = [[None]]
        elif len(self.sol[0]) == 0:
            self.sol == [[None]]
        
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
# params contains other useful variables, like the current spatial coordinates
# val, val1, val2 etc. contain potential steady state solution values
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
            # assert (len(element) == 1)
            row.append(element)
        U.append(row)
    return U, dim

# Make U, but each set of boundary conditions is only allowed
# 1 solution
def makeSmallU(a,b,n,params):
    U = []
    filled = [] # filled boundaries
    dim = [((n-1-i)*a+i*b)/(n-1) for i in range(n)]
    for val1 in dim:
        row = [] #new row for variable
        for val in dim:
            element = []
            for val2 in dim:
                if steadyStateTest([val1,val,val2],params,dim) and ([val1,val2] not in filled): # not hardcoded anymore!
                    element.append(solutionTuple(val))
                    filled.append([val1,val2])
                else: element.append(solutionTuple())
            # assert (len(element) == 1)
            row.append(element)
        U.append(row)
    return U, dim

# If all U's have the same range and dimension
# (In other words, without the labels, all the U's are identical)
# numMats: number of matrices
# a, b, n, params as defined in makeU, although params contains all U's
# varNames: list of all variable names in all the U's
# dimU: the dimension of each U
# paramlen is the length of params for each U
# difparams is a boolean indicating whether each U's params are different or not
def makeAllU(numMats,a,b,n,params,varnames,dimU,paramlen=0,difparams=False):
    assert (numMats + dimU - 1 == len(varnames)) # we have exactly enough varnames
    Us = []
    if not(difparams):
        templateTensor, dim1 = makeU(a,b,n,params)
    for i in range(numMats):
        if difparams:
            templateTensor, dim1 = makeU(a,b,n,params[i*paramlen,(i+1)*paramlen])
        Us.append(labeledTensor(templateTensor, varnames[i:(i+dimU)], [n]*dimU))

    return Us, dim1

def makeAllSmallU(numMats,a,b,n,params,varnames,dimU,paramlen=0,difparams=False):
    assert (numMats + dimU - 1 == len(varnames)) # we have exactly enough varnames
    Us = []
    if not(difparams):
        templateTensor, dim1 = makeSmallU(a,b,n,params)
    for i in range(numMats):
        if difparams:
            templateTensor, dim1 = makeSmallU(a,b,n,params[i*paramlen,(i+1)*paramlen])
        Us.append(labeledTensor(templateTensor, varnames[i:(i+dimU)], [n]*dimU))

    return Us, dim1

# Older version that is more effective
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
    vs = []
    '''for i in range(8):
        vs.append(uleft+math.pow(-1,i)*dif + uright+math.pow(-1,i//2)*dif - 2*(ui+math.pow(-1,i//4)*dif))'''
    if abs(ui - max(dim)) <= .0001:
        for i in range(8):
            vs.append(uleft+math.pow(-1, i)*dif + uright+math.pow(-1, i//2)*dif - 2*(ui + math.pow(-1,i//4)*dif))
    else:
        for i in range(4):
            vs.append(uleft+math.pow(-1,i)*dif + uright+math.pow(-1,i//2)*dif - 2*ui)
    '''if abs(ui - max(dim)) <= .0001 and abs(uleft - max(dim)) <= .0001:
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
            vs.append(uleft-dif + uright+math.pow(-1,i)*dif - 2*(ui-dif))'''

    return len(set(np.sign(vs))) > 1 # multiple signs? There was a 0 in the subcube. One sign? The plane doesn't intersect

    # Fisher equation - new, nonfunctional, version
    '''v = (uleft - 2*ui + uright) + factor*ui*(1-ui) # value at the center of subcube
    grad = [1, -2 + (-2*ui + 1)*factor, 1] # gradient

    mag = 0 # magnitude
    for i in grad:
        mag += math.pow(i,2)
    mag = math.pow(mag,.5)
    maxchange = mag * dif # the dot of the gradient with itself is mag^2, divided by mag and multiplied by dif
    return (abs(v) < maxchange)'''

    # Old Fisher checker
    num1 = uleft + uright - 2*ui  # u_{i+1} + u_{i-1} - 2*u_i
    num2 = 5*ui * (1 - ui)  # u_i(1-u_i)
    num1 *= factor
    return (abs(roundtores(num1, dim) + roundtores(num2, dim) - roundtores(0, dim)) < .00001)


# Generalized matrix multiplication
# A times B (labelled tensors)
def matMult(A, B, exposed, abridge=False):
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

    # THIS IS THE CORRECTED
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
            prod = subela.mult(subelb) # changes subela
            if (not (subela.getSol()[0][0] is None) and not (subelb.getSol()[0][0] is None)):
                el.add(prod)

            # Only want 1 of the solutions in el
            if abridge:
                print ("Hi what's up update me later")

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
    maxdev = .5 * (numMats) * math.pow(dif, 2)
    for e in USol.getTensor().flatten():
        sol = e.getSol()

        uniques = []
        i = 0
        notuniques = []
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
                    if (dev <= maxdev):  # not a unique solution
                        isUnique = False
                        notuniques.append(solutionTuple(s))
                        break

                if isUnique:
                    uniques.append(s)

            i += 1

        # remove all not unique solutions
        for rep in notuniques:
            e.delete(rep)

# restrict the bound of solutions, returning it to trueBounds
def boundSolutions(USol, tb, b):
    for e in USol.getTensor().flatten():
        sol = e.getSol()

        invalids = []
        for s in sol: # for every solution for this boundary
            for val in s: # for every value in the solution
                if (val > tb[1] + b) or (val < tb[0] - b): # if the value is outside the true bounds
                    invalids.append(solutionTuple(s)) # is invalid solution
                    break

        for inv in invalids:
            e.delete(inv) # delete invalid solution

def convertToThousanths(arr):
    toRtn = []
    for el in arr:
        toRtn.append(round(el,3))

    return toRtn

# print only all the solutions
# Assumes 2 dimensional
def printSols(USol, dim1, bins):
    assert bins > 0
    count = 0; countsol = 0;
    b = dim1[1] - dim1[0]
    solsDict = {}
    for e in USol.getTensor().flatten():
        sols = e.getSol()
        if (sols[0][0] is not None): # solutions exist
            count += 1
            boundaries = []
            for s in sols:
                boundaries.append(convertToThousanths([s[0],s[-1]])) # the new boundary conditions

            assert len(boundaries) == len(sols)
            for i in range(len(boundaries)):
                ind = ((boundaries[i][0]+.00001)//b)*bins + (boundaries[i][1]+.00001)//b
                if ind in solsDict.keys():
                    solsDict[ind].add(solutionTuple(convertToThousanths(sols[i])[1:-1]))
                else:
                    solsDict[ind] = solutionTuple(convertToThousanths(sols[i][1:-1]))

            '''for j in range(dimSol):
                indices.append(i//int(math.pow(bins,j)) % bins)
                vals.append('%.2f'%dim1[indices[-1]])''' # old boundary calculation

            countsol += str(e).count('(')

    print ("There were " + str(count) + " sets of boundary conditions with solutions")
    print ("There were " + str(countsol) + " solutions")
    print ("The solutions are: ")
    for key, value in sorted(solsDict.items()):
        bounds = [dim1[int(key//bins+.00001)], dim1[int(key%bins+.00001)]]
        print ("Value for bc's " + str(bounds) + ": " + str(value))

# old printing method
def oldPrintSols(USol, dim1, bins, dimSol):
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

def main():
    numMats = 8; dimU = 3
    params = []
    trueBins = 41; trueBounds = [0,2]; trueDim = [((trueBins-1-i)*trueBounds[0]+i*trueBounds[1])/(trueBins-1) for i in range(trueBins)]
    b = (trueBounds[1]-trueBounds[0])/(trueBins-1) # bin size
    adjust = (1/(numMats-2)) * (trueBounds[1]-trueBounds[0]) # amount by which to adjust bounds
    rawRang = [trueBounds[0]-adjust, trueBounds[1]+adjust] # raw range - have to adjust it by bins
    rang = [roundtores(rawRang[0],[1,1+b]),roundtores(rawRang[1],[1,1+b])] # true range: has possible bin values as bound ends
    bins = round(trueBins + (trueBounds[0]-rang[0])//b + (rang[1]-trueBounds[1])//b )# the number of bins to run on is the number of bins in the small range, plus twice the number
    # of bins in the adjustment value

    alused = al[8:] + al[0:8]
    bound = numMats + dimU - 1 # how many variable names we need - 1
    assert (bound >= 0)
    varnames = list(alused[0:bound]); i = 0
    while (bound > 26): # we want distinct variable names
        bound -= 26
        newvars = [(varnames[j] + str(i)) for j in range(min(bound,26))]
        varnames += newvars
        i += 1

    Us, dim1 = makeAllU(numMats,rang[0],rang[1],bins,params,varnames,dimU)

    prod = [];
    for i in range(len(Us)-2):
        if prod: # if not empty
            rightDims = Us[i+1].getDims()[-2:]
            assert len(rightDims) == 2
            allDims = ['i'] + rightDims
            prod = matMult(prod,Us[i+1], allDims)
        else:
            prod = matMult(Us[0],Us[1],['i','k','l'])

        print ("mult " + str(i+1) + " done")

    if prod: prod = matMult(prod,Us[-1],[varnames[0], varnames[-1]]) # the final multiplication
    else: prod = matMult(Us[0], Us[1], ['i','k','l'])
    reduceSolutions(prod, dim1, numMats)
    boundSolutions(prod, trueBounds, b)

##    U12 = matMult(U1,U2,['i','k','l'])
##    print ("mult 1 done")
##    U123 = matMult(U12,U3,['i','l','m'])
##    print ("mult 2 done")
##    U14 = matMult(U123,U4,['i','m','n'])
##    print ("mult 3 done")
##    U15 = matMult(U14,U5,['i','n','o'])
##    print ("mult 4 done")
##    U16 = matMult(U15,U6,['i','o','p'])
##    print ("mult 5 done")
##    USol = matMult(U16,U7,['i','q'])

    printSols(prod, trueDim, bins)

start_time = time.time()
main()
print(" SECONDS IN RUN TIME -------- : %s" % (time.time() - start_time))
