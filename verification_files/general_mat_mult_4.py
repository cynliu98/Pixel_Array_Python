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

    def add(self, tup2): # rightwards add
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

# rounding to resolution
# Version that is more consistent in behavior, but that has been put aside as not effective
# def roundtores(num, dim): #resolution n
    # dif = 1000000000 #no practical problem would have a dif this big
    # best = -1
    # if (min(dim) - num > (dim[1] - dim[0])/2 + .00001 or num - max(dim) > (dim[1]-dim[0])/2):
        # if our number is too far out of range
        # return math.inf
    # for val in dim:
        # if (abs(num - val) < dif + 0.00001):
            # dif = abs(num - val)
            # best = dim.index(val)
        # else: #shape is down then up. Once we've gone up, done
            # return dim[best]
    # return dim[best]


# Older version that is more effective
def roundtores(num, dim): #resolution n
    dif = 1000000000 #no practical problem would have a dif this big
    best = -1
    if ((min(dim) - num) > (dim[1] - dim[0])/2 or (num - max(dim)) > (dim[1]-dim[0])/2):
        # if our number is too far out of range
        return math.inf
    for val in dim:
        if (abs(num - val) < dif + .00001): # arbitrarily choose to round up or down
            dif = abs(num - val)
            best = dim.index(val)
        else: #dif gets lower and then higher. Once we've gone up, done
            return dim[best]
    return dim[best]

# Thank you Stack Overflow
def removeDupsOrder(vars):
    seen = set()
    seen_add = seen.add
    return [x for x in vars if not (x in seen or seen_add(x))]

# Psuedocode for new steady state testing
# Isolate the variable you want to return in the equation
# Is there a value within d/2 (not inclusive lower inclusive upper for
# upwards rounding, opposite for downs)of the given bin such that
# the expression involving u_i could equation the expression
# involving everything else?
# If yes, return True
# Note that this is the traditional pixel array method. However, it can become inefficient.
# Emailed David 7/26/17

# Test for the steady state. Varies for the PDE
# Used in making the U matrix
# Comment out irrelevant assert statements
# params contain other variables as described in makeU
def steadyStateTest(orderedvars,params,dim):
    assert(len(orderedvars) == 3)
    h = .1 # interval size

    # heat equation
    # return (abs(roundtores((orderedvars[0] + orderedvars[2])/2, dim) - orderedvars[1]) < .00001)
    return (abs(roundtores((orderedvars[0] + orderedvars[2])/2, dim) - orderedvars[1]) < .00001)

    # Fisher equation
    # see if the new method works
    # still somewhat sketchy
    # can we assume equations with 0 as known steady state will include 0 in bin range
    dim = [i*math.pow(h,-2) for i in dim]
    num1 = -orderedvars[0] * math.pow(h, -2)
    num2 = (orderedvars[2] - 2*orderedvars[1]) * math.pow(h, -2) + 2 * orderedcars[1]*(1 - orderedvars[1])
    # num1 = 2*orderedvars[1] - orderedvars[0] - orderedvars[2] #2u_i - u_{i+1} - u_{i-1}
    # num2 = 2 * orderedvars[1] * (1 - orderedvars[1]) #u_i(1-u_i)
    # num1 *= -1 * math.pow(h, -2)
    return (abs(roundtores(num1, dim) - roundtores(num2, dim)) < .00001)

    # Newell-Whitehead-Segel
##    num1 = 2*orderedvars[1] - orderedvars[0] - orderedvars[2]
##    num1 *= 5 * math.pow(h, -2)
##    num2 = orderedvars[1] * (1 - math.pow(orderedvars[1],2))
##    return (abs(roundtores(num1, dim) - roundtores(num2, dim)) < .00001)

    # Zeldovich–Frank–Kamenetsky
##    num1 = 2*orderedvars[1] - orderedvars[0] - orderedvars[2]
##    num1 *= 1 * math.pow(h, -2)
##    num2 = orderedvars[1]*(1 - orderedvars[1])*(orderedvars[1])
##    return (abs(roundtores(num1, dim) - roundtores(num2, dim)) < .00001)

# Generalized matrix multiplication
# A times B (labelled tensors)
def matMult(A, B, exposed):
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

        uniques = [];
        i = 0;
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

def main():

    numMats = 10; dimU = 3
    params = []
    bins = 50

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

    prod = matMult(prod,Us[-1],[varnames[0], varnames[-1]]) # the final multiplication
    reduceSolutions(prod, dim1, numMats)

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

    count = 0; i = 0; countsol = 0;
    for e in prod.getTensor().flatten():
        if e.getSol()[0][0] is not None: # solutions exist
            count += 1
            leftc = i//bins; rightc = i%bins
            left = '%.3f'%(dim1[leftc])
            right = '%.3f'%(dim1[rightc])
            countsol += str(e).count('(')
            print ("Value for bc's (" + str(left) + ", " +
                  str(right) + "): " + str(e))
        i += 1
        if (i>=1000 and i%1000 == 0): input("Press enter once you're down copying")


    print ("There were " + str(count) + " sets of boundary conditions with solutions")
    print ("There were " + str(countsol) + " solutions")

start_time = time.time()
main()
print(" SECONDS IN RUN TIME -------- : %s" % (time.time() - start_time))