# UROP 2017-2018

import itertools
import numpy as np
import random as rd
import time
import math
from string import ascii_lowercase as al
import matplotlib.pyplot as plt

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
        if len(self.sol) == 0:
            self.sol = [[None]] # default empty
        
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
    dim = [round(((n-1-i)*a+i*b)/(n-1), 3) for i in range(n)]
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

# num: number to round
# r: resolution
# c: "offset" of the bin values (e.g. [.1, .6, 1.1] has offset of .1 from 0)
def roundtores(num, r, c): #resolution n
    i = (num - c)/r # solve for i
    roundi = round(i+.00001) # round i to the nearest integer, round up
    return r*roundi+c

# num: number to round
# r: resolution
# bounds: upper and lower bounds (inclusive).
# offset: c in the expression below
# returns an index i such that |num - r*i+c| is minimized, if num is within bounds
def roundtoresIndex(num, r, bounds, offset):
    if ((num + .00001) >= bounds[0]) and ((num - .00001) <= bounds[1]):
        return round((num - offset)/r + .00001)
    return -1

# bound: original bin bounds
# r: resolution (difference between adjacent bins)
# n: number of matrices in product
# returns: new bin bounds, number of added values (even integer)
def expand(bound, r, n):
    # this code is almost identical to that in Fall's PASS
    print ("Expand boiz")
    adjust = (1/(n-2)) * (bound[1] - bound[0])
    rawRange = [bound[0] - adjust, bound[1] + adjust]
    rang = [roundtores(rawRange[0], r, bound[0]), roundtores(rawRange[1], r, bound[0])]

    valsAdded = round((bound[0] - rang[0])/r) * 2

    return rang, valsAdded

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
    # return heatVerticesOld(ui,uleft,uright,dif,dim)
    # return heatEps(ui,uleft,uright,h,factor,dif)

    # Fisher Equation
    # return fisher(ui,uleft,uright,dif,h,factor,dim,params)

    # Functional W-J code
    # return WJTest(ui,uleft,uright,dif,h,factor,params,dim)

    # Benjamin-Bona-Mahony
    # return bbmEps(ui,uright,h,dif)

    # Sine-Gordon equation
    return sgEps(ui,uleft,uright,h,factor,dif)

def heatEps(ui,uleft,uright,h,factor,dif):
    val = (uright - 2*ui + uleft)*factor
    eps = dif*2*math.sqrt(3)*factor
    return abs(val) < eps

def heatVerticesOld(ui,uleft,uright,dif,dim):
    vs = []
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
    return len(set(np.sign(vs))) > 1 # multiple signs? There was a 0 in the subcube. One sign? The plane doesn't intersect

def fisher(ui,uleft,uright,dif,h,factor,dim,params):
    mu = params[0]
    binSize = dim[1]-dim[0]
    minVal = dim[0]

    num1 = uleft + uright - 2*ui  # u_{i+1} + u_{i-1} - 2*u_i
    num2 = mu*ui * (1 - ui)  # u_i(1-u_i)
    num1 *= factor
    return (abs(roundtores(num1,binSize,minVal) + roundtores(num2,binSize,minVal) - roundtores(0,binSize,minVal)) < .00001)

def fisherEps(ui,uleft,uright,h,factor,dif,params):
    mu = params[0]
    if (-2 + h**2*(1 - 2*ui)) <= 0:
        magGrad = math.sqrt(2 + (-2 + h**2*(1-2*ui-dif*2))**2)
    else: magGrad = math.sqrt(2 + (-2 + h**2*(1-2*ui+dif*2))**2)
    val = (uright - 2*ui + uleft)*factor + mu*ui*(1-ui)
    epsilon = dif*math.sqrt(3)*factor*magGrad
    return abs(val) < epsilon

def WJTest(ui,uleft,uright,dif,h,factor,params,dim):
    p = float(params[0]); delta = float(params[1])
    if ((p < 0) or (delta < 0)) and ((ui == 0) or (uleft == 0)): # check for negative powers of 0
        return False
    if (abs(p - round(p)) > .0001 and uleft < 0) or (abs(delta - round(delta)) > .0001 and (ui < 0 or uleft < 0)):
        return False

    binSize = dim[1]-dim[0]
    source = math.pow(uleft,p)
    diffusion = (math.pow(ui,delta)*(uright - ui) - math.pow(uleft,delta)*(ui - uleft)) * factor
    return (abs(roundtores(source + diffusion, binSize, dim[0])) < .00001)

def bbm(ui,uleft,uright,h,factor,dim):
    ux = (uright - ui)/h
    return (abs(roundtores(ux*(1 + ui), dim[1]-dim[0],dim[0])) < .00001)

def bbmEps(ui, uright, h, dif):
    val = (uright - ui)/h * (1 + ui)
    if (ui <= -.6):
        magGrad = math.sqrt((1+ui-dif)**2 + (2*(ui-dif)+1)**2)/h
    else:
        magGrad = math.sqrt((1+ui+dif)**2 + (2*(ui+dif)-1)**2)/h
    eps = abs(magGrad * dif * math.sqrt(3))
    return abs(val) < eps

def sg(ui,uleft,uright,h,factor,dim):
    s = math.sin(ui) # assume ui in radians
    uxx = (uright - 2*ui + uleft)*factor
    return (abs(roundtores(s-uxx,dim[1]-dim[0],dim[0])) < .00001)

def sgEps(ui,uleft,uright,h,factor,dif):
    val = math.sin(ui) - (uright - 2*ui + uleft)*factor
    # we use roundtoresIndex to find k such that k*pi is the largest
    # multiple of pi less than ui
    k = roundtoresIndex(ui,math.pi,[float("-inf"), float("inf")],0)
    if (k%2 == 0):
        magGrad = factor*math.sqrt(2 + math.pow(((h**2)*math.cos(ui-dif) + 2),2))
    else:
        magGrad = factor*math.sqrt(2 + math.pow(((h**2)*math.cos(ui+dif) + 2), 2))
    eps = magGrad * dif * math.sqrt(3)
    return abs(val) < eps

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

# Determining whether two solutions for the same set of boundaries are identical
# Algorithm: difference between two solutions is less than
# 1/2 * (number of variables) * (dif)^2 where dif is
# the difference between two adjacent bins (possible solution values)
def reduceSolutions(USol, dim, numMats):
    dif = dim[1] - dim[0]
    maxdev = .5 * (numMats) * math.pow(dif,2)
    for e in USol.getTensor().flatten():
        sol = e.getSol()
        # print ("This is what a solution looks like in reduceSolutions:")
        # print (sol)

        uniques = []; i = 0; notuniques = []
        if sol[0][0] is not None:
            for s in sol:
                if (i == 0):
                    uniques.append(s); i+=1
                else:
                    if isUnique(s,uniques,maxdev):
                        uniques.append(s)
                    else: notuniques.append(solutionTuple(s))

            # remove all not unique solutions
            for rep in notuniques:
                e.delete(rep)

# subroutine for reducing solutions
# determines whether a solution is "unique": significantly
# different from all the other solutions in uniques
def isUnique(sol, uniques, maxdev):
    for unique_sol in uniques:
        dev = 0
        for j in range(len(sol)):
            dev += math.pow(sol[j] - unique_sol[j],2)

        dev *= .5
        if (dev <= maxdev): # if solution is too similar any: not unique
            return False

    return True

# Remove all solutions containing values outside true range
# Removal code should be similar to that of reduceSolutions
# DO NOT remove things within .00001 (or other arbitrary arithmetic error bound) of actual bounds
# tb: true bounds
def boundSolutions(USol, tb):
    for e in USol.getTensor().flatten():
        sol = e.getSol()

        invalids = []
        for s in sol: # for every solution for this boundary
            for val in s: # for every value in the solution
                if not(val is None):
                    if (val > tb[1] + 0.00001) or (val < tb[0] - 0.00001): # if the value is outside the true bounds
                        invalids.append(solutionTuple(s)) # is invalid solution
                        break

        for inv in invalids:
            e.delete(inv) # delete invalid solution
        
# print the entire matrix
# basically useless
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
# Assumes 1D solutions
def printSols(USol):
    # assert bins > 0
    count = 0; countsol = 0;

    for e in USol.getTensor().flatten():
        sols = e.getSol()
        if sols[0][0] is not None: # solutions exist
            count += 1
            boundaries = []
            for s in sols: # Fix this, probably by fixing reduction/directly changing boundaries here
                boundaries.append([s[0],s[-1]]) # boundary conditions: not actually on the boundary!

            '''for j in range(dimSol):
                indices.append(i//int(math.pow(bins,j)) % bins)
                vals.append('%.2f'%dim1[indices[-1]])''' # old boundary calculation

            countsol += str(e).count('(')
            assert len(boundaries) == len(sols)
            for i in range(len(boundaries)):
                print ("Value for bc's " + str(boundaries[i]) + ": " + str(sols[i][1:-1]))

    print ("There were " + str(count) + " sets of boundary conditions with solutions")
    print ("There were " + str(countsol) + " solutions")

# a modified version of printSols, that only prints solutions if they are
# considered unique without taking into account initial boundaries
# we use the same uniqueness-detection subroutine as reduceSolutions
# treat boundaries as keys in a dictionary
def printUniqueSols(USol, dim, numMats):
    # assert bins > 0
    dif = dim[1] - dim[0]
    maxdev = .5 * (numMats) * math.pow(dif,2)

    count = 0; countsol = 0;
    all_boundaries = dict()
    for e in USol.getTensor().flatten():
        sols = e.getSol()
        if sols[0][0] is not None: # solutions exist
            for s in sols:
                b = (s[0],s[-1]) # get the tuple of boundary conditions
                # print ("Here is s in printUniqueSols!")
                # print (s)
                if b in all_boundaries:
                    uniques = all_boundaries[b]
                    if (isUnique(s[1:-1],uniques,maxdev)):
                        all_boundaries[b].append(s[1:-1])
                        countsol += 1
                else:
                    all_boundaries[b] = [s[1:-1]]
                    count += 1
                    countsol += 1

            '''for j in range(dimSol):
                indices.append(i//int(math.pow(bins,j)) % bins)
                vals.append('%.2f'%dim1[indices[-1]])''' # old boundary calculation

            # countsol += str(e).count('(')
            # assert len(all_boundaries) == len(sols)
    boundaries = all_boundaries.keys()
    for b in boundaries:
        sols  = all_boundaries[b]
        print ("Value for bc's " + str(b) + ": " + str(sols))

    print ("There were " + str(count) + " sets of boundary conditions with solutions")
    print ("There were " + str(countsol) + " solutions")
    return boundaries

# print only the boundary conditions with solutions
# do not care for the number of solutions
# Assumes 1D solutions
def printBCs(USol, dim1):
    count = 0; fulfilled = []
    for e in USol.getTensor().flatten():
        sols = e.getSol()
        if sols[0][0] is not None: # solutions exist
            count += 1
            boundaries = []
            for s in sols:
                if not([s[0], s[-1]] in fulfilled):
                    boundaries.append([s[0], s[-1]])
                    fulfilled.append([s[0], s[-1]])

            '''leftc = i//bins; rightc = i%bins
            left = '%.3f'%(dim1[leftc])
            right = '%.3f'%(dim1[rightc])'''

    boundaries.sort()
    print ("Fullfilled boundaries:")
    for b in boundaries:
        print (str(b))

    print ("There were " + str(count) + " sets of boundary conditions with solutions")

# convert a USol to a 2D boolean numpy array
# explicitly takes into account the extra bins from range expansion,
# and ignores anything extra, taking advantage of array structure
def convertToPlot(USol, trueRang, trueBins, bins):
    rawArray = []; el = []
    padBins = (bins-trueBins)/2; hPadCount = 0; vPadCount = 0
    i = 0
    for e in USol.getTensor().flatten():
        if (i-1)//bins != i//bins and i != 0: # start new row
            if (vPadCount >= padBins) and (vPadCount < bins-padBins):
                rawArray.append(el)
                el = []
            hPadCount = 0
            vPadCount += 1

        if (vPadCount >= padBins) and (vPadCount < bins-padBins): # if within trueRang for 1 variable
            if (hPadCount >= padBins) and (hPadCount < bins-padBins): # if within trueRang for the other variable
                if e.getSol()[0][0] is not None: # solutions exist
                    el.append(1) # True
                else: # solutions don't exist
                    el.append(0)

        i += 1; hPadCount += 1

    # rawArray.append(el) # don't forget the last row
    assert len(rawArray) == trueBins
    assert (len(rawArray[0])) == trueBins
    toRtn = np.array(rawArray) # convert to numpy

    return toRtn

# converts a list of fulfilled _real_ boundaries to a pixel array
# bs: list of boundary tuples
# trueRang: the desired range of boundary values for which we want to see the presence of solutions
# r: resolution
# assume for now same range for both (all) boundary variables
def newConvert(bs, trueRang, trueBins, r):
    plot = np.zeros((trueBins,trueBins))
    offset = 0
    for b in bs:
        # first element is row #, second is col #
        xi = roundtoresIndex(b[0], r, trueRang, offset)
        yi = roundtoresIndex(b[1], r, trueRang, offset)
        plot[xi][yi] = 1

    return plot

def main():
    # Actual testing time
    # params = [p, delta] fulfilling p = delta + 1

    numMats = 8; dimU = 3
    params = [1,.5]
    trueRang = [0,1]; trueBins = 11
    reso = (trueRang[1] - trueRang[0])/(trueBins-1)
    rang, addedBins = expand(trueRang,reso,numMats) # bounds, resolution, ""
    bins = trueBins + addedBins
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

    print (rang[0], rang[1], bins)
    Us, dim1 = makeAllU(numMats,rang[0],rang[1],bins,params,varnames,dimU)
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
    boundSolutions(prod, trueRang)

    # printall(USol, dim1)
    boundaries = printUniqueSols(prod, dim1, numMats) # return fulfilled boundaries
    # print (boundaries)
    input("Please press enter to continue ")
    pixelArray = newConvert(boundaries, trueRang, trueBins, reso)
    # pixelArray = convertToPlot(prod,trueRang,trueBins,bins) # make the corresponding boolean array
    maskedPA = np.ma.masked_where(pixelArray == 1, pixelArray)

    fig, ax = plt.subplots(figsize=(6, 6)) # scaling plot axes
    cmap = plt.cm.gray
    cmap.set_bad(color='yellow')
    # print (dim1[-1])
    # plot = ax.imshow(pixelArray, interpolation='none', extent=[dim1[0], dim1[-1], dim1[-1], dim1[0]])
    plot = ax.imshow(maskedPA, cmap=cmap, interpolation='none', extent=[trueRang[0], trueRang[-1], trueRang[-1], trueRang[0]])
    # plot = plt.imshow(pixelArray)  # draw plot
    plt.show(plot) # show plot
    # input("Please press enter to continue")

start_time = time.time()
main()
print(" SECONDS IN RUN TIME -------- : %s" % (time.time() - start_time))
