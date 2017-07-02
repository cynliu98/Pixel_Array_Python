#UROP Spring
#6/7/17
#Generalized matrix multiplication

import itertools
import numpy as np
import random as rd
import time

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
        for j in range(len(self.sol)):
            sol = self.sol[j]
            for i in range(len(sol)):
                if sol[i] is None:
                    tortn += str(sol[i])
                else: tortn += str('%.3f'%(sol[i]))
                if i != len(sol)-1:
                    tortn += ", "
            tortn += ")"
            if j != len(self.sol)-1:
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
        
    def mult(self, tup2): # rightwards multiply
        if (not (self.sol[0][0] is None) and not (tup2.getSol()[0][0] is None)):
            newself = []
            bothlists = [self.sol, tup2.getSol()]
            for e in itertools.product(*bothlists):
                e = list(e)
                e = [item for sublist in e for item in sublist]
                newself.append(e)
            # print (newself)
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

# rounding to resolution - from steady state program
def roundtores(num, dim): #resolution n
    dif = 1000000000 #no practical problem would have a dif this big
    best = -1
    for val in dim:
        if (abs(num - val) < dif + 0.00001):
            dif = abs(num - val)
            best = dim.index(val)
        else: #shape is down then up. Once we've gone up, done
            return dim[best]
    return dim[best]

# Thank you Stack Overflow
def removeDupsOrder(vars):
    seen = set()
    seen_add = seen.add
    return [x for x in vars if not (x in seen or seen_add(x))]

# Test for the steady state. Varies for the PDE
# Used in making the U matrix
# Comment out irrelevant assert statements
# params contain other 
def steadyStateTest(orderedvars,params,dim):
    assert(len(orderedvars) == 3)
    return (abs(roundtores((orderedvars[0] + orderedvars[2])/2, dim) - orderedvars[1]) < .0001)

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
    

def main():
    # Actual testing time
    params = []
    rU1, dim1 = makeU(0,5,40, params)
    rU2, dim2 = makeU(0,5,40, params) # symmetric
    rU1 = np.array(rU1)
    rU2 = np.array(rU2)
    # print (rU1)
    
##    a = solutionTuple([1,2,3,4,5])
##    z = solutionTuple()
##    print ("Nothing: " + str(z))
##    print (str(a))
##    b = solutionTuple(2)
##    a.add(b)
##    # a.mult(b)
##    print (str(a))
##    print (str(a.mult(b)))

    U1 = labeledTensor(rU1, ['i','j','k'], [40,40,40])
    U2 = labeledTensor(rU2, ['j','k','l'], [40,40,40])
    U3 = labeledTensor(rU1, ['k','l','m'], [40,40,40])
    U4 = labeledTensor(rU1, ['l','m','n'], [40,40,40])
    U5 = labeledTensor(rU2, ['m','n','o'], [40,40,40])
    U6 = labeledTensor(rU1, ['n','o','p'], [40,40,40])
    U7 = labeledTensor(rU1, ['o','p','q'], [40,40,40])

##    U12 = matMult(U1,U2,['i','k','l'])
##    print ("mult 1 done")
##    U34 = matMult(U3,U4,['k','l','m','n'])
##    print ("mult 2 done")
##    Ulefts = matMult(U12,U34,['i','m','n'])
##    print ("mult 5 done")
##    U56 = matMult(U5,U6,['m','n','o','p'])
##    print ("mult 3 done")
##    U567 = matMult(U56,U7,['m','n','q'])
##    print ("mult 4 done")
##    USol = matMult(Ulefts,U567,['i','q'])
##    print ("done")

    U12 = matMult(U1,U2,['i','k','l'])
    print ("mult 1 done")
    U123 = matMult(U12,U3,['i','l','m'])
    print ("mult 2 done")
    U14 = matMult(U123,U4,['i','m','n'])
    print ("mult 3 done")
    U15 = matMult(U14,U5,['i','n','o'])
    print ("mult 4 done")
    U16 = matMult(U15,U6,['i','o','p'])
    print ("mult 5 done")
    USol = matMult(U16,U7,['i','q'])
    
    i = 0
##    for e in USol.getTensor().flatten():
##        leftc = i//40; rightc = i%40
##        left = '%.3f'%(dim1[leftc])
##        right = '%.3f'%(dim1[rightc])
##        print ("Value for bc's (" + str(left) + ", " +
##               str(right) + "): " + str(e))
##        i += 1
    for i in range(10):
        leftc = rd.randint(0,39); rightc = rd.randint(0,39)
        left = '%.3f'%(dim1[leftc])
        right = '%.3f'%(dim1[rightc])
        print ("Value for bc's (" + str(left) + ", " +
               str(right) + "): " + str(USol.getTensor().flatten()[40*leftc+rightc]))
    print ("Finished")
    # print (str(USol.getTensor()))

start_time = time.time()
main()
print(" SECONDS IN RUN TIME -------- : %s" % (time.time() - start_time))
