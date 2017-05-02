#UROP Spring 2017
#5/2/17
#Generalized matrix multiplication

import itertools
import numpy as np

class labeledTensor(object):
    def __init__(self, mat, dims, resos):
        assert (len(list(mat.shape)) == len(dims)) # the dims can actually match
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

    def extractDim(self,dim): # extract possible values of a dimension
        assert(dim in self.dims)
        nestednum = self.dims.index(dim)
        vals = []
        # go only as deep as nestednum states
        # basically if nestednum = 0 we want self.tensor[i][0]...[0]
        # where there are as many [v]'s are there are dimensions
        # = len(dims)
        # also i ranges from 0 to the number of possible values
        assert (len(vals) == self.resolutions[nestednum]) # number of values is same as resolution
        return vals

    def getReso(self,dim): # get resolution of dimension
        assert (len(extractDim(dim)) == self.resolutions[self.dims.index(dim)])
        return self.resolutions[self.dims.index(dim)]

# n possible solution values y for each variable
# NOTE: n IS NOT THE NUMBER OF VARIABLES
# from steady state program
def makeU(a,b,n):
    U = []
    dim = [((n-1-i)*a+i*b)/(n-1) for i in range(n)]
    for val1 in dim:
        row = [] #new row for variable 
        for val in dim:
            element = []
            for val2 in dim:
                if abs(val - roundtores((val1+val2)/2, dim)) < .0001: #float errors
                    element.append([val])
                else: element.append([])
            row.append(element)
        U.append(row)
    return U, dim

# rounding to resolution - from steady state program
def roundtores(num, dim): #resolution n
    dif = 1000000000 #no practical problem would have a dif this big
    best = -1
    for val in dim:
        if (abs(num - val) < dif):
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
def steadyStateTest(vars): 
    pass

# Generalized matrix multiplication
# A times B (labelled tensors)
def matMult(A,B,exposed):
    avars = A.dims; bvars = B.dims;
    aresos = A.resolutions; bresos = B.resolutions
    allvar = removeDupsOrder(avars + bvars) # in order

    allresos = []
    for v in allvar:
        if v in avars:
            allresos.append(aresos[avars.index(v)])
        elif v in bvars:
            allresos.append(bresos[bvars.index(v)])
    
    for e in exposed: # all exposed variables must be variables
        assert(e in allvar)
    indexers = [i for i in allvar if not (i in exposed)] # in order
##    print ("avars " + str(avars))
##    print ("bvars " + str(bvars))
##    print ("exposed vars " + str(exposed))
##    print ("indexers " + str(indexers))
##    print ("All variables " + str(allvar))
##    print ("All resolutions " + str(allresos)) # also in order      

    matA = A.tensor; matB = B.tensor
    exposedi = [allvar.index(e) for e in exposed]
    indexi = [allvar.index(i) for i in indexers]
    exresos = [allresos[e] for e in exposedi]
    inresos = [allresos[i] for i in indexi]
##    print ("exposed indices: " + str(exposedi))
##    print ("indexer indicies: " + str(indexi))
    
    # Initialize empty numpy array (numpy.empty) with shape
    # Number of dimensions = Number of exposed variables + 1
    # Length of each dimension = resolution of that variable
    # Note that the final dimension is for containing the multiple variables
    # In a solution for a given set of boundary conditions
    rawshape = []
    for i in exposedi:
        rawshape.append(allresos[i])

    rawshape.append(1)
    shape = tuple(rawshape)
    # print ("Shape of solution array: " + str(shape))
    Usol = np.zeros(shape)
    
    # for each set of boundary conditions in Usol:
    for bci in np.ndenumerate(Usol):
        bci = list(list(bci)[0])[0:2] #extract tuple of bc indices
        # bc = the bci[0] through bci[n] values of exposed
        bcallv = []
        for v in exposed:
            if (v in avars):
                bcallv.append(A.extractDim(v))
            elif (v in bvars):
                bcallv.append(B.extractDim(v))
        assert (len(bci) == len(bcallv))
        # from the possible values of each dimension bcallv[i],
        # extract the bci[i]th value (the index of the boundary condition value)
        bc = [bcallv[i][bci[i]] for i in range(len(bci))]
        
        # print (str(bc))
    #    for tup in indexers:
    #        if (steadyStateTest(matA[bc][tup]) and steadyStateTest(matB[bc][tup])):
    #            
    #    tensor[right indices] = set(tensor[right indices])

    #return labeledTensor(Usol, exposed, exresos)
    return "foo"

def main():
    # basic testing - initialize a labeledTensor
    blargh = np.array([[[1,2,3]]])
    dimsblargh = ['fake', 'more fake', 'cnn']
    resosgood = [4,4,4]
    #resosbad = [100]
    test = labeledTensor(blargh,dimsblargh,resosgood)
    #testbad = labeledTensor(blargh,dimsblargh,resosbad)

    # basic testing - getter methods
##    print (test.getTensor())
##    print (test.getDims())
##    print (test.getResos())

    # Actual testing time
    rU1, dim1 = makeU(0,5,7)
    rU2, dim2 = makeU(0,5,7) # symmetric
    rU1 = np.array(rU1)
    rU2 = np.array(rU2)
    # print (U1)

    U1 = labeledTensor(rU1, ['i','j','k'], [7,7,7])
    U2 = labeledTensor(rU2, ['j','k','l'], [7,7,7])

    foo = matMult(U1,U2,['i','l'])

main()
