# UROP Summer 2017
# July 13
# A file for miscellaneous forms of solution analysis

import math

def readSolutions(fname):
    f = open(fname, 'r') # open the file of solutions
    sols = []; bcs = []
    for l in f:
        bcraw = l[(l.index('(')+1):(l.index(')'))]
        bcraw = bcraw.split(',')
        bc = []
        for ele in bcraw:
            ele.strip()
            bc.append(float(ele))

        if (bc not in bcs):
            # print (bc)
            bcs.append(bc)

        unparsedSols = l[(l.index(':')+4):-2] # remove all brackets etc.
        # print (unparsedSols)
        bcsol = []
        try:
            while True:
                i = unparsedSols.index(')') # while there are more solutions
                unparsedSol = unparsedSols[:i]

                solNums = unparsedSol.split(',')
                sol = []
                for n in solNums:
                    n.strip()
                    sol.append(float(n))

                bcsol.append(sol)
                unparsedSols = unparsedSols[(unparsedSols.index('(')+1):]

        except:
            sols.append(bcsol)

    return sols, bcs

# Detect whether there are solutions indistinguishable
# from another, fixed solution
def detectSimilar(sol, dim, template):
    assert len(sol) == len(template)
    dif = dim[1] - dim[0]
    maxdev = .5 * (len(sol)) * math.pow(dif, 2)
    dev = 0
    for j in range(len(sol)):
        dev += math.pow(sol[j] - template[j], 2)

    dev *= .5
    if (dev <= maxdev):  # not a unique solution
        print (dev)
        return True

    return False

# Intersection of 2D arrays
def intersect(arr1, arr2):
    tortn = []
    for a in arr1:
        if a in arr2:
            tortn.append(a)

    return tortn

# Returns all solutions that are shared among
# Two sets of solutions with the same bins
# Assume solutions have been read
def returnOverlap(sols1, bcs1, sols2, bcs2):
    assert (len(sols1) == len(bcs1)) # solution is 2D array, each element with solutions corresponds to a boundary
    bcs = intersect(bcs1, bcs2)

    overlaps = []
    for i in range(len(bcs)):
        bc = bcs[i]
        ind1 = bcs1.index(bc); ind2 = bcs2.index(bc)
        overlaps.append(intersect(sols1[ind1], sols2[ind2]))

    return overlaps, bcs

# "L2 norm" calculator
# Adjusted for the nature of PA
def L2(sol,a,b,m):
    ideal = [((m+1-i)*a+i*b)/(m+1) for i in range (1,m+1,1)]
    L2 = 0
    for i in range(len(sol)):
        L2 += (sol[i] - ideal[i])**2

    return math.pow(L2,.5)

def main():
    '''  # detecting whether solutions were similar to 0
    stuff = readSolutions('solutions_negative_bins.txt')
    template = [0]*20
    dim = [-1.0 + i*.05 for i in range(41)]
    for s in stuff:
        if detectSimilar(s, dim, template):
            print ("We found a similar solution")
            print (s)
            return
    '''


    ''' # do both types of rounding, return overlapping (shared) solutions
    sols1, bcs1 = readSolutions('Round_1_Testing.txt')
    sols2, bcs2 = readSolutions('Round_1_Testing_Down.txt')
    overlaps, bcs = returnOverlap(sols1, bcs1, sols2, bcs2)
    numfulfilled = 0
    for i in range(len(bcs1)):
        print ("Value for bc's (" + str(bcs[i][0]) + ", " + str(bcs[i][1]) + "): ")
               + str(overlaps[i]))
        if (len(overlaps[i])):
            numfulfilled += 1

    print ("The number of fulfilled boundary conditions was: " + str(numfulfilled))
    '''

    worstL = [-1,-1,-1] # the worst 3 L2 values
    worstsols = [[],[],[]] # the worst 3 solutions
    sols, bcs = readSolutions('heat_again.txt')
    for i in range(len(bcs)):
        for j in range(len(sols[i])):
            val = L2(sols[i][j],bcs[i][0],bcs[i][1],9)
            if val > worstL[2]:
                worstL.append(val); worstsols.append(sols[i][j])
                worstL.sort(reverse=True); worstsols.sort(reverse=True)
                worstL = worstL[0:3]; worstsols = worstsols[0:3]

    print ("The worst 3 L2 norms were: " + str(worstL))
    print ("The worst 3 solutions corresponding to those norms were: ")
    for sol in worstsols:
        print (sol)


main()
