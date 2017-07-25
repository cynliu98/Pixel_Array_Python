# UROP Summer
# 7/20/17
# R^2 analysis of solutions

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

# test how close solution is to being linear
# m variables per solution
def rsquared(sol,a,b,m):
    idealraw = [((m+1-i)*a+i*b)/(m+1) for i in range (m+2)]
    assert len(idealraw) == (m+2)
    ideal = idealraw[1:-1]
    # print ("(a,b): (" + str(a) + ", " + str(b) + ")")
    # print (ideal)
    assert len(ideal) == m
    avg = (sum(sol))/(m)
    linedis = 0
    variance = 0
    for i in range(m):
        linedis += (sol[i] - ideal[i])**2
        variance += (sol[i] - avg)**2

    if variance < 0.0001: return 1
    return (1 - linedis/variance) #r^2

def main():
    sols, bcs = readSolutions('Heat_equation_results.txt')
    # print (sols)

    # return

    maxr = - math.inf; minr = math.inf # maximum and minimum similarity
    maxsol = []; minsol = [] # the best matching and worst matching solutions
    avg = 0 # average r^2
    avgpos = 0; n = 0 # average positive r^2 and number of positive r^2's
    numMats = len(sols[0][0])
    numSols = 0
    prev = []
    assert len(sols) == len(bcs)

    for i in range(len(bcs)):
        for j in range(len(sols[i])):
            print ("The solution: " + str(sols[i][j]))
            print ("The boundaries: " + str(bcs[i]))
            rsq = rsquared(sols[i][j],bcs[i][0],bcs[i][1],numMats)
            avg += rsq
            numSols += 1
            if rsq > 0:
                avgpos += rsq
                n += 1
            if rsq > maxr:
                maxr = rsq
                maxsol = sols[i][j]
            if rsq < minr:
                minr = rsq
                minsol = sols[i][j]

    avg /= float(numSols)
    avgpos /= float(n)

    print ("The maximum r^2 was: " + str(maxr))
    print ("Corresponding to solution: " + str(maxsol))
    print ("The minimum r^2 was: " + str(minr))
    print ("Corresponding to solution: " + str(minsol))
    print ("The average r^2 was: " + str(avg))
    print ("The average positive r^2 was: " + str(avgpos) + " and there were " + str(n) + " such solutions.")

main()


