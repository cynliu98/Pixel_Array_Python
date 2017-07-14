# UROP Summer 2017
# July 13
# A file for analyzing many solutions
# And their discrepancies

import math

def readSolutions(fname):
    f = open(fname, 'r') # open the file of solutions
    sols = []
    for l in f:
        unparsedSols = l[(l.index(':')+4):-2] # remove all brackets, parentheses, etc.
        try:
            while True:
                i = unparsedSols.index('),') # while there are more solutions
                unparsedSol = unparsedSols[:i]
                unparsedSols = unparsedSols[(unparsedSols.index('(')+1):]

                solNums = unparsedSol.split(',')
                sol = []
                for n in solNums:
                    n.strip()
                    sol.append(float(n))

                sols.append(sol)

        except:
            continue

    return sols


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

def main():
    stuff = readSolutions('solutions_negative_bins.txt')
    template = [0]*20
    dim = [-1.0 + i*.05 for i in range(41)]
    for s in stuff:
        if detectSimilar(s, dim, template):
            print ("We found a similar solution")
            print (s)
            return

    print ("We found no similar solutions")

main()
