#UROP
#4/15/17

import itertools
import random
#at some point, change to numpy

#n possible solution values y for each variable
#NOTE: n IS NOT THE NUMBER OF VARIABLES
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

#multiply U1 by U2
#on ith iteration
#iterate r times
#def mult(U1,U2,i,n):
#psuedocode
#base case: r = 1
#return
#Confirmed writing this pain in Julia or R

#test how close solution is to being linear
#m variables per solution
def rsquared(sol,a,b,m):
    ideal = [((m-1-i)*a+i*b)/m for i in range (1,m-1,1)]
    avg = (sum(sol))/(m-2)
    linedis = 0
    variance = 0
    for i in range(m-2):
        linedis += (sol[i] - ideal[i])**2
        variance += (sol[i] - avg)**2

    if variance == 0: return 1
    return (1 - linedis/variance) #r^2

def findSol(a,b,n,m):
    #dim consists of all possible values of the solution y
    U, dim = makeU(a,b,n)

    Usol = []
    for left in dim: # "left" boundary condition
        row = []
        for right in dim: # "right" boundary condition
            element = []
            for l in itertools.product(*[range(n)]*(m-2)):
                #print (l)
                l = [dim.index(left)] + list(l) + [dim.index(right)]
                works = [(abs(dim[l[i]] - roundtores((dim[l[i-1]]+dim[l[i+1]])/2, dim)) < .0001) for i in range(1,m-1,1)]
                if all(works):
                         element.append({tuple([left] + [dim[l[i]] for i in range(1,m-1,1)] + [right])})
            row.append(element)
        Usol.append(row)
    return Usol

def main():
    #discretization of y solution
    Usol = findSol(0,5,10,8)
    sumr2 = 0
    for row in Usol:
        #print(row)
        for bcs in row: #boundary conditions bc
            j = random.randint(0,len(bcs)-1)
            fullsol = list(list(bcs[j])[0])
            print ("CAT PARTY ------- " + str(fullsol))
            sumr2 += rsquared(fullsol[1:-1],fullsol[0],fullsol[-1],8)

    sampledr2 = sumr2/((10-1)**2)
    print (sampledr2)

##    testing = True
##    while (testing):
##        rawsol = input("Please input a comma-separated (w/o spaces) solution, with boundary conditions ")
##        fullsol = rawsol.strip().split(',')
##        fullsol = [float(num) for num in fullsol]
##        print (rsquared(fullsol[1:-1],fullsol[0],fullsol[-1],6))
##        resp = input("Do you want to test another solution? 0 if no 1 if yes ")
##        if not(int(resp)):
##            testing = False

main()  
