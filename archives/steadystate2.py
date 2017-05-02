#UROP
#3/31/17

import itertools

def makeU(a,b,n): #resolution m; bounds [a,b] for both variables (square symmetric)
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

def linscore(sol,l,r,n): #test how close solution is to being linear
    pass

def findSol(a,b,n):
    U, dim = makeU(a,b,n)

    Usol = []
    for left in dim: # "left" boundary condition
        row = []
        for right in dim: # "right" boundary condition
            element = []
            for l in itertools.product(*[range(n)]*(n-2)):
                l = [left] + list(l) + [right]
                works = [(abs(dim[l[i]] - roundtores((l[i-1]+l[i+1])/2, dim)) < .0001) for i in range(1,n-1,1)]
                if all(works):
                         print("heyo")
                         element.append({tuple([left] + [dim[l[i]] for i in range(1,n-1,1)] + [right])})
            row.append(element)
        Usol.append(row)
    return Usol

def main():
    Usol = findSol(0,5,8)
    for row in Usol:
        print(row)

main()  
