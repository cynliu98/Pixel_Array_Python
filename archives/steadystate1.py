#UROP
#3/31/17

import itertools

def makeU(a,b,n): #resolution n; bounds [a,b] for both variables (square symmetric)
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

    #for now, multiply 4 instances of U
    #generalize later? with parameter

    Usol = []
    for left in dim: # "left" boundary condition
        row = []
        for right in dim: # "right" boundary condition
            element = []
            for j,k,l,m in itertools.product(range(n),range(n),range(n),range(n)):
                if abs(dim[j] - roundtores((left+dim[k])/2, dim)) < .0001 and \
		   abs(dim[k] - roundtores((dim[j]+dim[l])/2,dim)) < .0001 and \
		   abs(dim[l] - roundtores((dim[k]+dim[m])/2,dim)) < .0001 and \
		   abs(dim[m] - roundtores((dim[l]+right)/2,dim)) < .0001:
                    element.append({(dim[j],dim[k],dim[l],dim[m])})
            row.append(element)
        Usol.append(row)

    return Usol

def main():
    Usol = findSol(0,5,6)
    for row in Usol:
        print(row)

main()  
