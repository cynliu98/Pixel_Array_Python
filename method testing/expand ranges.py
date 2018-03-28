# Testing method for UROP 2017-2018
# A mostly functioning version of this was present in your post-boundary modified code

# num: number to round
# r: resolution
# c: "offset" of the bin values (e.g. [.1, .6, 1.1] has offset of .1 from 0)
def roundtores(num, r, c): #resolution n
    i = (num - c)/r # solve for i
    roundi = round(i+.00001) # round i to the nearest integer, round up
    return r*roundi+c

# arr: original bin bounds
# r: resolution (difference between adjacent bins)
# n: number of matrices in product
# returns: new dim, number of added values (even integer)
def expand(arr, r, n):
	# this code is almost identical to that in Fall's PASS
	adjust = (1/(n-2)) * (arr[1] - arr[0])
	rawRange = [arr[0] - adjust, arr[1] + adjust]
	rang = [roundtores(rawRange[0], r, arr[0]), roundtores(rawRange[1], r, arr[0])]

	valsAdded = round((arr[0] - rang[0])/r) * 2

	return rang, valsAdded

# testing
r = .06
n = 12
arr = [2,4]

# expected answer: [1.82, 4.18], 6
rang, valsAdded = expand(arr,r,n)
print (rang, valsAdded)
