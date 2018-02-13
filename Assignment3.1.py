problem 1)

#defining myreduce func
def myreduce(f,lst1):
	result1=lst1[0]
	for i in range(1,len(lst1)):
		result1=f(result1,lst1[i])
	return result1


#callng function
>>> myreduce(lambda x,y:x+y,lst1)
41
>>> myreduce(lambda x,y:x*y,lst1)
96768
>>> myreduce(lambda x,y:x*y,[5,3,2])
30
>>> myreduce(lambda x,y:x+y,[5,3,2])
10

problem 2)
#defining my_filter func
def myfilter(f,lst1):
	return [ i for i in lst1 if f(i)]

#calling function

>>> myfilter(lambda x:x%2==0,lst1)
[4, 2, 4, 6, 8]
>>> myfilter(lambda x:x%2==1,lst1)
[1, 7, 9]

