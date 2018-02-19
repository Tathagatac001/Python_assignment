import itertools

subjects=["Americans ","Indians"]
verbs=["play","watch"]
objects=["Baseball","Cricket"]

sentence=[subjects]+[verbs]+[objects]

result=list(itertools.product(*sentence))

for i in result:
	print " ".join(i)
	
	
