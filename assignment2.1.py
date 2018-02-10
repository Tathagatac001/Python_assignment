my_input=raw_input('Please enter numbers with comma separated :\n')
lst1=[]
lst=my_input.split(',')
for i in lst:
	lst1.append(int(i))
print lst1
