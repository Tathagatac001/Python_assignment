def longestWord(p_lst):
	return sorted(p_lst,key=len).pop()
	#print max(mylist, key=len)


lst=['PYTHON','XYZ','DEFG','C']
print longestWord(lst)
