def filter_long_words(p_lst,n):
	return filter(lambda x:len(x)>n,p_lst)

filter_long_words(['Python','C++','Java','Unix','Oracle','Perl','HTML/CSS'],4)