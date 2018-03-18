import pandas as pd
 
df = pd.DataFrame({'X': [7, 2, 0, 3, 4, 2, 5, 0, 3, 4]})

def dist_cal(l):
	new_lst=[]
	tmp=[]
	strt_ind=0
	all_done=0
	for i in l:
		if all_done==1:
			break
		flg=0
		strt_ind=strt_ind-1
		tmp=l[strt_ind::-1]
		for ind,val in enumerate(tmp):
			if tmp[0]==0:
				new_lst.append(0)
				flg=1
				break
			elif val==0:
				new_lst.append(ind)
				flg=1
				break
		if flg==0:
			for ind,val in enumerate(tmp):
				new_lst.append(len(tmp)-ind)
				all_done=1
	return pd.Series(new_lst[::-1])
	
		
			
df['Y']=df[['X']].apply(dist_cal)


