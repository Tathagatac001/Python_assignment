df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm',
'Budapest_PaRis', 'Brussels_londOn'],
'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )',
'12. Air France', '"Swiss Air"']})


df.FlightNumber=df['FlightNumber'].interpolate().astype(int)

tmp=pd.DataFrame()
tmp['From']=df['From_To'].str.split('_').str[0]
tmp['To']=df['From_To'].str.split('_').str[1]


tmp=tmp.applymap(lambda x: x.title())

df.drop(columns='From_To',inplace=True)
df=pd.contact([df,tmp],axis=1)
	
delays=pd.DataFrame(list(df.RecentDelays))	

def gen_col():
	cols=[]
	for i in delays.columns:
		cols.append('delay_{}'.format(i+1))
	return cols
	
		
delays.columns=gen_col()
df.drop(columns='RecentDelays',inplace=True)
df=pd.contact([df,delays],axis=1)
