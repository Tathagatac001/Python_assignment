lst=[]
for i in range(2000,3201):
    '''test python'''
    if i%7==0 and i%5 !=0:
        lst.append(str(i))
print(','.join(lst)) 
