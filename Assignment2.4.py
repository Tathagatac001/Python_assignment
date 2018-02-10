stmt=re.sub('\n',' ','''WE, THE PEOPLE OF INDIA, having solemnly resolved to constitute India into a
SOVEREIGN, SOCIALIST, SECULAR, DEMOCRATIC REPUBLIC and to secure to all
its citizens''')

print '{0},{1},\n\t{2},!\n\t\t{3},{4}'.format(*stmt.split(',')[0:-1]) +',{0},{1}'.format(*stmt.split(',')[-1].split()[0:2])+'\n\t\t {}'.format(" ".join(stmt.split(',')[-1].split()[2:]))
