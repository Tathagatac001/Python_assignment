import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


url='https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_original.csv'
titanic = pd.read_csv(url)
labels = ['Female', 'Male']
titanic['sex'] = titanic['sex'].fillna('Female')
sizes = [titanic['sex'].value_counts()['Female'],titanic['sex'].value_counts()['Male']]
sizes

fig,ax1 = plt.subplots()
ax1.pie(sizes,labels=labels,startangle=90,autopct='%1.1f%%')
ax1.axis('equal')
plt.tight_layout()
plt.show()



fig,ax1 = plt.subplots()
colors = {'male':'red','female':'blue'}
grouped = titanic.groupby('sex')
for key, group in grouped:
	group.plot(ax=ax, kind='scatter', x='age', y='fare', label=key, color=colors[key])
plt.show()
