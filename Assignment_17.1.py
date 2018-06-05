#import all necessary modules
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import tree

#load the dataset
Url="https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv"
titanic = pd.read_csv(Url)

#Data Exploration/Analysis
titanic.info()#we need to convert a lot of features into numeric ones later on, so that the machine learning algorithms can process them
titanic.shape#Total 891 rows and 12 columns present in the dataset
titanic.loc[:,titanic.isnull().any().tolist()]#Age Cabin and Embarked contains null data

titanic.describe()#from this we can see 38%passenger servived
titanic.corr()#from this correlation we can see there are no dependencies of servival on Name Ticket and PassengerId hence dropping
titanic=titanic.drop(['Name','Ticket','PassengerId'],axis=1)

#filling null column values 
titanic['Cabin'].mode()[0]
titanic['Cabin']=titanic['Cabin'].fillna("B96")
titanic['Embarked'].mode()[0]
titanic['Embarked']=titanic['Embarked'].fillna("S")
titanic['Age']=titanic['Age'].fillna(titanic.Age.mean())

titanic['Embarked']=titanic['Embarked'].map({"S":1,"C":2,"Q":3})
titanic['Sex']=titanic['Sex'].map({"male":1,"female":0})


#visualizing data

colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(titanic.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

#from heatmap we checked that Pclass,Sex,Fare,Embarked are strongly co-related with Servived column

#Now converting Age and Fare from float to int.
titanic['Fare'] = titanic['Fare'].astype(int)
titanic['Age'] = titanic['Age'].astype(int)

#Now mapping Age into different category
titanic.loc[ titanic['Age'] <= 16, 'Age']= 0
titanic.loc[(titanic['Age'] > 16) & (titanic['Age'] <= 32), 'Age'] = 1
titanic.loc[(titanic['Age'] > 32) & (titanic['Age'] <= 48), 'Age'] = 2
titanic.loc[(titanic['Age'] > 48) & (titanic['Age'] <= 64), 'Age'] = 3
titanic.loc[ titanic['Age'] > 64, 'Age']=4

#Mapping Fare into int

titanic.loc[ titanic['Fare'] <= 7.91, 'Fare'] = 0
titanic.loc[(titanic['Fare'] > 7.91) & (titanic['Fare'] <= 14.454), 'Fare'] = 1
titanic.loc[(titanic['Fare'] > 14.454) & (titanic['Fare'] <= 31), 'Fare']   = 2
titanic.loc[ titanic['Fare'] > 31, 'Fare'] 

#Since most of the data in Caib column is null we are removing this column.
titanic=titanic.drop('Cabin',axis=1)

# evaluate the model by splitting into train and test sets
X=titanic.drop('Survived',axis=1)
y=titanic['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
decision_tree = tree.DecisionTreeClassifier(max_depth = 3)
decision_tree.fit(X_train, y_train)


#check accuracy on train and test data
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_decision_tree
#--81.7%
acc_decision_tree_test = round(decision_tree.score(X_test, y_test) * 100, 2)

#80.97%

#confusion matrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(decision_tree, X_test, y_test, cv=10)
confusion_matrix(y_test, predictions)



