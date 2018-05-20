"import all necessary modules"
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
import seaborn as sns

# Load Dataset and create DataFrame
boston = load_boston()
boston_df = pd.DataFrame(boston.data)
boston_df.columns = boston.feature_names

#Check if any null value present in any column
boston_df.columns[boston_df.isnull().any()].tolist()  #since this retun null.there is no such column

#add price column to boston_df dataframe from boston data.
boston_df['PRICE']=boston.target

'''--column set are :Index([u'CRIM', u'ZN', u'INDUS', u'CHAS', u'NOX', u'RM', u'AGE', u'DIS',u'RAD', u'TAX', u'PTRATIO', u'B', u'LSTAT',u'PRICE'],
dtype='object')'''

#Now create Corelation Matrix and plot heatmap 
sns.set(style="white")
# Compute the correlation matrix
corr = boston_df.dropna().corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(30, 10))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, square=True, linewidths=.5, ax=ax)

# RAD and TAX are highly co-related.
#Price negatively corelated with LSTAT(Strong),PTRATIO(Strong),TAX(high), INDUS(High), CRIM(Highly) and 
#NOX highly corelated with RM.
#Also Price positively corelated with RM(High), ZN(High), CHAS(Medium), DIS(MEDIUM) & B(Medium)

#draw scatter plot between RM and PRICE column since hetamp say they are strongly related to each other
sns.regplot(y="PRICE", x="RM", data=boston_df, fit_reg = True)
plt.show()

#draw scatter plot between ZN and PRICE column since heatmap say they are highly related to each other
sns.regplot(y="PRICE", x="ZN", data=boston_df, fit_reg = True)
plt.show()

#draw scatter plot between LSTAT and PRICE column (negatively corelated)
sns.regplot(y="PRICE", x="LSTAT", data=boston_df, fit_reg = True)
plt.show()


#draw scatter plot between PTRATIO and PRICE column (negatively corelated)
sns.regplot(y="PRICE", x="PTRATIO", data=boston_df, fit_reg = True)
plt.show()

#draw scatter plot between CRIM and PRICE column (negatively corelated)
sns.regplot(y="PRICE", x="CRIM", data=boston_df, fit_reg = True)
plt.show()

#draw scatter plot between TAX and PRICE column (negatively corelated)
sns.regplot(y="PRICE", x="TAX", data=boston_df, fit_reg = True)
plt.show()

#draw scatter plot between INDUS and PRICE column (negatively corelated)
sns.regplot(y="PRICE", x="INDUS", data=boston_df, fit_reg = True)
plt.show()

#from scatter plot we colud find that "RM" "PTRATIO" and "LSTAT" are strongly ralated to PRICE

#select only important varible for the model using stats model

# Split data into training and test datasets
X=boston_df.drop('PRICE',axis=1)
Y=boston_df['PRICE']

x_train, x_test, y_train, y_test = cross_validation.train_test_split(
X,Y, test_size=0.33, random_state=5)
lm = LinearRegression()
lm.fit(x_train, y_train)
pred_test = lm.predict(x_test)

print ('Estimated intercept coefficient:', lm.intercept_)
print ('Number of coefficients:', len(lm.coef_))


#Fit a linear regression model to the training set
#Predict the output on the test set

from sklearn import linear_model
# Create linear regression object
regr2 = linear_model.LinearRegression()
# Train the model using the training sets
regrfit= regr2.fit(x_train, y_train)
#Predicting the target variable from test.
y_pred = regr2.predict(x_test)
# The coefficients
print('Coefficients: \n', regr2.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((y_pred - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr2.score(x_test, y_test))


#Calculate the mean squared error

#using just the test data
#using just the training data

# The mean squared error
print("Mean squared error with test data: %.2f"
      % np.mean((y_pred - y_test) ** 2))

y_pred1 = regr2.predict(x_train)

print("Mean squared error with train data: %.2f"
      % np.mean((y_pred1 - y_train) ** 2))

#Residual plots

plt.scatter(regr2.predict(x_train), regr2.predict(x_train) - y_train, c='b', s=40, alpha=0.5)
plt.scatter(regr2.predict(x_test), regr2.predict(x_test) - y_test, c='g', s=40)
plt.hlines(y = 0, xmin=0, xmax = 50)
plt.title('Residual Plot using training (blue) and test (green) data')
plt.ylabel('Residuals')
plt.show()



#cross validation using k=4 to check performance of the model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
X=boston_df.drop('PRICE',axis=1)
X1=X.values
y=boston_df.PRICE.values
# We give cross_val_score a model, the entire data set and its "real" values, and the number of folds:
scores = cross_val_score(regrfit, X1,y, cv=10)
# Print the accuracy for each fold:
print ('Accuracies of 10 folds: ' , scores)

# And the mean accuracy of all 10 folds:
print ('Mean accuracy of 10 folds: ' , scores.mean())

#('Mean accuracy of 10 folds: ', 0.2098)

#Now using "RM" "PTRATIO" and "LSTAT" feature(since these features are highly related with price)

lm.fit(X[['PTRATIO','LSTAT','RM']], boston_df.PRICE)
msePTRATIO = np.mean((boston_df.PRICE - lm.predict(X[['PTRATIO','LSTAT','RM']])) ** 2)
print (msePTRATIO)

lm.score(X[['PTRATIO','LSTAT','RM']], boston_df.PRICE)
#.67

#using cross validation we get

X=X[['PTRATIO','LSTAT','RM']]
X1=X.values
y=boston_df.PRICE.values

x_train, x_test, y_train, y_test = cross_validation.train_test_split(
X[['PTRATIO','LSTAT','RM']],Y, test_size=0.33, random_state=5)
regr2 = linear_model.LinearRegression()
# Train the model using the training sets
regrfit= regr2.fit(x_train, y_train)
# We give cross_val_score a model, the entire data set and its "real" values, and the number of folds:
scores = cross_val_score(regrfit, X1,y, cv=10)
# Print the accuracy for each fold:
print ('Accuracies of 10 folds: ' , scores)

# And the mean accuracy of all 10 folds:
print ('Mean accuracy of 10 folds: ' , scores.mean())

#('Accuracies of 10 folds: ', array([ 0.6804222 ,  0.63048272,  0.40831934, -0.1577623 ]))
#('Mean accuracy of 10 folds: ', 0.39036549269558019)

#Mean accuracy is same in both cases


#Now using stat model we get

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

model1=sm.OLS(y_train,x_train)
result=model1.fit()
result.summary()

# We will consider the feature names with p_values < .05
#From the result we will select below columns

X=boston_df.drop(['PRICE','AGE','INDUS','CHAS','NOX'],axis=1)

#linear fit

lm.fit(X[['CRIM', 'ZN', 'RM', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']], boston_df.PRICE)
msePTRATIO = np.mean((boston_df.PRICE - lm.predict(X[['CRIM', 'ZN', 'RM', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']])) ** 2)
print (msePTRATIO)
#23.3233129352
lm.score(X[['CRIM', 'ZN', 'RM', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']], boston_df.PRICE)
#0.72372144563256779

#Using cross validation we get

x_train, x_test, y_train, y_test = cross_validation.train_test_split(
X[['CRIM', 'ZN', 'RM', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']],Y, test_size=0.33, random_state=5)
regr2 = linear_model.LinearRegression()
# Train the model using the training sets
regrfit= regr2.fit(x_train, y_train)
scores = cross_val_score(regrfit, X.values,y, cv=10)
# Print the accuracy for each fold:
print ('Accuracies of 10 folds: ' , scores)

# And the mean accuracy of all 10 folds:
print ('Mean accuracy of 10 folds: ' , scores.mean())

#('Accuracies of 10 folds: ', array([ 0.75482895,  0.56748153,  0.7237375 ,  0.64915842,  0.77281827,
#       -0.27155203, -0.53018999,  0.14458537]))
#('Mean accuracy of 10 folds: ', 0.27135850375945921)

#Since the Mean accuracy is .35 it is the best model
