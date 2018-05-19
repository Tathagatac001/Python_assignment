#import necessary models
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

# load dataset
dta = sm.datasets.fair.load_pandas().data
# add "affair" column: 1 represents having affairs, 0 represents not
dta['affair'] = (dta.affairs > 0).astype(int)


#Data Exploration
dta.groupby('affair').mean()

#from table we can see the women who have low marriage rate have an affair

dta.groupby('rate_marriage').mean()
#from the table we saw age, yrs_married, and children appears to correlate with a declining marriage rating.

# histogram of education
%matplotlib inline
dta.age.hist()
plt.title('Histogram of Over18')
plt.xlabel('Over18')
plt.ylabel('Frequency')

# histogram of marriage rating
dta.rate_marriage.hist()
plt.title('Histogram of Marriage Rating')
plt.xlabel('Marriage Rating')
plt.ylabel('Frequency')

# barplot of marriage rating grouped by affair (True or False)
pd.crosstab(dta.rate_marriage, dta.affair.astype(bool)).plot(kind='bar',stacked=True)
plt.title('Marriage Rating Distribution by Affair Status')
plt.xlabel('Marriage Rating')
plt.ylabel('Frequency')

#stacked plot on Years married and percentage

affair_yrs_married = pd.crosstab(dta.yrs_married, dta.affair.astype(bool))
affair_yrs_married.div(affair_yrs_married.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Affair Percentage by Years Married')
plt.xlabel('Years Married')
plt.ylabel('Percentage')


# create dataframes with an intercept column and dummy variables for
# occupation and occupation_husb
y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + \
                  religious + educ + C(occupation) + C(occupation_husb)',
                  dta, return_type="dataframe")
# print X.columns

# rename column names for the dummy variables for better looks:
X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
                        'C(occupation)[T.3.0]':'occ_3',
                        'C(occupation)[T.4.0]':'occ_4',
                        'C(occupation)[T.5.0]':'occ_5',
                        'C(occupation)[T.6.0]':'occ_6',
                        'C(occupation_husb)[T.2.0]':'occ_husb_2',
                        'C(occupation_husb)[T.3.0]':'occ_husb_3',
                        'C(occupation_husb)[T.4.0]':'occ_husb_4',
                        'C(occupation_husb)[T.5.0]':'occ_husb_5',
                        'C(occupation_husb)[T.6.0]':'occ_husb_6'})

# and flatten y into a 1-D array so that scikit-learn will properly understand it as the response variable.
y = np.ravel(y)


# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, y)

# check the accuracy on the training set
model.score(X, y)
#0.72588752748978946

# what percentage had affairs?
y.mean()
#.32

# examine the coefficients
pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))
#from the coeff table is is clear the increase of marriage rate and religiousness decrease the change of having affair.
#Also students (for both woman and and husband) have less chance of having affair


# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)
# predict class labels for the test set
predicted = model2.predict(X_test)
print predicted
# generate class probabilities
probs = model2.predict_proba(X_test)
print probs

# generate evaluation metrics
print metrics.accuracy_score(y_test,predicted)
print metrics.roc_auc_score(y_test,probs[:, 1])
#0.729842931937
#0.745950606951
#The accuracy is 73%,

# we can also see the confusion matrix and a classification report with other metrics.
print metrics.confusion_matrix(y_test, predicted)
print metrics.classification_report(y_test, predicted)


# evaluate the model using 10-fold cross-validation
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print scores
print scores.mean()

#72.4% accuracy.

