#%%[markdown]
# You may use web search, notes, etc. 
# Do not use help from another human. If you use help from another student, 
# then I have no choice but to consider that student not a human, and will be 
# booted off my class immediately. You will also arrive at the same fate.
# 
#%%
from tkinter.font import families
import pandas as pd
import dm6103 as dm
df = dm.api_dsLand('Diet6wk','Person')
df.columns.values[3] = 'origweight'
df.info()

#%%
# The dataframe is on a person's weight 6 weeks after starting a diet. 
# Build these models:
# 
# 1. Using statsmodels library, build a linear model for the wight6weeks as a 
# function of the other variables. Use gender and Diet as categorical variables. 
# Print out the model summary. What is the r-squared value of the model?  
# 
from statsmodels.formula.api import ols
model = ols('weight6weeks~C(gender)+C(Diet)+Age+Height+origweight', data=df).fit()
model.summary()

#%%[markdown]
# The r-squared value of the model is 0.930

#%%
# 2. Again using the statsmodels library, build a multinomial-logit 
# regression model for the Diet (3 levels) as a function of the other variables. 
# Use gender as categorical again. Print out the model summary. 
# What is the  model's "psuedo r-squared" value?  
# 
# from statsmodels.formula.api import glm
# use this for multinomial logit in statsmodels library, 
# instead of glm for binomial.
from statsmodels.formula.api import mnlogit 
# Sample use/syntax:
# model = mnlogit(formula, data)
mn_model = mnlogit('Diet~C(gender)+Age+Height+origweight+weight6weeks', data=df).fit()
print(f'The pseudo r-squared value is {mn_model.prsquared}')


#%%
# 3a. Use SKLearn from here onwards. 
# Use a 2:1 split, set up the training and test sets for the dataset, 
# with Diet as y, and the rest as Xs. Use the seed value/random state as 1234
#  for the split.
#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'Diet'], df.Diet,
                                                    test_size=0.33, random_state=1234)

#%%
# 
# 3b. Build the corresponding logit regression as in Q2 here using sklearn. 
# Train and score it. What is the score of your model with the training set 
# and with the test set?
# 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_train_pred = log_reg.predict(X_train)
y_train_prob = log_reg.predict_proba(X_train)
y_test_pred = log_reg.predict(X_test)
y_test_proba = log_reg.predict(X_test)
print(f'Train data cm:\n{confusion_matrix(y_train, y_train_pred)}')
print(f'Performance on train data:\n{classification_report(y_train, y_train_pred)}')
print(f'Train score: {log_reg.score(X_train, y_train)}')
print(f'Test data cm:\n{confusion_matrix(y_test, y_test_pred)}')
print(f'Performance on test data:\n{classification_report(y_test, y_test_pred)}')
print(f'Test score: {log_reg.score(X_test, y_test)}')

#%%
# 4. Using the same training dataset, now use a 3-NN model, 
# score the model with the training and test datasets. 
# 
from sklearn.neighbors import KNeighborsClassifier as KNN
knn_clf = KNN(n_neighbors=3)
knn_clf.fit(X=X_train, y=y_train)
knn_train_pred = knn_clf.predict(X_train)
knn_test_pred = knn_clf.predict(X_test)
print(f'Train data cm:\n{confusion_matrix(y_train, knn_train_pred)}')
print(f'Performance on training data:\n{classification_report(y_train, knn_train_pred)}')
print(f'Train score: {knn_clf.score(X_train, y_train)}')
print(f'Test data cm:\n{confusion_matrix(y_test, knn_test_pred)}')
print(f'Performance on testing data:\n{classification_report(y_test, knn_test_pred)}')
print(f'Score on test data:\n{knn_clf.score(X_test, knn_test_pred)}')

#%%
