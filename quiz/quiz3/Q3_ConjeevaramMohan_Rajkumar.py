#%%[markdown]
# You may use web search, notes, etc. 
# Do not use help from another human. If you use help from another student, 
# then I have no choice but to consider that student not a human, and will be 
# booted off my class immediately. You will also arrive at the same fate.
# 
#%%
import pandas as pd
import dm6103 as dm
df = dm.api_dsLand('Diet6wk','Person')
df.columns.values[3] = 'origweight'
df.info()

# The dataframe is on a person's weight 6 weeks after starting a diet. 
# Build these models:
# 
# 1. Using statsmodels library, build a linear model for the wight6weeks as a function of the other variables. Use gender and Diet as categorical variables. Print out the model summary. What is the r-squared value of the model?  
# 



#%%
# 2. Again using the statsmodels library, build a multinomial-logit regression model for the Diet (3 levels) as a function of the other variables. Use gender as categorical again. Print out the model summary. What is the  model's "psuedo r-squared" value?  
# 
# from statsmodels.formula.api import glm
from statsmodels.formula.api import mnlogit  # use this for multinomial logit in statsmodels library, instead of glm for binomial.
# Sample use/syntax:
# model = mnlogit(formula, data)



#%%
# 3a. Use SKLearn from here onwards. 
# Use a 2:1 split, set up the training and test sets for the dataset, with Diet as y, and the rest as Xs. Use the seed value/random state as 1234 for the split.
#


#%%
# 
# 3b. Build the corresponding logit regression as in Q2 here using sklearn. Train and score it. What is the score of your model with the training set and with the test set?
# 




#%%
# 4. Using the same training dataset, now use a 3-NN model, score the model with the training and test datasets. 
# 



#%%
