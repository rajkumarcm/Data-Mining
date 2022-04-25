#%%[markdown]
# Quiz 2
# Name:  
# 
# You may use web search, notes, etc. 
# Do not use help from another human. If you use help from another student, 
# then I have no choice but to consider that student not a human, and will be 
# booted off my class immediately. You will also arrive at the same fate.
#
# From the titanic dataframe: 
# Complete the tasks below without importing any libraries except pandas (and dm6103).
# 
# 1. what is the total fare paid by all the passengers on board? 
# 
# 2. create a boolean array/dataframe for female passengers. Use broadcasting and filtering 
# to obtain a subset of females, and find average age of the female passengers on board 
# 
# 3. create a boolean array/dataframe for survived passengers. Use broadcasting and filtering 
# to obtain a subset of survivers, and find the average age of the survived passengers on board? 
# 
# 4. What is the average age of the female passengers who survived? 
# 
# survival : Survival,	0 = No, 1 = Yes
# pclass : Ticket class, 1 = 1st, 2 = 2nd, 3 = 3rd
# sex : Gender / Sex
# age : Age in years
# sibsp : # of siblings / spouses on the Titanic
# parch : # of parents / children on the Titanic
# ticket : Ticket number (for superstitious ones)
# fare : Passenger fare
# embarked : Port of Embarkment	C: Cherbourg, Q: Queenstown, S: Southampton

import pandas as pd
import dm6103 as dm
df = dm.api_dsLand('Titanic','id')
print(df.info())
print(df.head())


#%%
# 1. what is the total fare paid by all the passengers on board? 
#


#%%
# 2. create a boolean array/dataframe for female passengers. Use broadcasting and filtering 
# to obtain a subset of females, and find average age of the female passengers on board 
# 

#%%
# 3. create a boolean array/dataframe for survived passengers. Use broadcasting and filtering 
# to obtain a subset of survivers, and find the average age of the survived passengers on board? 
# 

#%%
# 4. What is the average age of the female passengers who survived? 
# 

# %%

