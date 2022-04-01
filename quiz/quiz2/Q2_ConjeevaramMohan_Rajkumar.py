#%%[markdown]
# Quiz 2
# Name:  Rajkumar Conjeevaram Mohan
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

#%%
import pandas as pd
import dm6103 as dm
df = dm.api_dsLand('Titanic','id')
print(df.info())
print(df.head())


#%%
# 1. what is the total fare paid by all the passengers on board? 
#

# I am probably misinterpreting what you mean by total here.
# If you meant to say the sum of fares paid by all the passengers then this is the answer
#print(f'The total fare paid by all the passengers was {df.loc[:, "fare"].sum()}')

# What I believe the question means is, how much in total did each of the passengers pay ?
# If this is the question, then the answer is
print(f"The total fare paid by all the passengers on board would be: {df['fare']}")

#%%
# 2. create a boolean array/dataframe for female passengers. Use broadcasting and filtering 
# to obtain a subset of females, and find average age of the female passengers on board 
# 
females_b = df.loc[:, 'sex'] == 'female'
females = df.loc[females_b]
f_avg_age = females['age'].mean()
print(f"The average of female passengers on board was {round(f_avg_age)}")
# df.loc[df.loc[:, 'sex']=='female', 'age'].mean() Since you had asked to create separate subsets I did it that way

#%%
# 3. create a boolean array/dataframe for survived passengers. Use broadcasting and filtering 
# to obtain a subset of survivers, and find the average age of the survived passengers on board? 
# 
survived_b = df.loc[:, 'survived'] == 1
survived = df.loc[survived_b]
s_avg_age = survived['age'].mean()
print(f"The average age of those who survived was {round(s_avg_age)}")

#%%
# 4. What is the average age of the female passengers who survived? 
#
female_survived_b = females['survived'] == 1 
females_survived = females.loc[female_survived_b]
sf_avg_age = females_survived['age'].mean()

print(f'The average age of the female passengers who survived would be {round(sf_avg_age)}')

# %%

