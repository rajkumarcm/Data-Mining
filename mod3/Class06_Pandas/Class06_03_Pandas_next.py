# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
# import numpy as np
# %pip install pandas
# %pip3 install pandas
# %conda install pandas
import pandas as pd
import dm6103 as dm

#%%
nfl = dm.dbCon_dsLand('nfl2008_fga','GameDate')
print("\nReady to continue.")

#%%
dm.dfChk(nfl, True)
# nfl.head()
# nfl.tail()
# nfl.info()

print("\nReady to continue.")

#%%
# You can select subsets using loc and iloc functions
# index locator iloc
colsubset = nfl.iloc[:,0:4]
print(colsubset.head())

print("\nReady to continue.")

#%%
# locator loc
colsubset = nfl.loc[:,'HomeTeam':'sec'] 
print( colsubset.head() )
# including the beginning AND THE END 

print("\nReady to continue.")

#%%
rowsubset = nfl.iloc[2:7 , : ]
rowsubset

# print("\nReady to continue.")

#%%
import sys
try: 
  rowsubset = nfl.loc[20081005:20081012 , : ]
  print("success")
except KeyError as err :  # except (RuntimeError, TypeError, NameError):
  print(f"Key Error: {err}")
except ValueError as err :  # except (RuntimeError, TypeError, NameError):
  print(f"Value Error: {err}")
except:
  print(f"unexpected error: {sys.exc_info()[0]}" )
# Error. Non-unique row index.

print("\nReady to continue.")

#%% [markdown]
#
# ## Unique Index (UI)
#
# Always try to have unique index. In this case, there is no one column that serves as 
# a unique index, you can either specify no index so that pandas will use generic rowID 
# as index, or use multi-index/hierarchical index.

# nfl = pd.read_csv('nfl2008_fga.csv')
# nfl = dm.dbCon_dsLand('nfl2008_fga','GameDate')
nfl = dm.dbCon_dsLand('nfl2008_fga')
print(nfl.shape)
print(nfl.head())

print("\nReady to continue.")

#%% (Optional)
# add index name
nfl.index.name = 'gameID'
nfl.head()

print("\nReady to continue.")

#%%
# or use GameDate AND HomeTeam to create an unique pair as index (Advance Indexing)
nfl2 = pd.read_csv('nfl2008_fga.csv', index_col=[0,2] ) 
nfl2.head()

#%%
# This multi-index is still not unique
nfl2.loc[(20081130,'CLE')]

# print("\nReady to continue.")

#%% [markdown]
#
# # Broadcasting
#
# This is a simple example on broadcasting, as with numpy

nfl['season']='08-09' # broadcast to all elements (in that column) with the new assignment

print("\nReady to continue.")

#%% [markdown]
# # Data Structure
# 
# If you have learned or used database before, look at this dataframe using those lenses. 
# Is that usually how the data is stored in a DB (RDB)?  
# There are some replications and redundancies. 
# Can you spot them?
#
# For example, the values for 'season' is always the same, should we use one table for each season instead? 
# 
# Knowing the minutes of the game clock, why need to record the quarter qtr?
#
# Already have the record of home and away teams, knowing which one is the kickteam, why do we need to record the defense def?
#
# There are more than 10 games on a typical day, should we group them in a table for, say, week_n, therefore 
# no need to repeat the same date for 10+ times? Is that easier for storage and for retrival for data to be 
# stored in such RDB fashion?
#
# We are not here to say which method or system is best to store data and retrival. There are just so many reasons, good reasons, 
# why things are done in certain ways. 
#
# We will find ourselves in many situations the data collect is not in a form that is convenient to our analysis. 
# We might need to perform sql joins, lookups, merge tables, pivot tables, etc, (data wrangling) to prepare the data for our analysis.

#%%
# If you did some data processing, and like to save the result as a csv file, you can
# import os
import os
os.getcwd()  # make sure you know what folder you are on 
#%%
os.chdir('.') # do whatever you need to get to the right folder, 
# or make sure you include the full path when saving the file
nfl.to_csv('nfl_clean.csv')  

print("\nReady to continue.")

#%% 
# 
# In class exercise
# 
# Task 1: import the dataset for "diet" using pandas 
# The name of the dataset is called 'Diet6wk' 
# And you can use 'Person' column as the index.
# Use the dm.dbCon_dsLand( ) function to load this dateframe. 
# 

print("\nReady to continue.")

#%%
# Task 2: Save this dataframe to you local drive, to use it later.
# 

# #%%