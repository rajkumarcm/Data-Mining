# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dm6103 as dm

# The dataset is obtained from 
# https://gssdataexplorer.norc.org 
# for you here. But if you are interested, you can try get it yourself. 
# create an account
# create a project
# select these eight variables: 
# ballot, id, year, hrs1 (hours worked last week), marital, 
# childs, income, happy, 
# (use the search function to find them if needed.)
# add the variables to cart 
# extract data 
# name your extract
# add all the 8 variables to the extract
# Choose output option, select only years 2000 - 2018 
# file format Excel Workbook (data + metadata)
# create extract
# It will take some time to process. 
# When it is ready, click on the download button. 
# you will get a .tar file
# if your system cannot unzip it, google it. (Windows can use 7zip utility. Mac should have it (tar function) built-in.)
# Open in excel (or other comparable software), then save it as csv
# So now you have Happy table to work with
#
# When we import using pandas, we need to do pre-processing like what we did in class
# So clean up the columns. You can use some of the functions we defined in class, like the total family income, and number of children. 
# Other ones like worked hour last week, etc, you'll need a new function. 
# Happy: change it to numeric codes (ordinal variable)
# Ballot: just call it a, b, or c 
# Marital status, it's up to you whether you want to rename the values. 
# 
#
# After the preprocessing, make these plots
# Box plot for hours worked last week, for the different marital status. (So x is marital status, and y is hours worked.) 
# Violin plot for income vs happiness, 
# (To use the hue/split option, we need a variable with 2 values/binomial, which 
# we do not have here. So no need to worry about using hue/split for this violinplot.)
# Use happiness as numeric, make scatterplot with jittering in both x and y between happiness and number of children. Choose what variable you want for hue/color.
# If you have somewhat of a belief that happiness is caused/determined/affected by number of children, or the other 
# way around (having babies/children are caused/determined/affected by happiness), then put the dependent 
# variable in y, and briefly explain your choice.

dfhappy = dm.api_dsLand('Happy') 


#%%
print(f'Happy Datatypes:\n{dfhappy.dtypes}')

#%%[markdown]
# Year appears to be clean
print(f'\nYear:\n{dfhappy.year.unique()}')


#%%[markdown]
# Analyzing hrs1 variable
print(f'\nhrs1 unique values:\n{dfhappy.hrs1.unique()}')
print(f'\nhrs1 value_counts:\n{dfhappy.hrs1.value_counts()}')


#%%[markdown]
# Replacing Not applicable, No answer, and Don't know in hrs1
dfhappy.hrs1 = dfhappy.hrs1.replace({'Not applicable': np.nan, 'No answer': np.nan, "Don't know": np.nan})

# %%
print(f'\nhrs1 unique values:\n{dfhappy.hrs1.unique()}')


# %%
print(f'\nMarital unique values:\n{dfhappy.marital.unique()}')

# %%[markdown]
# Replacing No answer in Marital
dfhappy.marital = dfhappy.marital.str.strip().replace({'No answer': np.nan})

#%%
dfhappy.childs = dfhappy.childs.str.strip().replace({'Eight or m': min(8 + np.random.chisquare(2) , 12), 'Dk na': np.nan})

# def cleanDfChilds(row):
#   thechildren = row["childs"]
#   try: thechildren = int(thechildren) # if it is string "6", now int
#   except: pass
  
#   try: 
#     if not isinstance(thechildren,int) : thechildren = float(thechildren)  # no change if already int, or if error when trying
#   except: pass
  
#   if ( isinstance(thechildren,int) or isinstance(thechildren,float) ) and not isinstance(thechildren, bool): return ( thechildren if thechildren>=0 else np.nan )
#   if isinstance(thechildren, bool): return np.nan
#   # else: # assume it's string from here onwards
#   thechildren = thechildren.strip()
#   if thechildren == "Dk na": return np.nan
#   if thechildren == "Eight or more": 
#     thechildren = min(8 + np.random.chisquare(2) , 12)
#     return thechildren # leave it as decimal
#   return np.nan # catch all, just in case
# # end function cleanDfChilds

# dfhappy.childs = dfhappy.apply(cleanDfChilds, axis=1)
# dfhappy.childs.dropna(inplace=True)
# %%[markdown]
# Analyzing income
print(f'\nIncome:\n{dfhappy.income.unique()}')

# %%
def cleanDfIncome(row, colname): # colname can be 'rincome', 'income' etc
  thisamt = row[colname].strip()
  if (thisamt == "Don't know"): return np.nan
  # It's ok if not part of the data given, still this can be used for other similar datasets
  if (thisamt == "Not applicable"): return np.nan
  if (thisamt == "Refused"): return np.nan 
  if (thisamt == "Lt $1000"): return np.random.uniform(0,999)
  if (thisamt == "$1000 to 2999"): return np.random.uniform(1000,2999)
  if (thisamt == "$3000 to 3999"): return np.random.uniform(3000,3999)
  if (thisamt == "$4000 to 4999"): return np.random.uniform(4000,4999)
  if (thisamt == "$5000 to 5999"): return np.random.uniform(5000,5999)
  if (thisamt == "$6000 to 6999"): return np.random.uniform(6000,6999)
  if (thisamt == "$7000 to 7999"): return np.random.uniform(7000,7999)
  if (thisamt == "$8000 to 9999"): return np.random.uniform(8000,9999)
  if (thisamt == "$10000 - 14999"): return np.random.uniform(10000,14999)
  if (thisamt == "$15000 - 19999"): return np.random.uniform(15000,19999)
  if (thisamt == "$20000 - 24999"): return np.random.uniform(20000,24999)
  if (thisamt == "$25000 or more"): return ( 25000 + 10000*np.random.chisquare(2) )
  return np.nan

dfhappy.income = dfhappy.apply(cleanDfIncome, colname='income', axis=1)

# %%[markdown]
print(f'\nHappy values:\n{dfhappy.happy.unique()}')
# Replacing Pretty happy, Very happy, and Not too happy to ordinal values while the rest to nan
dfhappy.happy = dfhappy.happy.replace({'Pretty happy': 3, 'Very happy': 2, 'Not too happy': 1, 
                                       "Don't know": np.nan, 'Not applicable': np.nan,
                                       'No answer': np.nan})

#%%[markdown]
# Analyzing ballet
print(f'\nBallot values:\n{dfhappy.ballet.unique()}')
# Converting them into characters a, b, c, and d
dfhappy.ballet = dfhappy.ballet.replace({'Ballot a': 'a', 'Ballot b': 'b', 'Ballot c': 'c', 'Ballot d': 'd'})
# Values of ballet after the conversion
print(dfhappy.ballet.unique())

# %%[markdown]
# Drop all na's
dfhappy.dropna(inplace=True)

#%%[markdown]
# Convert all variables to appropriate data types
dfhappy.hrs1 = dfhappy.hrs1.astype(int)
dfhappy.marital = dfhappy.marital.astype(str)
dfhappy.childs = dfhappy.childs.astype(int)
# income already is in float64
dfhappy.happy = dfhappy.happy.astype(int)

#%%
# Describe data

print(f'\nDescribe year:\n{dfhappy.year.describe()}')
print(f'\nYear Value Counts:\n{dfhappy.year.value_counts()}')

print(f'\nDescribe hrs:\n{dfhappy.hrs1.describe()}')

print(f'\nDescribe childs:\n{dfhappy.childs.describe()}')
print(f'\nChilds Value Counts:\n{dfhappy.childs.value_counts()}')

print(f'\nDescribe marital:\n{dfhappy.marital.describe()}')
print(f'\nMarital Value Counts:\n{dfhappy.marital.value_counts()}')

print(f'\nDescribe Income:\n{dfhappy.income.describe()}')

print(f'\nDescribe happy:\n{dfhappy.happy.describe()}')
print(f'\nhappy Value Counts:\n{dfhappy.happy.value_counts()}')

print(f'\nDescribe ballet:\n{dfhappy.ballet.describe()}')
print(f'\nballet Value Counts:\n{dfhappy.ballet.value_counts()}')

# %%
