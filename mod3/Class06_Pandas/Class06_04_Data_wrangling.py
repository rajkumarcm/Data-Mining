# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import dm6103 as dm
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# plt.style.use('classic')

#%%
# load the dataframe
#
dfgap = dm.dbCon_dsLand('gapminder','id')
# dfgap = dm.api_dsLand('gapminder','id')   #  Will learn this soon.
print("\nReady to continue.")

#%%
# After a successful retrival of the dataframe, you can consider saving it as csv locally. 
# That way you don't always need to re-connect to this DB (I don't know data changing often here.) 
# To do that, run the following codes   
import os
os.getcwd()  # make sure you know what folder you are on 
#%%
os.chdir('.') # do whatever you need to get to the right folder, 
# or make sure you include the full path when saving the file
dfgap.to_csv('gapminder.csv')
# then next time, you can simply use 
# dfgap = pd.read_csv('gapminder.csv', index_col='id' ) 


#%%
#%%
# Create one more column 
# dfgap['gdp'] = dfgap.gdpPercap * dfgap.pop     # doesn't work!
dfgap['gdp'] = dfgap['gdpPercap'] * dfgap['pop'] # works

#%%
# This describes pivot, pivot table, and stack/unstack pretty well.
# https://nikgrozev.com/2015/07/01/reshaping-in-pandas-pivot-pivot-table-stack-and-unstack-explained-with-pictures/
# 
# Melt is not described in the above, and it's the opposite of pivot.
# 
# This provide a slightly different presentation, with Melt included as well
# https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html 
# 
# In different contexts, these are studied under 
# indexing and primary/secondary keys, 
# or group_by functions and aggregates
# 
# Think of the datasets that you used before. 
# What were the circumstances you could use these re-shaping?
# 
print("\nReady to continue.")

#%%
# pivot and pivot_table on gapminder
dfgap_pvt = dfgap.pivot(index='country', columns='year' , values='gdpPercap')
dm.dfChk(dfgap_pvt)
#
# What is the shape of this df?
# What are the convenient things to work with this df?
# What are the cons with this structure?

print("\nReady to continue.")

#%%
# pivot gapminder, multi values
dfgap_pv2 = dfgap.pivot(index='country', columns='year' , values=['lifeExp','gdpPercap'])
dm.dfChk(dfgap_pv2)
#
# What is the shape of this df?
# What are the convenient things to work with this df?
# What are the cons with this structure?
print("\nReady to continue.")

#%%
# The columns now are multi-indexed !!
#
print(dfgap_pv2['lifeExp'])   # OR  print(dfgap_pv2.lifeExp) 

#%%
# print(dfgap_pv2['lifeExp'].mean()) # OR
print(dfgap_pv2.lifeExp.mean()) # default axis = 0 (column-wise).  This gives you the mean for each column or year
# print(dfgap_pv2['lifeExp'].mean(axis=1)) # OR
print(dfgap_pv2.lifeExp.mean(axis=1)) # axis = 1 (row-wise).  This gives you the mean for each row or country

print("\nReady to continue.")

#%% 
# There are different ways you can pull info from this structure
print( dfgap_pv2.lifeExp[1952] ) # this pull out the column of 1952
# print(dfgap_pv2.lifeExp['Spain'])  # This does not work on row selection
print(dfgap_pv2.lifeExp['Spain':'Sri Lanka']) # BUT this works! Pay attention, pandas slicing does include the end value!
print(dfgap_pv2.lifeExp['Spain':'Spain'])  # So this works too, and it is NOT EMPTY
print(dfgap_pv2.lifeExp[1952]['Spain']) # This gives you the data for Spain at 1952
# But to get better and more precise slicing, better use loc/iloc
print(dfgap_pv2.lifeExp.loc['Spain':'Sri Lanka', 1952:1972]) # Again, inclusive of end points for pandas loc
print(dfgap_pv2.lifeExp.iloc[2:4, 0:3]) # BUT NOT including the end point for iloc
#
#
# In general, it is much easier to perform group_by or filtering/subsetting with the column multi-indexing here, if well chosen to fit your needs.
print("\nReady to continue.")

#%%
# last try on multi values
dfgap_pvall = dfgap.pivot(index='year', columns='country')
dm.dfChk(dfgap_pvall)
#
# What is the shape of this df? (SUPER WIDE)

print("\nReady to continue.")

#%% [markdown]
# next try multiple columns, instead of multiple values
# dfgap_pc2 = dfgap.pivot(index='year', columns=['continent','country'] , values='gdpPercap')
# above doesn't work, but the next line works.  
# dfgap_pc2 = dfgap.pivot_table(index='year', columns=['continent','country'] , values='gdpPercap')
#
# Overall, the difference between pivot and pivot_table in pandas are these:  
#
# * pivot_table generalize pivot, can handle duplicate values for one pivoted index/column pair.
# * pivot_table allows aggreate functions with "aggfunc="
# * pivot_table allows multi-index (on rows)
# * pivot allows "values=" with numeric or string types. pivot_table only allow numeric (with str or categorical columns ignored)

#%% 
# so here it is, multiple indexes(indicies)/columns, instead of multiple values
dfgap_pc2 = dfgap.pivot_table(index='year', columns=['continent','country'], values='gdpPercap')
# dm.dfChk(dfgap_pc2)
#
# What is the shape of this df?
# What are the convenient things to work with this df?
# What are the cons with this structure?
print("\nReady to continue.")


#%%
# next try multiple indexes, 
dfgap_pi2 = dfgap.pivot_table(index=['continent','country'], columns='year' , values=['pop','gdpPercap'])
dm.dfChk(dfgap_pi2)
#
# Without running the code, can you tell what is the shape of this df?
print("\nReady to continue.")

#%%
# Stack: Opposite to pivot
dfgap_sptpi2 = dfgap_pi2.stack()
dm.dfChk(dfgap_sptpi2)
dfgap_pi2.head()
print("\nReady to continue.")


#%%
# in this case, it is actually more convenient and sensible to index by the country. 
# The previous examples simply exaggerate the pivot tables are usually "WIDE"
dfgap_ptCountryGdpp = dfgap.pivot_table(index='country', columns=['continent','year'] , values=['pop','gdpPercap'])
dm.dfChk(dfgap_ptCountryGdpp)

#%%
# Again, you can do filtering and selections like this:
# dfgap_ptCountryGdpp['gdpPercap']['Oceania'][1967] # same as 
dfgap_ptCountryGdpp.gdpPercap.Africa[1967]
# OR 
dfgap_ptCountryGdpp[ ('gdpPercap','Africa',1967) ]  # Using a tuple for selecting multi-Index (in the columns)


#%%
# Stack: Opposite to pivot
dfgap_sptCountryGdpp = dfgap_ptCountryGdpp.stack()
dm.dfChk(dfgap_sptCountryGdpp)
dfgap_ptCountryGdpp.head()
print("\nReady to continue.")

#%% [markdown]
# # Stack ~ Melt
# Melt is similar to stack, resulting in a long dataframe
# melt however does not preserve the custom index colummns!!
dfgap_mpi2 = dfgap_pi2.melt()
dm.dfChk(dfgap_mpi2)
print("\nReady to continue.")

#%% 
# need to do something like this instead
dfgap_pi2_defaultIndex = dfgap_pi2.reset_index() # reset the index from (continent,country) to default
dfgap_mpi2 = dfgap_pi2_defaultIndex.melt(id_vars=['continent','country'])
dm.dfChk(dfgap_mpi2)

print("\nReady to continue.")

#%%
# Finally, unstack is like pivot/pivot_table, which will be opposite to Stack/Melt



#%% [markdown]
# Roughly speaking:  
# long format (melt/stack) is good for data analysis of the raw data
# wide format (pivot/unstack) is good for summary, aggregates, and presentation



#%%
