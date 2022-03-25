# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('classic')
import dm6103 as dm

print("\nReady to continue.")


#%% [markdown] 
# # GSS Data Explorer
# The dataset here is obtained from [GSS Data Explorer](https://gssdataexplorer.norc.org). 
# It is real, anonymous data that you can select different fields and filters. 
# Register for a free account to use it. 
# You can download the data in different formats. #


#%%
# import os
gss = dm.api_dsLand('GSS_demographics','id')

print("\nReady to continue.")

#%%
print('\n',gss.head(),'\n')

#%%
# Try these
rowind = 1
colind = 2
colname = 'degree'

print(gss.iloc[rowind,colind], '\n') 

print(gss.loc[rowind][colname], '\n') 

print(gss.iat[rowind,colind], '\n')  # iat is for a single value only

print(gss.at[rowind,colname]) 

print("\nReady to continue.")

#%%
#
# Also try these:
print(gss.loc[2,'year'])
print(gss['year'][2]) # this might work, but better stick with the one above
print(gss.loc[:,'year'].loc[2])

print("\nReady to continue.")

#%%
# what was the problem?
# Make sure index is unique
gss = dm.api_dsLand('GSS_demographics', ['id','year'] )

print('\n',gss.head(),'\n')
gss.describe()

print("\nReady to continue.")

#%%
# Is the index unique?
print(gss.index.is_unique)
# False

print("\nReady to continue.")

#%%
# What's next?
dups = gss.index.duplicated()
print(dups)

print("\nReady to continue.")

#%%
dupentries = gss[dups]
print(dupentries.shape)

print("\nReady to continue.")

#%%
# Not too many. Let's take a peek at those entries.
print(dupentries)

print("\nReady to continue.")

#%%
# figure out how you want to clean it. 
# In this case, either just use dups to clean, or condition id>=0 to clean
gssc = gss[dups == False]
print(gssc.shape)
print(gssc.index.is_unique)

print("\nReady to continue.")

#%%
# Mission accomplished!
# Now we can try some basic df manipulation with multi-index / hierarchical index
print(gssc.loc[1,:])  ## this works
print('\n')
print(gssc.loc[(1,2012),:]) ## this works too for multi-index 

print("\nReady to continue.")

#%%
try: print( gssc.loc[1:2,:] )
except pd.errors.UnsortedIndexError as err :
  print(f"UnsortedIndexError Error: {err}\n")
  print("does not work with range for multi index with numeric as index.\n")

try: 
  print( gssc.iloc[1:2,:] )
  print("works with iloc however\n")
except: 
  print("still does not work with range even for iloc\n")

#%%
try: print( gssc.loc[(1,2012):(2,2012)] )
except pd.errors.UnsortedIndexError as err :
  print(f"UnsortedIndexError Error: {err}\n")
  print("Does not work with tuple-index as range for multi index neither\n")

print("\nReady to continue.")

#%%
# slicer doesn't work neither
# print(gssc.loc[( slice(1,4)  ,2012),age])

print("\nReady to continue.")

#%%
# Try change index to str type
gssc = gssc.reset_index()
gssc['id'] = gssc['id'].astype(str)
gssc['year'] = gssc['year'].astype(str)
# print(gssc.info())
# gssc.describe()
gssc.id.dtype  # 'O' for object
gssc.set_index(['id','year'], inplace=True)
gssc.info()

print("\nReady to continue.")

#%%
# Try loc and iloc on multi-index (non-numeric) again
print(gssc.loc["1",:])
# print(gssc.loc[("1","2012"),:])

print("\nReady to continue.")

#%%
print(gssc.loc[("1","2012"), 'degree':"sex" ] )
print(gssc.loc[   ("5","2012"), 'degree':"sex" ] )
# these work

print("\nReady to continue.")

#%%
try: print( gssc.loc["1":"2",:] )
except pd.errors.UnsortedIndexError as err :
  print(f"UnsortedIndexError Error: {err}\n")
  print("Does not work with range for multi index with numeric as index\n")
# Does NOT work

print("\nReady to continue.")

#%%
try: print( gssc.loc[('1','2012'):('2','2012'), 'age' ] )
except pd.errors.UnsortedIndexError as err :
  print(f"UnsortedIndexError Error: {err}\n")
  print("nor using tuple-index as range for multi index with numeric as index\n")

print("\nReady to continue.")

#%%
# slicer doesn't work neither
# print(gssc.loc[( slice(1,4)  ,2012),age])

idx = pd.IndexSlice
import sys
try: 
  print( gssc.loc[ idx[1:4, 'id'], idx[1:3, 'year'] ] )
except pd.errors.UnsortedIndexError as err :  # except (RuntimeError, TypeError, NameError):
  print(f"UnsortedIndexError Error: {err}")
except:
  print(f"unexpected error: {sys.exc_info()[0]}" )

print("\nReady to continue.")

#%%
# print(gss.dtypes)
# gss.sex = pd.to_numeric(gss.sex, errors='coerce')
gss.info()
gss.dtypes

# assert statements 
# try: assert 1 == 1
# except: print("assertion not true")

print("\nReady to continue.")

#%%
# for now, let's go back to basics, use the dataframe with the intrinsic interger position as index. 
print('\n',gssc.head(),'\n')
gssc = gssc.reset_index()
print('\n',gssc.head(),'\n')
# At least we removed the id == -1 entries already

print("\nReady to continue.")

#%%
# Look at educ ("Highest year of school completed")
educ = gssc['educ']

print("\nReady to continue.")

#%%
plt.hist(educ.dropna(), label='educ')
# plt.savefig('nice_fig.png')
plt.show()

#%% 
# That was terrible
print(gssc[gssc['educ'] >40])

#%% 
# Let's drop these 12 perpetual learners from our dataframe
educ = educ[gssc['educ'] <40]    
print("\nReady to continue.")

#%% 
gssc = gssc[gssc['educ'] <40] 
plt.hist(educ.dropna(), label='educ',edgecolor='black', linewidth=1.2)
plt.xlabel('Years of Education')
plt.ylabel('Rel freq.')
plt.savefig('hist_educ.png')
plt.show()

#%%
# When the numerical variable is not truely continuous, but some discrete 
# or integer values with a finite range, histogram with arbitrary number of bins might 
# not be the best way to present it.
# We can use Probability Mass Function PMF instead. (Similar to Probability Density Function pdf)
# Without an built-in function, you either google 
# or use this hack (if you know the range of values 

bins = np.linspace(0.5, 20.5, 21)
plt.hist(educ, bins, alpha=0.5, edgecolor='black', linewidth=1)
plt.xticks(np.arange(0,21, step=2))
plt.xlabel('Years of Education')
plt.ylabel('PMF / Frequency')
plt.show() 
plt.savefig('pmf_educ.png')
plt.show() 


#%% 
# Next plot it with 2 series by gender
educ_s1 = educ[ gssc['sex']==1 ]
educ_s2 = educ[ gssc['sex']==2 ]
print("\nReady to continue.")

#%%
# Here is the plot

bins = np.linspace(0.5, 20.5, 21)
plt.style.use('seaborn-deep')
plt.hist(educ_s1, bins, alpha=0.5, label='s1',edgecolor='black', linewidth=1)
plt.hist(educ_s2, bins, alpha=0.5, label='s2',edgecolor='black', linewidth=1)

plt.xticks(np.arange(0,21, step=2))
plt.xlabel('Years of Education')
plt.ylabel('Frequency')
plt.legend(loc='upper right')

plt.show() 

#%%
# easier to see 

bins = np.linspace(0.5, 20.5, 21)
plt.style.use('seaborn-deep')
plt.hist([educ_s1,educ_s2], bins, alpha=0.5, label=['s1','s2'],edgecolor='black', linewidth=1)

plt.xticks(np.arange(0,21, step=2))
plt.xlabel('Years of Education')
plt.ylabel('Frequency')
plt.legend(loc='upper right')

plt.savefig('hist_educ_gender.png')
plt.show() 

#%% [markdown]
# ## IMAGINE 
# How much it will be better if we had used our python skill and rename sex 1/2 to M/F  
# Same for others such as industry, etc.
# On the other hand, it's much easier with the current libraries and setup to leave them 
# in numerical form to perform machine learning... 

#%%
# Let us try some different plots
# But first, get some better idea of the dataframe
gssc.describe()

#%% 
# Let us get a box plot of age for the subset degree == 1 ( or 2 and 3 and 4)
# model from https://matplotlib.org/3.1.1/gallery/statistics/boxplot_demo.html
# import matplotlib.pyplot as plt
# import numpy as np
from matplotlib.patches import Polygon

subDegree1 = gssc[ gssc['degree']==1 ]

# create a 2x3 subplot areas for contrasts
fig, axs = plt.subplots(2, 3) 

# basic plot
axs[0, 0].boxplot(subDegree1['age'])
axs[0, 0].set_title('basic plot')

# notched plot
axs[0, 1].boxplot(subDegree1['age'], 1)
axs[0, 1].set_title('notched plot')

# change outlier point symbols
axs[0, 2].boxplot(subDegree1['age'], 0, 'gD')
axs[0, 2].set_title('change outlier\npoint symbols')

# don't show outlier points
axs[1, 0].boxplot(subDegree1['age'], 0, '')
axs[1, 0].set_title("don't show\noutlier points")

# horizontal boxes
axs[1, 1].boxplot(subDegree1['age'], 0, 'rs', 0)
axs[1, 1].set_title('horizontal boxes')

# change whisker length
axs[1, 2].boxplot(subDegree1['age'], 0, 'rs', 0, 0.75)
axs[1, 2].set_title('change whisker length')

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9, hspace=0.4, wspace=0.3)

# Next plot multiple boxplots on one Axes
subDegree2 = gssc[ gssc['degree']==2 ]
subDegree3 = gssc[ gssc['degree']==3 ]
subDegree4 = gssc[ gssc['degree']==4 ]

# Multiple box plots on one Axes
fig, ax = plt.subplots()
data = [ subDegree1['age'], subDegree2['age'], subDegree3['age'], subDegree4['age'] ]
plt.boxplot(data)
plt.xlabel('Degree (code)')
plt.ylabel('Age')

# plt.savefig('boxplot_age_degree.png')
plt.show()

#%%
# Let us also try a violinplot, similar to boxplot
# fig, axes = plt.subplots()

plt.violinplot( [ list(subDegree1['age']), list(subDegree2['age']), list(subDegree3['age']), list(subDegree4['age'])] , positions = [1,2,3,4] )
plt.xticks(np.arange(0,5))
plt.xlabel('Degree (code)')
plt.ylabel('Age')

# plt.savefig('violin_age_degree.png')
plt.show()


#%%
# One more plot - scatterplot
# income (respondent) vs age

plt.plot(gssc.age, gssc.rincome, 'o')
plt.ylabel('Respondent income')
plt.xlabel('Age')
plt.show()

# Doesn't really work

#%%
# subset
weThePeople = gssc[ gssc['rincome'] < 80 ]
weThePeople.shape
weThePeople.describe()

#%%
plt.plot(weThePeople.age, weThePeople.rincome, 'o')
plt.ylabel('Respondent income monthly? ($1k)')
plt.xlabel('Age')
plt.show()

#%%
# Change alpha value
plt.plot(weThePeople.age, weThePeople.rincome, 'o', alpha = 0.1)
plt.ylabel('Respondent income monthly? ($1k)')
plt.xlabel('Age')
plt.show()

#%%
# Change marker size 
plt.plot(weThePeople.age, weThePeople.rincome, 'o', markersize=3, alpha = 0.1)
plt.ylabel('Respondent income monthly? ($1k)')
plt.xlabel('Age')
plt.show()

#%%
# Add jittering 
fuzzyincome = weThePeople.rincome + np.random.normal(0,1, size=len(weThePeople.rincome))
plt.plot(weThePeople.age, fuzzyincome, 'o', markersize=3, alpha = 0.1)
plt.ylabel('Respondent income monthly? ($1k)')
plt.xlabel('Age')
plt.show()

#%%
# Add jittering to x as well
fuzzyage = weThePeople.age + np.random.normal(0,1, size=len(weThePeople.age))
plt.plot(fuzzyage, fuzzyincome, 'o', markersize=3, alpha = 0.1)
plt.ylabel('Respondent income monthly? ($1k)')
plt.xlabel('Age')
plt.show()

plt.savefig('scatter_income_age.png')
plt.show()




#%%
