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
# Violin plot for income vs happiness, choose one sensible variable to color (hue) the plot. 
# Use happiness as numeric, make scatterplot with jittering in both x and y between happiness and number of children. Choose what variable you want for hue/color.
# If you have somewhat of a belief that happiness is caused/determined/affected by number of children, or the other 
# way around (having babies/children are caused/determined/affected by happiness), then put the dependent 
# variable in y, and briefly explain your choice.

df = dm.api_dsLand('Happy') 


 
#%%
#1) Let's look at some basics of the data frame:
print(df.columns)

#%%
#2) some better column names:
cols = ['Year','ID','Hrs','Marital','Num_children','Family_income_tot','Happiness','Ballot']

### naming columns is up to you; they should however be:
#1) meaningful (i.e. represent the column; not 1,2,3 etc)
#2) consistent (If you capitalize the first letter of one, you should do it for all, unless you have a specific reason)
#3) no spaces (some operating systems don't work well with spaces, only use spaces if you are making a visualization out of your data)

df.columns = cols

print(df.columns)

#%%
### From dfChk, we can see that Hrs, Num_children, Family_income_tot are all objects, when they could be numeric. let's look at them in detail:

print(df['Hrs'].unique()) #Note; this works because pandas automatically attaches an index


#%%
### Hrs here has the string 'Not applicable'; let's replace it with np.NaN. In fact, let's just do it for the whole dataset:

df = df.replace('Not applicable',np.NaN)

### We also see 'No answer' and "Don't know"; let's do the same.

df = df.replace('No answer',np.NaN)

df = df.replace("Don't know", np.NaN) # Note we had to use double quotes here

print(df['Hrs'].unique()) # much better. 
### Now we ensure this column is numeric
df.loc[:,'Hrs'] = pd.to_numeric(df.loc[:,'Hrs'])

print(df['Num_children'].unique()) # a little trickier, what do we do with eight or more? it is up to you; I'll just use 8

#%%
### One option, replace with chi squared distribution:

def cleanDfchild(row):
  thisage = row["Num_children"]
  try: thisage = int(thisage) # if it is string "36", now int
  except: pass
  
  try: 
    if not isinstance(thisage,int) : thisage = float(thisage)  # no change if already int, or if error when trying
  except: pass
  
  if ( isinstance(thisage,int) or isinstance(thisage,float) ) and not isinstance(thisage, bool): return ( thisage if thisage>=0 else np.nan )
  if isinstance(thisage, bool): return np.nan
  # else: # assume it's string from here onwards
  thisage = thisage.strip()
  if thisage == "No answer": return np.nan
  if thisage == "Eight or more": 
    # strategy
    # let us just randomly distribute it, say according to chi-square, 
    # deg of freedom = 2 (see distribution) 
    # values peak at zero, most range from 0-5 or 6. Let's cap it 15
    thisage = min(8 + 2*np.random.chisquare(2) , 15)
    return thisage # leave it as decimal
  return np.nan # catch all, just in case

df.loc[:,'Num_children'] = df.apply(cleanDfchild,axis=1) #This won't do anything unless you comment out line 132
### Note; another reason to use df.apply or series.map is because when the data is large it is MUCH faster

#print(df.loc[:,'Family_income_tot'].unique()) # We could fill wit distributions if we wanted! or pick an average value for each; or just leave as a categorical variable.

## Either way we have to replace th Refused with NaN

df = df.replace('Refused',np.NaN)

print(df['Happiness'].unique()) # let's make our ordinal scale [0,1,2]

#%%
### One fast way to do this is a dictionary, or a function

def ordinalHappy(row):
    happy = row['Happiness']
    dic = {'Very happy':2,'Pretty happy':1,'Not too happy':0,np.NaN:np.NaN}
    if isinstance(happy,float) or isinstance(happy,int):
        return dic[happy]
    else:
        return dic[happy.strip()]
    return 'Something went horribly wrong'

df['Happiness'] = df.apply(ordinalHappy,axis=1)
print(df.head())

#%%
### For ballot, we can see that entries are 'ballot a','ballot b', etc. let's trim it down to a,b,c
print(df.loc[:,'Ballot'].unique())
def ballotClean(row):
    ballot = row['Ballot']
    if isinstance(ballot,str):
        return ballot[-1].lower()
    else:
        return ballot
    return "Something went horribly wrong"

df['Ballot'] = df.apply(ballotClean,axis=1)

#print(df['Ballot'].unique()) #check 

#%%
### Now loet's make plots!! remember we need to remove null values in order to plot.
df[['Marital','Hrs']].dropna().boxplot(column='Hrs',by='Marital')
plt.title('Hours worked last week by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Hours')
plt.show()


#%%
### violin plot:
df_violin = df[['Family_income_tot','Happiness']].dropna()
print(df_violin['Family_income_tot'].unique())
subIncome1 = df_violin.loc[df_violin['Family_income_tot'] == '$25000 or more','Happiness']
subIncome2 = df_violin.loc[df_violin['Family_income_tot'] == '$20000 - 24999','Happiness']
subIncome3 = df_violin.loc[df_violin['Family_income_tot'] == '$10000 - 14999','Happiness']
subIncome4 = df_violin.loc[df_violin['Family_income_tot'] == '$8000 to 9999','Happiness']
subIncome5 = df_violin.loc[df_violin['Family_income_tot'] == '$15000 - 19999','Happiness']
subIncome6 = df_violin.loc[df_violin['Family_income_tot'] == '$7000 to 7999','Happiness']
subIncome7 = df_violin.loc[df_violin['Family_income_tot'] == '$1000 to 2999','Happiness']
subIncome8 = df_violin.loc[df_violin['Family_income_tot'] == '$6000 to 6999','Happiness']
subIncome9 = df_violin.loc[df_violin['Family_income_tot'] == '$3000 to 3999','Happiness']
subIncome10 = df_violin.loc[df_violin['Family_income_tot'] == 'Lt $1000','Happiness']
subIncome11 = df_violin.loc[df_violin['Family_income_tot'] == '$5000 to 5999','Happiness']
subIncome12 = df_violin.loc[df_violin['Family_income_tot'] == '$4000 to 4999','Happiness']


plt.violinplot([list(subIncome1),list(subIncome2),list(subIncome3),list(subIncome4),list(subIncome5),list(subIncome6),list(subIncome7),
list(subIncome8),list(subIncome9),list(subIncome10),list(subIncome11),list(subIncome12)])
plt.show()

### Make sure to check multiple jitters; too much and you won't see the distribution

fuzzyhappy = df[['Happiness','Num_children']].dropna()['Happiness'] + np.random.normal(0,np.std(df['Happiness'])/5., size=len(df[['Happiness','Num_children']].dropna()))

fuzzyChildren = df[['Happiness','Num_children']].dropna()['Num_children'] + np.random.normal(0,np.std(df['Num_children'])/5, size=len(df[['Happiness','Num_children']].dropna()))

plt.plot(fuzzyhappy,fuzzyChildren,'o', markersize=3, alpha = 0.1)

plt.show()

#%%
### Above we test if number of children is dependent on happiness

### Below we test if happiness is dependent on number of children

plt.plot(fuzzyChildren,fuzzyhappy,'o', markersize=3, alpha = 0.1)

plt.show()

### There really doesn't appear to be a relationship; more children does not seem to have less happiness in general, and more

#%%
