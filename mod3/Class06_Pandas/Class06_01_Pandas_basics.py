# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
# %pip install pandas
# %pip3 install pandas
# %conda install pandas
import numpy as np
import pandas as pd
print("\nReady to continue.")

#%%
# Standard quick checks
def dfChk(dframe, valCnt = False): 
  cnt = 1
  print('\ndataframe Basic Check function -')
  
  try:
    print(f'\n{cnt}: info(): ')
    cnt+=1
    print(dframe.info())
  except: pass

  print(f'\n{cnt}: describe(): ')
  cnt+=1
  print(dframe.describe())

  print(f'\n{cnt}: dtypes: ')
  cnt+=1
  print(dframe.dtypes)

  try:
    print(f'\n{cnt}: columns: ')
    cnt+=1
    print(dframe.columns)
  except: pass

  print(f'\n{cnt}: head() -- ')
  cnt+=1
  print(dframe.head())

  print(f'\n{cnt}: shape: ')
  cnt+=1
  print(dframe.shape)

  if (valCnt):
    print('\nValue Counts for each feature -')
    for colname in dframe.columns :
      print(f'\n{cnt}: {colname} value_counts(): ')
      print(dframe[colname].value_counts())
      cnt +=1

# examples:
# dfChk(df)

#%% 
# Let's start with pandas series
# with the simplest case, building from a python list
fiblist = [0,1,1,2,3,5,8,13,21,34] # first 10 numbers in Fibonacci sequence (seeds 0 and 1)
print("\nReady to continue.")

#%%
# pandas series
s = pd.Series(fiblist)
print(s)
print(s.values)

#%%
# add series name to it
s = pd.Series(fiblist, name='Fibonacci sequence')
print(s)
print(s.values)
# Series name is to annotate the series.
# Must be one of BIFS, usually we use str.
print("\nReady to continue.")

#%%
# selection same as list
s0 = s[0]
print(f"s0 is of type= {type(s0)}, value= {s0}" )
# would be same as 
# print("s0 is of type= {}, value= {}".format( type(s0), s0 ) )
# OR 
# print("s0 is of type= %s, value= %s" % ( type(s0), s0 ) )
s3 = s[3]
print(f"s3 is of type= {type(s3)}, value= {s3}" )
s05 = s[0:5]
print(f"s05 is of type= {type(s05)}, value= \n{s05}?" )
print(s05)
print("\nReady to continue.")

#%% 
# Can we build from Numpy array?
nplist = np.array(fiblist)
# pandas series
try: 
  s = pd.Series(nplist)
  print(s)
except:
  print("Cannot create pandas series directly from numpy ndarray")
  # pass
print("\nReady to continue.")

#%%
# build pandas series from list generator?
listGen = ( 2*n+1 for n in range(10**3) )
print(listGen) # this is a generator object
print(type(listGen))

#%%
success = False
try: 
  s = pd.Series(listGen)
  print("Success! Pandas list created from list generator")
  success = True
  s0 = s[0]
  print(f"s0 is of type= {type(s0)}, value= {s0}" )
  s3 = s[3]
  print(f"s3 is of type= {type(s3)}, value= {s3}" )
  s05 = s[0:5]
  print(s05)
except:
  print("Cannot create pandas series directly from list generator")
  # pass

if success:
  print(s[10**2])
else:
  pass

print("\nReady to continue.")

#%%
# So it works? Let's try again
# WARNING # If you run this cell, be prepared to interrupt the kernel manually
# 
# listGen = ( 2*n+1 for n in range(10**100  ) )
# print(listGen) # this is a generator object
# print(type(listGen))
# print("Working so far!")

success = False
# try: 
#   s = pd.Series(listGen)
#   print("Success! Pandas list created from list generator")
#   success = True
#   s0 = s[0]
#   print(f"s0 is of type= {type(s0)}, value= {s0}" )
#   s3 = s[3]
#   print(f"s3 is of type= {type(s3)}, value= {s3}" )
#   s05 = s[0:5]
#   print(s05)
# except:
#   print("Cannot create pandas series directly from list generator")
#   # pass

# if success:
#   print(s[10**10])
# else:
#   pass

# # So it doesn't really work, unless later versions will create a pandas series generator 
# # instead of trying to create an ordinary pandas series

print("\nReady to continue.")

#%%
# Next concept is on index  
# similar to primary key in Relational Database (RDB) structures  
# index does not have to be unique however...
pdfib = pd.Series(fiblist)
print(pdfib,'\n')
# exactly the same thing with an extra name
pdfib = pd.Series(fiblist, name='FibVal')
print(pdfib,'\n')

print("\nReady to continue.")

#%%
fibindex2 = ['one','two','three','four','five','six','seven','eight','nine','ten']
pdfib2 = pd.Series(fiblist, name='FibVal', index=fibindex2)
print(pdfib2,'\n')

print("\nReady to continue.")

#%%
# Just for fun, 
fibindex3 = [9,8,7,6,5,4,'three','two','one',0]
pdfib3 = pd.Series(fiblist, name='FibVal', index=fibindex3)
print(pdfib3,'\n')

fibindex4 = [5,9,7,2,0,3,1,6,4,8]
pdfib4 = pd.Series(fiblist, name='FibVal', index=fibindex4)
print(pdfib4,'\n')

print("\nReady to continue.")

#%%
# Seems like no big deal, nothing to see here.  
# We can do some more advanced indexing (for dataframes) that is consistent with primary/secondary key used in RDB 
# For now, know that index again can only be BIFS.
# Like dictionary, we often use str or int.
print(pdfib2['nine'])
# The series in any case still has an intrinsic integer index as with a python list
# So the above is the same as this here
print(pdfib2[8])

print("\nReady to continue.")

#%%[markdown]
# Notice with integer index, we lost the ability to reference the series using the intrinsic index
# The index we assigned will take precedence. 
# What do you think the call below will produce? Guess before you run.
#

#%%
# We can also use simple range syntax for filtering
print(pdfib2['three':'eight'], '\n')
# same as 
print(pdfib2[2:7], '\n')
# EXCEPT ..... ???? (fill in the ellipses...)

print("\nReady to continue.")

#%%
# What about 
print(pdfib3,'\n')
print(pdfib3[5],'\n')

#%%
# Compare to?
print(pdfib2,'\n')
print(pdfib2[5],'\n')
# and
print(pdfib4,'\n')
print(pdfib4[5],'\n')

#%%
# And the range? 
print(pdfib3,'\n')
print(pdfib3[5:7],'\n')
# Compare to?
print(pdfib2,'\n')
print(pdfib2[5:7],'\n')
# and
print(pdfib4,'\n')
print(pdfib4[5:7],'\n')

#%%
# One last try
print(pdfib3,'\n')
import sys
try:
  print(pdfib3[6:'one'],'\n')
except TypeError as err :
  print(f"Type Error: {err}")
except:
  print(f"unexpected error: {sys.exc_info()[0]}" )

print("\nReady to continue.")

#%%
# The pandas series index is immutable (cannot be changed)
print(pdfib.index[7])
import sys
try:
  pdfib.index[5] = 'newSix'
  print("changed 5th-index successfully.")
except TypeError as err :
  print(f"Type Error: {err}")
except:
  print(f"unexpected error: {sys.exc_info()[0]}" )

print("\nReady to continue.")

#%%
# you can change the entire index list however
print(pdfib.index)
try:
  pdfib.index = [9,9,9,9,5,4,3,2,1,0]
  print("changed the entire index successfully.")
  print(pdfib)
except:
  pass

print("\nReady to continue.")

#%%
# Now with the index changed, what do we get from
print(pdfib[5:7])
print(pdfib[5:0])
print(pdfib[9])

print("\nReady to continue.")

#%%
# What if the (integer) index is not defined in our series? 
# Will it now treat it as the intrinsic index and produce pdfib[8] -> 21 ?
# Guess.
# 
import sys
try:
  print(pdfib[8])
except KeyError as err :
  print(f"Key Error: {err}")
except:
  print(f"unexpected error: {sys.exc_info()[0]}" )

print("\nReady to continue.")

#%%
# Not that this works to give you the 9-th (normally index = 8) element
print(pdfib[8:9]) 
type(pdfib[8:9]) 
# although it gives you a series instead of a single value.

print("\nReady to continue.")


#%% [markdown]
#
# # Indexing in Pandas series
# - Is the index int? Can you and should you avoid int/float/boolean?
# - Is the index unique? If not, watch out when used for filtering.
print(pdfib[5])
print(f'type = {type(pdfib[5])}\n')
print(pdfib[9])
print(f'type = {type(pdfib[9])}\n')

print("\nReady to continue.")

#%%
# # Pandas Dataframe
# Let us build Pandas DataFrames from a few different methods.  
# First, some basic building blocks:
fiblist = [0,1,1,2,3,5,8,13,21,34] # first 10 numbers in Fibonacci sequence (seeds 0 and 1)
fibindex = ['one','two','three','four','five','six','seven','eight','nine','ten']

sqlist = [81,64,49,36,25,16,9,4,1,0] # 10 squared numbers in reversed order 
# sqlist = ['a','a','a','a','a','a','b','b','b','c'] # it's acceptable for pandas dataframe to 
# have different data type between columns, unlike numpy arrays
sqindex = ['ten','nine','eight','seven','six','five','four','three','two','one']

print("\nReady to continue.")

#%% 
# ## Method 1, from lists directly
# If no index is specified, pandas will auto generate integral index

# , and you do not specify an index, 
# pandas will use the natural integer indexing 
# Try two different scenario
# Scenario 1 - With two different pandas series, with same index (unique)
pandasdf = pd.DataFrame({'fib':fiblist, 'sq':sqlist})
print()
print(pandasdf, '\n')

# You can add index at that time, or at a later time using pandasdf.index
pandasdf = pd.DataFrame({'fib':fiblist, 'sq':sqlist}, index=fibindex)
print(pandasdf)

print("\nReady to continue.")

#%% 
# ## Method 2, from pandas series
# If the dictionaries do not have indexes, and you do not specify an index, 
# pandas will use the natural integer indexing 
# Try two different scenario
# Scenario 1 - With two different pandas series, with same index (unique)
fiblist = [0,1,1,2,3,5,8,13,21,34] # first 10 numbers in Fibonacci sequence (seeds 0 and 1)
fibindex = ['one','two','three','four','five','six','seven','eight','nine','ten']
pdfib = pd.Series(fiblist, index=fibindex)
print(pdfib,'\n')
sqlist = [81,64,49,36,25,16,9,4,1,0] # 10 squared numbers in reversed order 
# sqlist = ['a','a','a','a','a','a','b','b','b','c'] # it's acceptable for pandas dataframe to 
# have different data type between columns, unlike numpy arrays
sqindex = ['ten','nine','eight','seven','six','five','four','three','two','one']
pdsq = pd.Series(sqlist, index=sqindex)
print(pdsq,'\n')

print("\nReady to continue.")

#%%
pddf = pd.DataFrame({'fib':pdfib, 'sq':pdsq})
# A couple of notes:
# If the indexes from the two series are ordered the same way, there will not be re-ordering. 
# right now, it will be re-ordered alphabetically
# Column names has to be unique. Duplicate ones will overwrite the existing ones.
# print()
print(pddf, '\n')
print(type(pddf), '\n')
print(pddf.shape, '\n')
print(pddf.columns, '\n')

print("\nReady to continue.")

#%% 
# ## Method 2, Scenario 2
# if the two different pandas series indexes are not unique 
# fiblist = [0,1,1,2,3,5,8,13,21,34] # first 10 numbers in Fibonacci sequence (seeds 0 and 1)
fibindex2 = ['one','one','one','four','five','six','seven','eight','nine','ten']
pdfib2 = pd.Series(fiblist, index=fibindex2)
print(pdfib2,'\n')
# sqlist = [81,64,49,36,25,16,9,4,1,0] # 10 squared numbers in reversed order 
sqindex2 = ['ten','nine','eight','seven','six','five','four','three','two','one']
pdsq2 = pd.Series(sqlist, index=sqindex2)
print(pdsq2,'\n')

print("\nReady to continue.")

#%%
import sys
try: 
  pddf2 = pd.DataFrame({'fib':pdfib2, 'sq':pdsq2})
  print("success")
except FileNotFoundError as err :  # except (RuntimeError, TypeError, NameError):
  print(f"File Not Found Error: {err}")
except ValueError as err :  # except (RuntimeError, TypeError, NameError):
  print(f"Value Error: {err}")
except:
  print(f"unexpected error: {sys.exc_info()[0]}" )

# Error creating DF with duplicate indexes
# In RDB terms, pandas will not perform *outer/inner join* when the key is not unique

print("\nReady to continue.")

#%% [markdown]
# ## Method 3 
# Let's try import a dataset (csv) and poke around
# For VS Code, I installed "Edit csv" (janisdd) plug-in (optional) to help view and edit the csv if needed.
# Also have "Excel Viewer" (GrapeCity) installed. You can find and try others.

#%%
import os
os.getcwd()  # make sure you know what folder you are on 
#%%
os.chdir('.') # do whatever you need to get to the right folder, 
nfl = pd.read_csv('nfl2008_fga.csv', index_col=0 ) 
#  
# OTHER read_csv optional arguments
# use header=None if the csv file has no header row
# use names = [ list ] to supply col headers
# use na_values =  to replace na values with NaN
# use parse_dates =  to format date columns
# 
# The code above will use the GameDate as index (which is not unique. NOT a good idea, as you'll see.)


print("\nReady to continue.")

#%%
dfChk(nfl, True)
# nfl.head()
# nfl.tail()
# nfl.info()

print("\nReady to continue.")


#%%