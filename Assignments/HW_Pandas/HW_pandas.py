# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%% [markdown]
#
# # HW Pandas
# ## By: Rajkumar Conjeevaram Mohan
# ### Date: 13th March 2022
#

#%% [markdown]
# Let us improve our Stock exercise and grade conversion exercise with Pandas now.
#

#%%
import dm6103 as dm
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Load the data frame from api
dfaapl = dm.api_dsLand('AAPL_daily', 'date')
print("\nReady to continue.")


# ######  QUESTION 1      QUESTION 1      QUESTION 1   ##########

# What are the variables in the df? 
# What are the data types for these variables?

# ######  END of QUESTION 1    ###   END of QUESTION 1   ##########

#%%
# You can access pd dataframe columns using the dot notation as well as using column names
print(dfaapl.price, '\n')
# same as 
print(dfaapl['price'])


#%% 
# Step 1
# Create the Stock class 
# 

class Stock:
  """
  Stock class of a publicly traded stock on a major market
  """
  import dm6103 as dm
  import os
  import numpy as np
  import pandas as pd
  def __init__(self, symbol, name, init_tbname) :
    """
    :param symbol: stock symbol
    :param name: company name
    :param init_tbname: the initial table name on our DSLand API with historical data. Date is index, with eod price and vol as columns.
    """
    # note that the complete list of properties/attributes below has more than items than 
    # the numnber of arguments of the constructor. That's perfectly fine. 
    # Some property values are to be assigned later after instantiation.
    self.symbol = symbol.upper()
    self.name = name
    self.data = self.import_history(init_tbname) # this is a pandas df, make sure import_history() returns a pd dataframe
    # the pandas df self.data will have columns price, volume, delta1, delta2, and index is date
    self.init_delta1() # Calculate the daily change values from stock price itself, append to df
    self.init_delta2() # Calculate the daily values second derivative, append to df
    self.firstdate = self.data.index[-1] 
    self.lastdate = self.data.index[0] 
  
  def import_history(self, tbname):
    """
    import stock history from api_dsLand, with colunms date, eod_price, volume
    """
    return dm.api_dsLand( tbname, 'date' )  # use date as index
  
  def init_delta1(self):
    """
    compute the daily change from price_eod, append to data as new column as delta1
    """
    # notice that:
    # aapl['price'] returns a pandas series
    # aapl[['price']] returns a pandas dataframe
    # aapl['price'].values returns a numpy array of the values only

    self.data['delta1'] = 0  # initialize a new column with 0s
    self.data['delta1'] = self.data['price'][0:-1] - self.data.price.values[1:]   # self.data['price'] is same as self.price for df
    # the first term on the right is the full pd series with index attached. Second one is a simple numpy array without the date 
    # index. That way, the broadcasting will not try to match the indices/indexes on the two df
    return # you can choose to return self
  
  def init_delta2(self):
    """
    compute the daily change for the entire list of delta1, essentially the second derivatives for price_eod
    """
    # essentially the same function as init_delta1.

    # ######  QUESTION 2      QUESTION 2      QUESTION 2   ##########

    self.data['delta2'] = self.data['delta1'][0:-1] - self.data['delta1'].values[1:]
    
    # ######  END of QUESTION 2    ###   END of QUESTION 2   ##########

    return # you can choose to return self


  def add_newday(self, newdate, newprice, newvolume):
    """
    add a new data point at the beginning of data df
    """
    # Make plans 
    # insert a new row to self.data with 
    # (date, price, volume, delta1, delta2) to the pandas df, 
    # and also should update self.lastdate
    #

    # Simplistic though. 
    # In my opinion, it would only make sense to write dynamic code
    # if the values are either a list or an array, but in our case
    # the values are in three different variables, so ....
    # df2 = pd.DataFrame({'price': newprice, 'volume':newvolume}, index=[newdate])
    # self.data = pd.concat([self.data, df2])
    
    # update self.lastdate 
    self.lastdate = newdate
    # ######  QUESTION DONE      QUESTION DONE      QUESTION DONE   ##########

    # This part here is done, assuming the next function is set properly.
    
    # ######  END of QUESTION DONE    ###   END of QUESTION DONE   ##########

    # get ready a new row, in the form of a pandas dataframe.
    # Pandas dataframe does not have an insert function. The usual method is to use .append() 
    # and .append() is most efficient to append a df to another df of the same columns.
    newRow = self.setNewRow(newdate, newprice, newvolume) # we do this quite a lot: assume it's done already, then implement it later, as long as it doesn't break the codes
    # need this function setNewRow() to return a dataframe
    self.data = newRow.append(self.data) # this will put the new row on top, and push self.data after the new data

    return self

  
  def setNewRow(self, newdate, newprice, newvolume):
    # first create a copy of the dataframe with a dummy first row
    # the correct newdate is set as the index value for this 1-row dataframe
    df = pd.DataFrame( dict( {'date': [ newdate ]}, **{ key: [0] for key in self.data.columns } ) )
    df = df.astype({'price':np.float32, 'delta1':np.float32, 'delta2':np.float32})
    df.set_index( 'date', inplace=True ) 
    # df.index = [ newdate ] # this is already set properly above.
    df.price[0] = newprice
    # ######  QUESTION 3      QUESTION 3      QUESTION 3   ##########

    # write your codes here
    # set volume value
    df.loc[newdate, 'volume'] = newvolume
    # set delta1 value
    
    df.delta1[0] = newprice - self.data.iloc[0]['price']
    # set delta2 value
    df.delta2[0] = df.iloc[0]['delta1'] - self.data.iloc[0]['delta1']
    
    # ######  END of QUESTION 3    ###   END of QUESTION 3   ##########
    return df  # return the dataframe with one one row of data
  
  def nday_change_percent(self,n):
    """
    calculate the percentage change in the last n days, returning a percentage between 0 and 100
    """
    # ######  QUESTION 4      QUESTION 4      QUESTION 4   ##########

    change = (self.data.iloc[0]['price'] - self.data.iloc[n]['price'])/self.data.iloc[n]['price']
    percent = change * 100
    
    # ######  END of QUESTION 4    ###   END of QUESTION 4   ##########
    print(self.symbol,": Percent change in",n,"days is {0:.2f}".format(percent))
    return percent
  

  def nday_max_price(self,n):
    """
    find the highest price within the last n days 
    """
    # ######  QUESTION 5      QUESTION 5      QUESTION 5   ##########

    # return ??  # you can try to use the .max() function of a pandas dataframe
    return self.data.iloc[:n]['price'].max()
    
    # ######  END of QUESTION 5    ###   END of QUESTION 5   ##########

  def nday_min_price(self,n):
    """
    find the lowest price within the last n days 
    """
    # ######  QUESTION 6      QUESTION 6      QUESTION 6   ##########

    return self.data.iloc[:n]['price'].min()
    
    # ######  END of QUESTION 6    ###   END of QUESTION 6   ##########

#%%
# ######  QUESTION 7      QUESTION 7      QUESTION 7   ##########
# Try these:
filename = 'AAPL_daily'
aapl = Stock('AAPL','Apple Inc',filename)
aapl.data.head() # record the answer here
aapl.data.tail() # record the answer here

aapl.nday_max_price(333) # record the answer here
aapl.nday_min_price(500) # record the answer here
aapl.nday_change_percent(500)  # record the answer here

aapl.add_newday('9/13/19',218.42,12345678)  # record the answer here
aapl.data.head()

# ######  END of QUESTION 7    ###   END of QUESTION 7   ##########


#%%
# Next, re-solve the grade-changing exercise (calculating GPA)
# 

#%%
dats = dm.api_dsLand('Dats_grades')
dm.dfChk(dats)


#%%
# What are the variables in the df? 
print('\nThe variables in the grade data frame would be:')
for colname in dats.columns:
  print(colname, end=', ')
print('')

# What are the data types for these variables?

print('\nThe datatypes for these variables would be:')
for colname, dt in zip(dats.dtypes.index, dats.dtypes.values):
  print(f"{colname}: {dt}")

# The file has grades for a DATS class. Eight homeworks (out of 10 each), 2 quizzes (out of 100 each), and 2 projects (out of 100 each)
# Find out the class average for each item (HW, quiz, project)
# Hint, use .mean() function of pandas dataframe

# ######  QUESTION 8      QUESTION 8      QUESTION 8   ##########
dats_avg = {cname:float(0) for cname in dats.columns}
for i, cname in enumerate(dats.columns):
  norm = 10
  if i >=8:
    norm = 1
  dats_avg[cname] = dats[cname].mean() * norm

print('\nAverage grades:')
print(pd.Series(dats_avg))

# write your codes here

# ######  END of QUESTION 8    ###   END of QUESTION 8   ##########

#%%
# create a new column right after the last hw column, to obtain the average HW grade
# for each student.
# Name column as HWavg. Make the average out of the total of 100.
# Hint: use .iloc to select the HW columns, and then use .mean(axis=1) to find the row average

# ######  QUESTION 9      QUESTION 9      QUESTION 9   ##########
dats = pd.concat([dats.iloc[:, :8], pd.DataFrame({'HWMean':dats.loc[:, 'H1':'H8'].mean(axis=1)}), dats.iloc[:, 8:]], axis=1)
# write your codes here

# ######  END of QUESTION 9    ###   END of QUESTION 9   ##########

dats.head() # check result


#%%
# The course total = 30% HW, 10% Q1, 15% Q2, 20% Proj1, 25% Proj2. 
# Calculate the total and add to the df as the last column, named 'total', out of 100 max.

# ######  QUESTION 10      QUESTION 10      QUESTION 10   ##########
hw_series = dats.loc[:, 'HWMean'] * 0.3
quiz_series = dats.loc[:, 'Q1'] * 0.1 + dats.loc[:, 'Q2'] * 0.15
proj_series = dats.loc[:, 'Proj1'] * 0.2 + dats.loc[:, 'Proj2'] * 0.25
total_grades = hw_series + quiz_series + proj_series
dats = pd.concat([dats, pd.DataFrame({'total': total_grades})], axis=1)

# write your codes here

# ######  END of QUESTION 10    ###   END of QUESTION 10   ##########

dats.head() # check result

#%%
# Now with the two new columns, calculate the class average for everything again. 

# ######  QUESTION 11      QUESTION 11      QUESTION 11   ##########

# write your codes here

# ######  END of QUESTION 11    ###   END of QUESTION 11   ##########

#%%
# Save out your dataframe as a csv file
# import os

# ######  QUESTION 12      QUESTION 12      QUESTION 12   ##########

# write your codes here

# ######  END of QUESTION 12    ###   END of QUESTION 12   ##########



#%%
# Finally, re-solve our homework exercise for calculating GPA using functions, but with a dataframe now.
# In Week03 hw, we wrote a function to convert course total to letter grades. You can use your own, or the one from the solution file here.
def find_grade(total):
  # write an appropriate and helpful docstring
  """
  convert total score into grades
  :param total: 0-100 
  :return: str
  """
  # ######  QUESTION 13      QUESTION 13      QUESTION 13   ##########

  # copy your codes here, either from your Week03 hw, or the solution file

  # ######  END of QUESTION 13    ###   END of QUESTION 13   ##########
  return # grade  

#%%
# Let us create one more column for the letter grade, just call it grade.
# Instead of broadcasting some calculations on the dataframe directly, we need to apply (instead of broadcast) this find_grade() 
# function on all the elements in the total column
# ######  QUESTION 14      QUESTION 14      QUESTION 14   ##########

# write your code using the .apply() function to obtaine a new column of letter grade (call that new column 'grade') from the total.

# ######  END of QUESTION 14    ###   END of QUESTION 14   ##########


#%%
# Create a bar chart for the grade distribution 
# Save your chart out to a file as an image.
# Hint: use .value_counts() on the grade column to make a bar plot

# ######  QUESTION 15      QUESTION 15      QUESTION 15   ##########

# write your codes here

# ######  END of QUESTION 15    ###   END of QUESTION 15   ##########



#%%

