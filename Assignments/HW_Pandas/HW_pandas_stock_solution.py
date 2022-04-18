# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%% [markdown]
#
# # HW06 
# ## By: xxx
# ### Date: xxxxxxx
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

dm.dfChk(dfaapl)

# What are the variables in the df? 
# What are the data types for these variables?

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
    self.data['delta2'] = 0  # initialize a new column with 0s
    self.data['delta2'] = self.data.delta1[0:-1] - self.data.delta1.values[1:]   # self.data['price'] is same as self.price for df
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
    # update self.lastdate 
    self.lastdate = newdate

    # get ready a new row, in the form of a pandas dataframe.
    # Pandas dataframe does not have an insert function. The usual method is to use .append() 
    # and .append() is most efficient to append a df to another df of the same columns.
    newRow = self.setNewRow(newdate, newprice, newvolume) # we do this quite a lot: assume it's done already, then implement it later.
    # need this function setNewRow() to return a dataframe

    self.data = newRow.append(self.data) # this will put the new row on top, and push self.data after the new data

    return self
  
  def setNewRow(self, newdate, newprice, newvolume):
    # first create a copy of the dataframe with a dummy first row
    # the correct newdate is set as the index value for this 1-row dataframe
    df = pd.DataFrame( dict( {'date': [ newdate ]}, **{ key: [0] for key in self.data.columns } ) )
    df.set_index( 'date', inplace=True ) 
    
    # df.index = [ newdate ] # this is already set properly above.
    df.price[0] = newprice
    df.volume[0] = newvolume
    df.delta1[0] = newprice - self.data.price[0]
    df.delta2[0] = df.delta1[0] - self.data.delta1[0]
    return df
  
  def nday_change_percent(self,n):
    """
    calculate the percentage change in the last n days, returning a percentage between 0 and 100
    """
    change = self.data.price[0]-self.data.price[n]
    percent = 100*change/self.data.price[n]
    print(self.symbol,": Percent change in",n,"days is {0:.2f}".format(percent))
    return percent
  
  def nday_max_price(self,n):
    """
    find the highest price within the last n days 
    """
    return self.data.price[0:n].max()

  def nday_min_price(self,n):
    """
    find the lowest price within the last n days 
    """
    return self.data.price[0:n].min()

#%%
# Try these:
filename = 'AAPL_daily'
aapl = Stock('AAPL','Apple Inc',filename)
aapl.data.head()
aapl.data.tail()
aapl.nday_max_price(333) # record the answer here
aapl.nday_min_price(500) # record the answer here
aapl.nday_change_percent(500)  # record the answer here

aapl.add_newday('9/13/19',218.42,12345678)
aapl.data.head()

# %%
