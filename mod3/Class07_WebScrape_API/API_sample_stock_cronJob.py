# -*- coding: utf-8 -*-
# The line above is used when I need to run this file as a shell script, as a cron job
#%%
import yfinance as yf
import datetime
# import numpy as np
import pandas as pd
# import re # regular expressions


#%%
def getClose(sym): 
  stock = yf.Ticker(sym.upper())
  lastday = stock.history(period='1d')   
  lastclose = lastday['Close'][0]
  return lastclose.__round__(2)

# import pandas as pd
import os
# os.chdir('.')  # make sure it's the correct working folder
# filename = 'stockportfolio.csv'
filepath = 'stockportfolio.csv'

def getDayQuotes(filepath):
  # import datetime
  tday = datetime.datetime.today()
  if tday.weekday() > 4 : return None   # 6-Sunday,5-Saturday, do nothing
  tdaystr = tday.strftime('%Y-%m-%d')

  # open file, import df
  df_last = pd.read_csv(filepath, header=[0,1], index_col=0, date_parser=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d') )

  df_new = pd.DataFrame(columns=df_last.columns) # empty dataframe for new day data
  df_new = df_new.append(pd.Series(name=tdaystr)) # add a blank row


  for i, sym in enumerate(df_last.columns): # loop through all the stock smybols in the csv file
    x = getClose(sym[0])
    df_new.iloc[0,i] = x

  df_last = pd.concat([df_new,df_last])
  df_last.index = pd.to_datetime(df_last.index, format='%Y-%m-%d') # row index = 'Date'

  df_last = df_last.iloc[0:25,:]  # trim number of rows to max 25, save to file

  df_last.to_csv(filepath, encoding='utf_8_sig')
  # df_last.to_csv(filepath, sep='\t')

  return None

print("\nReady to continue.")

#%%

getDayQuotes(filepath)

print("\nReady to continue.")

# %%
