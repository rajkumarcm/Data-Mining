#%% 
# -*- coding: utf-8 -*-
# The line above is used when I need to run this file as a shell script, as a cron job
#%%
import datetime
# import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
# from time import sleep
import re # regular expressions

# # Web scraping
#

#%%
def getUrl(sym):
  url = 'https://money.cnn.com/quote/quote.html?symb=' + sym.upper()
  return url 

def getSoup(url,timer=0,parser=''):
  # import requests
  # from bs4 import BeautifulSoup
  p = parser if ( parser=='lxml' or parser=='html.parser') else 'html5lib'
  if (timer > 0):
      print('slept',timer,'s')
      r = requests.get(url, timeout = timer )
  else:
      r = requests.get(url)
  s = BeautifulSoup(r.content, p) # or 'lxml' or 'html.parser'
  return s

def getRow(soup): # Yahoo finance v201911  # works on ^DJI
  selectlast = soup.select('tr > td.wsod_last > span') # return a list
  return selectlast[0] if len(selectlast) else None

def getClose(sym,timer=0): # Sometimes the response come back slow. Will re-try a few times 
  while timer < 4:
    s = getSoup(getUrl(sym),timer/4)
    if s is None:
      print('type s NoneType. timer =',timer/4)
      timer += 1
      continue
    d = getRow(s)
    if d is None:
      print('type d NoneType. timer =',timer/4)
      timer += 1
      continue
    return d.text.replace(',','')  # remove comma for (floats more than triple digits)
  return 'tried many times'

# import pandas as pd
import os
# os.chdir('.')  # make sure it's the correct working folder
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

  df_last = df_last.iloc[0:25,:]
  # trim number of rows to max 25, save to file

  df_last.to_csv(filepath, encoding='utf_8_sig')
  # df_last.to_csv(filepath, sep='\t')

  return None

# print("\nReady to continue.")


getDayQuotes(filepath)

print("\nReady to continue.")

# %%
