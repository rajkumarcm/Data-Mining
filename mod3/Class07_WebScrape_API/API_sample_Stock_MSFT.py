# 2022-02

#%%
# Use Yahoo finance API
# %pip install yfinance
import yfinance as yf

#%%
# https://github.com/ranaroussi/yfinance 

msft = yf.Ticker("MSFT")
msft.info  # get stock info
msft.actions    # show actions (dividends, splits)
msft.dividends    # show dividends
msft.splits    # show splits

#%%
msft.financials    # show financials
msft.quarterly_financials
msft.major_holders    # show major holders
msft.institutional_holders    # show institutional holders
msft.balance_sheet    # show balance sheet
msft.quarterly_balance_sheet

#%%
msft.cashflow    # show cashflow
msft.quarterly_cashflow
msft.earnings    # show earnings
msft.quarterly_earnings
msft.sustainability    # show sustainability

#%% 
msft.recommendations   # show analysts recommendations
msft.calendar   # show next event (earnings, etc)
msft.isin   # show ISIN code - *experimental*
# ISIN = International Securities Identification Number
msft.options   # show options expirations
msft.news   # show news
# get option chain for specific expiration
# opt = msft.option_chain('YYYY-MM-DD')
# data available via: opt.calls, opt.puts

#%%
hist = msft.history(period="max")    # get historical market data
lastday = msft.history(period='1d')   
lastclose = lastday['Close'][0]
print(f'last close = {lastclose}')

#%% 
