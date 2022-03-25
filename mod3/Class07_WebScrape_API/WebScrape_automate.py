# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%% [markdown]
# Other web info are not encoded in the url. 
# The previous example uses what is called "get-method" query 
# variables (with the question mark ? in the url address for key => value pairs) in the url to request info 
# from the web server. Many modern websites do not use such methods.
# Interactions with the server can nowadays be using ajax, jQuery, other backend DB connections, 
# asynchronous in nature and so forth.
# 
# For such situations, we can try to automate the browswer experience and navigate the site to pull the content 
# with the (chrome) driver

 
#%%[markdown] 
# First, install selenium
# %pip install selenium
# %pip3 install selenium
#  
# follow the url from the messages when you install selenium,  
#  
# or directly from <https://chromedriver.chromium.org/home>  
#  
# or <https://chromedriver.chromium.org/downloads>
#
# Optional: I always 
# 1. put it in, say, Download/Dev/Chromedriver folder 
# 2. rename it with the version number, like chromedriver_98.0.4758.102 
# 3. Either make a duplicate and rename back to chromedriver, or (on Macs) create a symbolic link (aka symlink) from the terminal (at that folder)
# > ln -s chromedriver_98.0.4758.102 chromedriver
# That way, my codes below always have the same file name, yet I can keep track of all the different versions if I ever need to go back.

# NOTE
# We are using the chrome driver class directly here as an example. 
# You can also consider using the chrome driver service instead in certain scenario. 
# https://chromedriver.chromium.org/getting-started 
# Check out more if interested.

#%%
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep
# import csv
import pandas as pd
from bs4 import BeautifulSoup

# Selenium
# To find multiple elements (these methods will return a list):
# find_element_by_id
# find_element_by_name
# find_element_by_xpath
# find_element_by_link_text
# find_element_by_partial_link_text
# find_element_by_tag_name
# find_element_by_class_name
# find_element_by_css_selector

# To find multiple elements (these methods will return a list):
# find_elements_by_name
# find_elements_by_xpath
# find_elements_by_link_text
# find_elements_by_partial_link_text
# find_elements_by_tag_name
# find_elements_by_class_name
# find_elements_by_css_selector


#%%
# Basic setup, test chrome launch
driver = webdriver.Chrome(r'C:\\Users\\Rajkumar\\Downloads\\dev\\chromedriver.exe')
# also need to install and locate the webdriver.Chrome location
# for my install, I need to set 
# driver = webdriver.Chrome(r'/Users/elo/Downloads/dev/chromedriver')  # mac OS
# driver = webdriver.Chrome(r'C:\\Users\\elo\\Downloads\\dev\\chromedriver.exe')  # windows

driver.get("https://www.weather.gov")
t = input("Any key to continue and close the automation browser")
driver.quit()
print("\nReady to continue.")

#%%
# On my Mac setup, the chromedriver is not a trusted app/known app. So I need to 
# open the System Preference > Security & Privacy > General, to "Allow Anyway" 
# when it first run.
#  
#%%
# We will need this function soon. 
# Woring backwards...
def getGovWeatherTemperature(soup):
  selectTemp = soup.select('div#current_conditions-summary p.myforecast-current-lrg') # return a list
  temp = (selectTemp[0].text) if (len(selectTemp)==1) else "error"
  # print(f"temperature found to be {temp}") 
  return temp

# print("\nReady to continue.")


# driver start
driver = webdriver.Chrome(r'C:\\Users\\Rajkumar\\Downloads\\dev\\chromedriver.exe')
driver.get("https://www.weather.gov")

# set up list for looping later
zips = [90210,20052,20051,20050]
zip = zips[0]

# inp = driver.find_element_by_id('inputstring') 
inp = driver.find_element_by_css_selector('input#inputstring') 
sleep(0.1) # make sure page loaded already
inp.clear() # search input box
print('cleared')
sleep(0.1) # make sure page loaded already
inp.send_keys(zip)
print('zip')
sleep(1.0) # make sure page loaded already
inp.send_keys(Keys.DOWN) 
print('keys down')
sleep(0.1) # make sure page loaded already
# driver.find_element_by_id('btnSearch').click() # go Button
driver.find_element_by_css_selector('input#btnSearch').click() # go Button
print('go clicked')
sleep(2.0) # make sure page loaded already
driver.refresh()

# from bs4 import BeautifulSoup
soup = BeautifulSoup(driver.page_source, 'html5lib')
print("soup ready")
# print(soup.prettify())
# t = 0 # Use below when implemented
t = getGovWeatherTemperature(soup)
print(f"temperature for zip: {zip} was found to be {t}") 

# terminates the automation browser
# IMPORTANT: if you code fails to reach here, you 
# should consider quiting the driver yourself from the 
# ipython interactive session.
driver.quit()

print("\nReady to continue.")


#%%
# from bs4 import BeautifulSoup

# Use a dataframe to record the data
testWeatherData = pd.DataFrame( columns=['zip','temperature','datetime','lat','long','elevation'])

# loop thru zipcodes
zips = [90210,20052,20051,20050]

# pull info 
# ?? 'div#current_conditions_detail table tbody tr:last-of-type td'
# ?? selectTemp = soup.select('div#current_conditions-summary p.myforecast-current-lrg') # return a list
# ?? thistemperature = getGovWeatherTemperature(soup)
# append data to dataframe
# testWeatherData = testWeatherData.append({ 'zip': zip, 'temperature': thistemperature }, ignore_index=True)

# NOW
# pull the other data and save to the dataframe: testWeatherData
# datetime
# lat
# long
# elevation





# %% [markdown]
#
# Other things to consider...
#
# # Generator functions and recursive functions
#
# Let's say we are using automation to construct a web crawler, from one site, crawling to 
# other web links. A typical function to perform such task will be nested/recursive. And it is 
# often such task will be written as generator functions (remember them? list comprehension and 
# generator functions?) That way, the list could go on forever, from one site crawl to another, 
# but python will not wait for the entire list to be stored in memory to continue.
#
# # Async nature of web traffic, Promises, Watchers
#
# Procedural programming and traditional OOP are all sequential in nature. The flow of logic 
# is well structured. The web programming has provided a lot of challenges to these fundamental 
# practices in recent years. Communications between servers and clients are often routed in 
# complicated and asynchronous fashion. There is no guaranteed the codes executed first will be 
# completed first. For example, when the web client trying to connect to the server DB for query, 
# the response might not be received for a while, depending on the routing and the other connections. 
# It is impractical and almost impossible to always wait for the response before moving on to 
# other tasks. 
#
# So the latest web programming, with ajax, angular, typescript, and many others new frameworks, 
# all have the asynchronous nature in mind. The function calls often have options to wait for 
# the *promise* to be delivered before executing other related codes. (Called a promise.) 
# In many situations, we just move on and carry on with the rest of the codes, by there are 
# *watchers* implemented throughout. When the data (called model) is changed, say from the 
# server response eventually, for example, the watchers will alerted all involved to change 
# their status, and update their values if needed. If these are not setup properly, 
# there could be a lot of inconsistent and unexpected behavior on the web applications. 
#
