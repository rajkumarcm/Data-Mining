#%% [markdown]
# # Web scraping
#
# Our goal here is to learn
# * Basic HTML structure
# * Basic CSS structure and selector rules
# * Basic webpage inspection tools from browsers
# * Use of BeautifulSoup or other web scraping libraries, plus parsers 
# 
# Started with https://www.datacamp.com/search?tab=courses (2021-10)
# This will list all courses. 
# I restricted it to python courses, then R courses.
# Save the two html files locally just in case the webpage structure 
# changes in the future. You can also change the codes below 
# to load up the current pages directly, instead of using a 
# saved html page. The urls (checked 2021-10) are
# https://www.datacamp.com/search?q=&tab=courses&facets%5Btechnology%5D%5B%5D=Python 
# https://www.datacamp.com/search?q=&tab=courses&facets%5Btechnology%5D%5B%5D=R
# 

#%%
 
import pandas as pd
from bs4 import BeautifulSoup
from time import sleep

langs=['r','p']  # I have one Python and one R html file for datacamp listings
htmls=[] # a list to store the two html objects from beautiful soup
dframes=[] # a list to store the two dataframes for the info
soups=[] # the soup objects 

import os
os.chdir('.')  # make sure it's the correct working folder
dirpath = os.getcwd() # print("current directory is : " + dirpath)

def pull_DC_articleDiv_info(d):
  t = d.h4.text.strip()
  o = d.p.text.strip()
  tm = d.div.span.find_all('span',recursive=False)[-1].text
  c = d.div.find_all('span',recursive=False)[1].find_all('span',recursive=False)[-1].text
  ins = d.div.find_all('span',recursive=False)[2].find_all('span',recursive=False)[-1].text
  return pd.Series( { 'title' : t, 'objective': o, 'time': tm, 'category': c, 'instructor': ins, 'code':'', 'status': '' } )


for i, l in enumerate(langs):
  with open( os.path.join( dirpath, 'DC_'+l.upper()+'_listing.html') , 'r') as file:
    htmls.append( file.read().replace('\n', '') )
  sleep(0.1)
  dframes.append( pd.DataFrame(columns=['title','code','status','category','instructor','time','objective']) )
  soups.append( BeautifulSoup(htmls[i], 'html5lib') ) # 'lxml', 'html.parser'

  articles = soups[i].select('div.dc-global-search-result__content')
  for d in articles: dframes[i] = dframes[i].append( pull_DC_articleDiv_info(d), ignore_index=True )
  dframes[i].to_csv(os.path.join( dirpath,'DC_'+l.upper()+'_listing.csv') ) 

#%%


