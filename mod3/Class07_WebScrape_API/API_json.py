# %%[markdown]
#
# # Working with API and JSON in Python
#
# Remember the `requests` library we used to perform a http call for web-scrapping?
# We are basically performing a very similar task. And in some cases, they can be
# the same thing, just under a different hood.
#
# We will make the api request call using `requests.get()` function.
# Instead of the web server serving us (the client browser) the html codes/pages,
# this time api call **usually** will have the server returning some data, and **often** in
# the form of JavaScript Object Notation (JSON) object.
#
# To python, a JSON object is nothing more than just some
# dictionary of dictionary of list of ...
# When something is referenced by some keywords, it will be stored as key => value in
# a dictionary. If there are similar things stored together, it will be as a list.
#
# Example: myHonda = {'color': 'red', 'wheels': 4, 'engine': {'cylinders': 4, 'size': 2.2} }
# Example: myGarage = [ {'VIN':'abcd', 'car': myHonda } , {'VIN':'fghi', 'car': myFiat} ]
# For the last example, when things is a list (also to a certain extend, for
# dictionaries as well), it will be much easier to name them using plural nouns. Avoid verbs.
# For example: myCars = [ {'VIN':'abcd', 'car': myHonda } , {'VIN':'fghi', 'car': myFiat} ]
# (it is nested, list of dictionary of ...) then the codes to loop thru will be like:
# for myCar in myCars : print(myCar['VIN'])
# Much easier to understand and follow.
#
# Recall that the standard .ipython (Jupyter notebook) files are saved as JSON format.
#

# %%
# import json
# %pip install requests
from cytoolz.dicttoolz import merge
import pandas as pd
import json
import requests
from requests.api import head
# from time import sleep  # sometimes, you want to slow down your API calls. For example, some providers limit the frequency of requests

# %%
# Basic
# Let us find out where is the International Space Station currently
#
response = requests.get("http://api.open-notify.org/iss-now.json")
status_code = response.status_code
print(f'status_code: {status_code}\n')  # 200

# Headers is a dictionary
print(f'headers: {response.headers}\n')
# Notice that the header is case-InSensiTiVe type
# Both lines below are the same
print(response.headers['content-Type'])
print(response.headers['content-type'])

# %% [markdown]
# The list of status codes:
#
# * 200 — Everything went okay, and the server returned a result (if any).
# * 301 — The server is redirecting you to a different endpoint. This can happen when a company switches domain names, or when an endpoint's name has changed.
# * 401 — The server thinks you're not authenticated. This happens when you don't send the right credentials to access an API.
# * 400 — The server thinks you made a bad request. This can happen when you don't send the information that the API requires to process your request (among other things).
# * 403 — The resource you're trying to access is forbidden, and you don't have the right permissions to see it.
# * 404 — The server didn't find the resource you tried to access.
#
# This "endpoint" does not need authentication, nor any other parameteres. So the requests
# call all went fine.
#

# %%
# We can now parse the response content to find info we need:
print(f'content type: {type(response.content)}')
print('content:')
print(response.content)  # 'b' indiates it is byte type.

# %%
# The content here is of JSON format. We can load it using:
print('JSON :')
print(response.json())

# %%
# Instead of using response.json(), the equivalent (longer version) will be
jsondata2 = json.loads(response.content)
# The opposite of json.loads() (from string to JSON) is json.dumps() (from JSON to string)
# Also, json.loads(string) will convert the string to json, while json.load(file_path)
# will convert the content in the file to json.
print(jsondata2)

print(response.json() == jsondata2)  # True # identical info
print(response.json() is jsondata2)  # False # these two are not shallow copies

# %%
# Examples of getting different status codes:
# 404, not found; wrong endpoint; read API doc
print(requests.get('http://api.open-notify.org/iss-pass').status_code)
# 400, bad reqeusts; read API doc to learn how to use this endpoint
print(requests.get('http://api.open-notify.org/iss-pass.json').status_code)

# %%
# After reading the documentation, what we need to use this endpoint iss-pass.json is:
parameters = {"lat": 37.78, "lon": -122.41}
response = requests.get(
    "http://api.open-notify.org/iss-pass.json", params=parameters)
print(f'status_code: {response.status_code}\n')  # 200, success
jsondata = response.json()
print(jsondata)
#
# This gets the same data as the command above
response = requests.get(
    "http://api.open-notify.org/iss-pass.json?lat=37.78&lon=-122.41")
# You can actually also paste this "url" in a browser directly.
#

# %%
# Many api access requires either authentication or key.
# Try the GitHub API
# for GitHub gwu-elo ac (exp 6.30.2022)
mygitheaders = {
    "Authorization": "Token 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"}
response = requests.get("https://api.github.com/user", headers=mygitheaders)
# In general, we can combine:
# response = requests.get("https://api.datasci.land/endpt.json", headers=mygitheaders, params=parameters)
user = response.json()
print(user)
# The endpoint here is 'user'

# %%
# Trying another endpoint,
response = requests.get(
    'https://api.github.com/repos/gwu-elo/Test_repo', headers=mygitheaders)
testrepo = response.json()
print(testrepo)

# %%
# Yet another endpoint,
response = requests.get(
    'https://api.github.com/users/gwu-elo/followers', headers=mygitheaders)
myfollowers = response.json()
print(response.json())  # It's an empty set (sad)

# %%
# The developers setup different endpoints to allow others CONSUME (access) their data.
# The data can be from the webserver, some database, wordpress (technically, the DB used
# by WP), etc etc. So roughly speaking, the data ultimately is from some data storage.
#
# We had use some direct DB connect library (mysql.connector) to retrieve data.
# By providing API endpoints, then you do not need to give away too much access to
# your data repositories.
#
# I have set up an endpoint to access the same database (MySQL) tables that you use in this
# class. With the api endpoint, I do not need to share with you a DB-level username/password,
# and also easily setup other restrictions.
# Let us try this:
# response = requests.get("https://api.datasci.land/endpt.json?apikey=K35wHcKuwXuhHTaz7zY42rCje")
# better to separate out the key and other parameters:
apikey = 'K35wHcKuwXuhHTaz7zY42rCje'
parameters = {"apikey": apikey}
response = requests.get(
    "https://api.datasci.land/endpt.json", params=parameters)
print(response.json())
print(f'\nheaders: {response.headers}')


# %%
# Okay, so we need to provide a table name...
parameters = {"apikey": apikey, 'table': "fooditems"}
response = requests.get(
    "https://api.datasci.land/endpt.json", params=parameters)
print(response.json())
# You can try these allowed tables, other than 'fooditems' : ['AStudentRecord','BikeShare','Dats_grades','Diet6wk','gapminder','gradAdmit','nfl2008_fga','Pizza','Titanic','USDANutrient','GSS_demographics'];

# Again, you can try the equivalent URL entered directly into a browser.

# %%
# FIRST, we will try CREATE
# We can also try using .post() instead of .get()
# Although in html, the main difference between get and post is that post variables
# are not shown in the url, while get will show as key/value pairs.
# In general, get is typically just passing parameters along with the requests
# that is neccessary, while post is often use to have the server create some new objects.
# Sometimes, the new object will be returned as response (like creating a new data record),
# other times, there might not be a new object at the end (like deleting a data table).

# Let us use our github example to test this:
payload = {"name": "test_api_post"}
# This will create a new repo named "test_api_post" (under the authenticated account),
# We need to pass in our authentication headers!
response = requests.post("https://api.github.com/user/repos",
                         json=payload, headers=mygitheaders)
print(f'headers: {response.headers}\n')
print(f'status_code: {response.status_code}\n')
# 201, means successfully created object on server.
# 422, failed (already exist?)
print(response.json())

# %%
# Next: UPDATES
payload = {"description": "Testing CRUD - Create, Read, Update, Delete",
           "name": "test_api_post"}
response = requests.patch(
    "https://api.github.com/repos/gwu-elo/test_api_post", json=payload, headers=mygitheaders)
print(f'headers: {response.headers}\n')
print(f'status_code: {response.status_code}\n')
# 200, success
print(response.json())

# %%
# Last one: DELETE
response = requests.delete(
    "https://api.github.com/repos/gwu-elo/test_api_post", headers=mygitheaders)
print(f'headers: {response.headers}\n')
print(f'status_code: {response.status_code}\n')
# 403, failed, permission not allowed
# 204, successfully deleted
# here, response is empty. response.json() results in an error.
print(response.content)

# %% [markdown]
# In development, you might have came across the term: CRUD, the four basic operations.
#
# Create, Read, Update, Delete.
#
# In SQL, they corresponds to INSERT, SELECT, UPDATE, DELETE.
#
# Here, they corresponds to post, get, patch/put, delete.
#
# These are the same/similar concepts in REST-ful API.  REST stands for
# Representational State Transfer (created by Roy Fielding). RESTful API uses http
# requests to access and consume/use data. POST, GET, PUT, DELETE are the four operations
# corresponding to CRUD in the standard context.


#%%
# From now on, we will use the api_dsLand() function to load certain data files
import dm6103 as dm
# accessible tables at dsLand via api_dsLand() functions are: 
# ['AStudentRecord','BikeShare','Dats_grades','Diet6wk','gapminder','gradAdmit','nfl2008_fga','Pizza','Titanic','USDANutrient','GSS_demographics', 'fooditems']

# %%
#
dfAdmit = dm.api_dsLand('gradAdmit')
# dfAdmit = pd.read_json('someFile.json') # if file saved locally
print(dfAdmit.info())

#%%
dfgap = dm.api_dsLand('gapminder', ind_col_name='id') # optional argument if you have an index column

# %%
#
# Above example has a very simple json data structure, and pandas just take this
# simple, well structured list, convert to dataframe.
# Let's try something more JSON-ish
dfFd = dm.api_dsLand("fooditems")

parameters = {"apikey": apikey, 'table': "fooditems"}
response = requests.get(
    "https://api.datasci.land/endpt.json", params=parameters)
jsonFd = response.json()
print(f'jsonFd data type: {type(jsonFd)}')
print(jsonFd)

# import pandas as pd
dfFd = pd.DataFrame(jsonFd)
print(f'dfFd data type: {type(dfFd)}')
print(dfFd)
print(dfFd.info())
print(f'dfFd shape: {dfFd.shape}')
print(dfFd.head())
# This doesn't quite work.
# %%
# Try again:
parameters = {"apikey": apikey, 'table': "fooditems"}
response = requests.get(
    "https://api.datasci.land/endpt.json", params=parameters)
jsonFood = response.json()
print(f'jsonFood data type: {type(jsonFood)}')
print(jsonFood)
fh = open('foodItems.json', 'w')
jdump = json.dumps(jsonFood)
print(jdump)
fh.write(jdump)
fh.close()

# %% [markdown]
# open the file in VSCode and view/format it.
# What structure should we use for the data?


#%%
###################### Try 'record_path' in json_normalize ####################
# reference:
# https://towardsdatascience.com/all-pandas-json-normalize-you-should-know-for-flattening-json-13eae1dfb7dd
# (Pdf printed version available.)
#


# %%
# How about loading json as Multi-index dataframe?
# Now, try method 1
# https://stackoverflow.com/questions/60204921/how-to-read-a-json-to-a-pandas-multiindex-dataframe
#
with open('foodItems.json') as data_file:
    d = json.load(data_file)

df = pd.concat({k: pd.DataFrame(v) for k, v in d.items()}).unstack(
    0).swaplevel(1, 0, axis=1).sort_index(axis=1)
print(df)
print(df.head())
print(f'\ndf shape: {df.shape}')  # (3,4)
print(df.columns)
# Creates 2-level multi-index columns


# %%
# Next, try method 2
#
# https://www.quora.com/How-do-I-convert-Json-to-Pandas-dataframe
#
df = pd.json_normalize(jsonFood, max_level=5)
print(df)
print(df.head())
print(f'\ndf shape: {df.shape}')  # (1,4)
print(df.columns)
# Flat single level column index, with prefixes

# %%
# Next, try method 2
#
# https://www.quora.com/How-do-I-convert-Json-to-Pandas-dataframe
#
df = pd.json_normalize(jsonFood['sweets'], max_level=5)
print(df)
print(df.head())
print(f'\ndf shape: {df.shape}')  # (1,4)
print(df.columns)
# Flat single level column index, with prefixes


# %%
# Problem is every time there is a list, instead of dictionary, json_normalize will only see
# the list as an object. It will not further normalize it.
#

# %%
# Next, try method 3
#
# https://stackoverflow.com/questions/45418334/using-pandas-json-normalize-on-nested-json-with-arrays 
# Try to do that recursively
#

df = pd.json_normalize(jsonFood, sep='_').pipe(
    lambda x: x.drop('sweets_items', 1).join(
        x.sweets_items.apply(lambda y: pd.Series(merge(y)))
    )
)
print(df)
print(df.head())
print(f'\ndf shape: {df.shape}')
print(df.columns)

# %%
# Now, try do it ourselves
# 
# First, We need to design the data structure that we see fit.
# 
# In our case, the "sweets" and "beverages" are rather different kinds, I would 
# keep them as two separate dataframe. 
# So let us handle the "sweets" donuts first.
#
def getHeadersFromJson(js, curheader):  # this basic js form should be a dictionary
    '''
    recursive function
    extract the keys in a nested JSON structure as colnames for pandas df
    :param JSON/dict js: Should be a dictionary on the top level
    :param str curheader: the cumulative current header name
    :return: headerlist - global list for all the column names
    '''
    headerlist = []
    for k, v in js.items():
        newheader = k if (curheader=='') else curheader +'_'+ k
        if (type(v) == dict):
            appendlist = getHeadersFromJson(v, newheader)
            headerlist.extend( appendlist )

        elif (type(v) == list):
            # get only the first row/element to try pull header the
            # first element SHOULD be a dictionary. Having a list of
            # list doesn't quite make sense in typical JSON structure.
            appendlist = getHeadersFromJson(v[0], newheader)
            headerlist.extend( appendlist )

        else:
            # assume it is the raw data now
            headerlist.append( newheader)
    return(headerlist)

with open('foodItems.json') as data_file:
    d2 = json.load(data_file)

d = d2['sweets'].copy()
sweetHeaderList = getHeadersFromJson(d,'')

print(sweetHeaderList)

# Got the list of column names for our intended dataframe structure!

#%%
# Now modify to pull the data into the dataframe as well.
import pandas as pd

def getFoodRowFromJson(js, df, currow, curheader): # df and currow already contains the header names, and last recorded values if no change is needed
    '''
    recursive function
    extract the values for the foodItems dataframe from the nested JSON 
    :param JSON/dict js: Should be a dictionary on the top level
    :param pd.dataframe df: the latest updated row values
    :param pd.dataframe currow: the latest/current updated row values
    :param str curheader: the cumulative current header name
    :return: pd.dataframe newrow - after added/updated new values
    '''
    thisrow = currow.copy()
    for k, v in js.items():
        newheader = k if (curheader=='') else curheader +'_'+ k
        if (type(v) == dict):
            newrow = getFoodRowFromJson(v, df, thisrow, newheader)
        elif (type(v) == list):
            # loop through the list and pull data
            for vi in v:
                newrow = getFoodRowFromJson(vi, df, thisrow, newheader)
        else:
            thisrow.loc[0,newheader] = v
            if (newheader == df.columns[-1] or newheader == df.columns[-4]) : 
                print(f'inside else if thisrow: {thisrow} \nNew header: {newheader} \nv: {v} \n')
                # reached the end of row, insert to df
                df.loc[len(df)] = thisrow.iloc[0] # tried append, which will create a new df, does not work recursively, and tried many other ways
                print(f'new df now: {df}')
                return currow
    return # should not get to this point. Must have reached the end before for properly structured JSON

foodItemsDf = pd.DataFrame(columns=sweetHeaderList) # empty frame to start
startrow  = foodItemsDf.copy().append(pd.Series(), ignore_index=True) # empty frame to start
with open('foodItems.json') as data_file:
    d2 = json.load(data_file)
    
d = d2['sweets'].copy()
getFoodRowFromJson(d, foodItemsDf, startrow, '');

print(foodItemsDf)
print(foodItemsDf.shape)


