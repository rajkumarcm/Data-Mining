# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('classic')
import dm6103 as dm

#%%
# First read in the gapminder dataset
# import os
# filepath = os.path.join( os.getcwd() ,'gapmindeR.csv')
# dfgap = pd.read_csv(filepath, index_col="id")
dfgap = dm.api_dsLand('gapminder', 'id')
dm.dfChk(dfgap)

#%%
# add contCode as numerical value 
contCodeList = list( dfgap.continent.unique() ) # contCodeList = ['Asia', 'Europe', 'Africa', 'Americas', 'Oceania']
dfgap['contCode']=pd.Categorical(dfgap.continent.apply( lambda x: contCodeList.index(x) ))
dfgap.head()

#%%[markdown]
# First, make some plots and animation
# example following https://python-graph-gallery.com/341-python-gapminder-animation/

#%%
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# plt.style.use('classic')

#%%
# pick a year
theyear = dfgap.year.unique()[0]

def plotGap1yr(yr):
  # plot 
  # initialize a figure
  my_dpi=96
  # fig = plt.figure(figsize=(680/my_dpi, 480/my_dpi), dpi=my_dpi)
  
  data = dfgap[dfgap.year==yr]

  # Change color with c and alpha. I map the color to the X axis value.
  plt.scatter(data['lifeExp'], data['gdpPercap'] , s=data['pop']/200000 , c=data['contCode'].cat.codes, cmap="Accent", alpha=0.6, edgecolors="white", linewidth=2)
  
  # Add titles (main and on axis)
  plt.yscale('log')
  plt.xlabel("Life Expectancy")
  plt.ylabel("GDP per Capita")
  plt.title("Year: "+str(yr) )
  plt.ylim(0,100000)
  plt.xlim(30, 90)

  # Save it
  # filepath = os.path.join( dirpath, 'Gapminder_step'+str(yr)+'.png')
  filepath = 'Gapminder_step'+str(yr)+'.png'
  plt.savefig(filepath, dpi=96)
  plt.gca()

plotGap1yr(theyear)

#%%
# Now loop thru all the years and create the png files
for yr in dfgap.year.unique():
  plotGap1yr(yr)

#%% [markdown]
# ## OPTIONAL
# create animated gif  
# ### Run these in bash (shell)  
# On Mac, you can use homebrew  
# If homebrew not install,  
# /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
# Now install ImageMagick
# brew install ImageMagick
#
# Finally, use convert (change to folder of the png's)
# convert -delay 80 Gapminder*.png animated_gapminder.gif
#
#
# Now you can view the gif 
# either in a browser (drag and drop)
# or other image software
# or IPython interactive window if you follow some steps and imports...

# %%
