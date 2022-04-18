# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%% [markdown]
#
# # HW06 
# ## By: xxx
# ### Date: xxxxxxx
#

#%% [markdown]
# Let us improve our Stock exercise with Pandas now.
#
#%%
# Step 0, try reading the data file and make it a dataframe this time
# filepath = "/Users/edwinlo/GDrive_GWU/github_elo/GWU_classes_p/DATS_6103_DataMining/Class04_OOP/Dats_Grades.csv"
import os
import numpy as np
import pandas as pd
import dm6103 as dm
import matplotlib.pyplot as plt

dats = dm.api_dsLand('Dats_grades')
print("\nReady to continue.")

dm.dfChk(dats)

# What are the variables in the df? 
# What are the data types for these variables?

#%%
# The file has grades for a DATS class. Eight homeworks (out of 10 each), 2 quizzes (out of 100 each), and 2 projects (out of 100 each)
# Find out the class average for each item (HW, quiz, project)
# Hint, use .mean() function of pandas dataframe
dats.mean()

#%%
# create a new column right after the last hw column, to obtain the average HW grade.
# use column name HWavg. Make the average out of the total of 100.
# Hint: use .iloc to select the HW columns, and then use .mean(axis=1) to find the row average
dats.insert(8,'HWavg',0)  # create a new column first with 0s.
dats.HWavg = dats.iloc[:,0:8].mean(axis=1) * 10
dats.head()

#%%
# The course total = 30% HW, 10% Q1, 15% Q2, 20% Proj1, 25% Proj2. 
# Calculate the total and add to the df as the last column, named 'total', out of 100 max.
dats['total'] = 0.3 * dats.HWavg + 0.1* dats.Q1 + 0.15* dats.Q2 + 0.20* dats.Proj1 + 0.25* dats.Proj2
# if append column to the end, no need to use insert
dats.head()

#%%
# Now with the two new columns, calculate the class average for everything again. 
dats.mean()

#%%
# Save out your dataframe as a csv file
# import os
filecleaned = os.path.join( os.getcwd(), 'dats_clean.csv')
dats.to_csv(filecleaned)  


#%%
# In Week03 hw, we wrote a function to convert course total to letter grades. You can use your own, or the one from the solution file here.
def find_grade(total):
  # write an appropriate and helpful docstring
  """
  convert total score into grades
  :param total: 0-100 
  :return: str
  """
  # use conditional statement to set the correct grade
  grade = 'A' if total>=93 else 'A-' if total >= 90 else 'B+' if total >= 87 else 'B' if total >= 83 else 'B-' if total >=80 else 'C+' if total >= 77 else 'C' if total >= 73 else 'C-' if total >=70 else 'D' if total >=60 else 'F' 
  return grade  

#%%
# Let us create one more column for the letter grade, just call it grade.
# Instead of broadcasting some calculations on the dataframe directly, we need to apply (instead of broadcast) this find_grade() 
# function on all the elements in the total column
dats['grade'] = dats['total'].apply(find_grade)
dats.head()


#%%
# Create a bar chart for the grade distribution
# import matplotlib as plt 
# Hint: use .value_counts() on the grade column to make a bar plot


ax = dats['grade'].value_counts().plot.bar()
ax.set_title("DATS grades distribution")
ax.set_xlabel("Grades", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)

plt.savefig('gradesBarChart.jpg')

#%%


