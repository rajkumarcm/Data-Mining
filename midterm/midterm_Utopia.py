# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dm6103 as dm
from scipy.stats import ttest_ind

world1 = dm.api_dsLand('World1', 'id')
world2 = dm.api_dsLand('World2', 'id')

print("\nReady to continue.")


#%% [markdown]
# # Two Worlds 
# 
# I was searching for utopia, and came to this conclusion: If you want to do it right, do it yourself. 
# So I created two worlds. 
#
# Data dictionary:
# * age00: the age at the time of creation. This is only the population from age 30-60.  
# * education: years of education they have had. Education assumed to have stopped. A static data column.  
# * marital: 0-never married, 1-married, 2-divorced, 3-widowed  
# * gender: 0-female, 1-male (for simplicity)  
# * ethnic: 0, 1, 2 (just made up)  
# * income00: annual income at the time of creation   
# * industry: (ordered with increasing average annual salary, according to govt data.)   
#   0. leisure n hospitality  
#   1. retail   
#   2. Education   
#   3. Health   
#   4. construction   
#   5. manufacturing   
#   6. professional n business   
#   7. finance   
# 
# 
# Please do whatever analysis you need, convince your audience both, one, or none of these 
# worlds is fair, or close to a utopia. 
# Use plots, maybe pivot tables, and statistical tests (optional), whatever you deem appropriate 
# and convincing, to draw your conclusions. 
# 
# There are no must-dos (except plots), should-dos, cannot-dos. The more convenicing your analysis, 
# the higher the grade. It's an art.
#

#%%
# Lets decide the utopia based on points
w1_points = 0
w2_points = 0

#%% Standardise data
def standardise(df):
    df -= df.mean(axis=0)
    df /= df.std(axis=0)
    return df

#%%
print(f'World1:\n{world1.head(5)}\n')
print(f'World2:\n{world2.head(5)}\n')

#%% Correlation between variables
w1_corr = world1.corr(method='spearman')
w2_corr = world2.corr(method='spearman')

fig, axes = plt.subplots(1, 2)
sns.heatmap(w1_corr, vmin=-1, vmax=1,
                      center=0, cbar=True, square=True, ax=axes[0])
sns.heatmap(w1_corr, vmin=-1, vmax=1,
                      center=0, cbar=True, square=True, ax=axes[1])
plt.show()


#%%
w1_by_ind = world1.groupby('industry')\
                  .agg(np.mean)
w1_by_ind_mean = w1_by_ind.loc[:, 'income00']

w2_by_ind = world2.groupby('industry')\
                  .agg(np.mean)
w2_by_ind_mean = w2_by_ind.loc[:, 'income00']

plt.figure()
w1_by_ind_mean.plot(color='blue', label='World1', alpha=0.5)
w2_by_ind_mean.plot(color='red', label='World2', alpha=0.5)
plt.legend()
plt.show()
# Since the plot shows that as the industry level increases, there is higher income, I think it is not necessary to do
# hypothesis test on this.
# Winner: NEUTRAL

#%%
ttest_inc_by_ind = ttest_ind(w1_by_ind_mean, w2_by_ind_mean, equal_var=False, alternative='two-sided')
print(f'Since the p value for t.test on income by industry between two worlds is {ttest_inc_by_ind.pvalue}, '
      f'both the worlds share similar income by industry trend.')
# Winner: NEUTRAL

#%%
# Industry and gender since there is some correlation between them
w1_by_gen = world1.groupby(['industry', 'gender']).size()
w2_by_gen = world2.groupby(['industry', 'gender']).size()

fig, axes = plt.subplots(1, 2, figsize=(10,4))
w1_by_gen.unstack(level=1).plot.bar(ax=axes[0])
w2_by_gen.unstack(level=1).plot.bar(ax=axes[1])
axes[0].set_ylabel('Count')
axes[1].set_ylabel('Count')
axes[0].set_title('World1')
axes[1].set_title('World2')
plt.show()


#%%

w1_by_gen = world1.groupby('gender')\
                  .agg(np.mean)
w1_by_gen_mean = w1_by_gen.loc[:, 'income00']

w2_by_gen = world2.groupby('gender')\
                  .agg(np.mean)
w2_by_gen_mean = w2_by_gen_mean = w2_by_gen.loc[:, 'income00']

plt.figure()
bar1 = w1_by_gen_mean.plot.bar(color='black', label='World1', alpha=0.5)
bar2 = w2_by_gen_mean.plot.bar(color='silver', label='World2', alpha=0.5)

plt.legend()
# bar1.bar_label(padding=3)
# bar2.bar_label(padding=3)
plt.show()
# Winner: On the next cell
#%%

ttest_inc = ttest_ind(world1['income00'], world2['income00'], equal_var=False, alternative='two-sided')
print(f'Since the p value for t.test on the income between the two worlds is {ttest_inc.pvalue}, '
      f'there is not statistical difference between the average income between the two worlds.')

ttest_inc_by_gen2 = ttest_ind(world1.loc[world1['gender']==0, 'income00'], world1.loc[world1['gender']==1, 'income00'],
                             equal_var=False, alternative='two-sided')
print(f'The test confirms that there is a racial bias in world1, in terms of income as the pvalue is '
      f'{ttest_inc_by_gen2.pvalue}')

ttest_inc_by_gen3 = ttest_ind(world2.loc[world2['gender']==0, 'income00'], world2.loc[world2['gender']==1, 'income00'],
                             equal_var=False, alternative='two-sided')
print(f'The test confirms that there is not statistical difference in income between male and female in the '
      f'second world {ttest_inc_by_gen3.pvalue}')
# Winner: Certainly world2
w2_points += 1
#%%
# Education and gender since there is some correlation between them
w1_by_gen = world1.groupby(['education', 'gender']).size()
w2_by_gen = world2.groupby(['education', 'gender']).size()

fig, axes = plt.subplots(1, 2, figsize=(10,4))
w1_by_gen.unstack(level=1).plot.bar(ax=axes[0])
w2_by_gen.unstack(level=1).plot.bar(ax=axes[1])
axes[0].set_ylabel('Count')
axes[1].set_ylabel('Count')
axes[0].set_title('World1')
axes[1].set_title('World2')
plt.show()

# w1_inc_pivot.plot.bar()
# plt.show()

#%%
































