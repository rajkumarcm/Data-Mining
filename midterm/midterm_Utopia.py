# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dm6103 as dm
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency

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
# Is there a difference between the two worlds such as the trend between industry and income is different.

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
tmp_df = pd.DataFrame({'World1':w1_by_gen, 'World2':w2_by_gen})

w1_change = w1_by_gen.unstack(level=-1).pct_change(periods=1, axis='columns')
w2_change = w2_by_gen.unstack(level=-1).pct_change(periods=1, axis='columns')
change_df = pd.DataFrame({'W1 Change': w1_change.iloc[:, 1], 'W2 Change': w2_change.iloc[:, 1]})

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
w1_by_gen.unstack(level=1).plot.bar(ax=axes[0, 0])
w2_by_gen.unstack(level=1).plot.bar(ax=axes[0, 1])
change_df.plot.bar(ax=axes[1, 0])
indices = list(range(change_df.shape[0]))
indices = list(filter(lambda x: x!=4, indices))
change_df2 = change_df.iloc[indices]
change_df2.plot.bar(ax=axes[1, 1])
axes[0, 0].set_ylabel('Number of Employees')
axes[0, 1].set_ylabel('Number of Employees')
axes[1, 0].set_ylabel('Percent')
axes[1, 1].set_ylabel('Percent')
axes[0, 0].set_title('World1')
axes[0, 1].set_title('World2')
axes[1, 0].set_title('Difference in employment count between genders')
axes[1, 1].set_title('Left plot with 4th industry ignored')
plt.show()

# Is there statistical significance in the frequencies of employment count between two worlds

_, p, _, _ = chi2_contingency(tmp_df)
print(f'There is definitely difference in number of people employed across different industries between'
      f'the two worlds as also the p value is {p}')

# Is there statistical significance in the frequencies of employment count between male and female in
# different industries in World1
_, p1, _, _ = chi2_contingency(w1_by_gen.unstack(level=-1))
_, p2, _, _ = chi2_contingency(w2_by_gen.unstack(level=-1))

print(f"I really want to declare immediately that World2 is the winner here, but the chi-squared test produced a "
      f"p value of {p2} for World2. I think it would make sense to run a t.test on percentage change in genders "
      f"across different industries between the two worlds. Perhaps the final result will allow us conclude "
      f"who the winner is.")




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
w1_by_eth = world1.groupby(['ethnic', 'gender']).agg(np.mean)
w2_by_eth = world2.groupby(['ethnic', 'gender']).agg(np.mean)


fig, axes = plt.subplots(1, 2, figsize=(10,4))
w1_by_eth.unstack(level=1).loc[:, 'income00'].plot.bar(ax=axes[0])
w2_by_eth.unstack(level=1).loc[:, 'income00'].plot.bar(ax=axes[1])
axes[0].set_ylabel('Income')
axes[1].set_ylabel('Income')
axes[0].set_title('World1')
axes[1].set_title('World2')
plt.show()
# Winner: World2
w2_points += 1

#%%
# There is definitely an issue here. Analyze. Remove this line once fixed

w1_married = world1[world1['marital']>0]
w1_married_by_ed = w1_married.loc[:, ['education', 'marital']]\
                           .groupby('education')\
                           .agg(np.sum)

w2_married = world2[world2['marital']>0]
w2_married_by_ed = w2_married.loc[:, ['education', 'marital']]\
                             .groupby('education')\
                             .agg(np.sum)

w1_w2_married_by_ed = w1_married_by_ed\
                        .join(w2_married_by_ed, on='education', how='inner',
                              lsuffix=' World1', rsuffix=' World2')

# fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# w1_married_by_ed.plot.bar(ax=axes[0])
# w2_married_by_ed.plot.bar(ax=axes[1])
# axes[0].set_ylabel('Count of married')
# axes[1].set_ylabel('Count of married')
# axes[0].set_title('World1')
# axes[1].set_title('World2')
plt.figure()
w1_w2_married_by_ed.plot.bar()
plt.show()

#%%

w1_married_by_age = world1.loc[:, ['age00', 'marital', 'gender']]
w1_married_by_age = w1_married_by_age.pivot_table(index='marital', columns='gender', values='age00', aggfunc=np.mean)
w2_married_by_age = world2.loc[:, ['age00', 'marital', 'gender']]
w2_married_by_age = w2_married_by_age.pivot_table(index='marital', columns='gender', values='age00', aggfunc=np.mean)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
w1_married_by_age.plot.bar(ax=axes[0])
w2_married_by_age.plot.bar(ax=axes[1])
axes[0].set_ylabel('Average age')
axes[1].set_ylabel('Average age')
axes[0].set_title('World1')
axes[1].set_title('World2')
plt.show()

ttest_ind(w1_married_by_age[0], w1_married_by_age[1], equal_var=False)

#%%
# Construct a contingency table for gender, and marital status. Run a chi squared test

w1_cont1 = pd.crosstab(world1.gender, world1.marital)
w2_cont1 = pd.crosstab(world2.gender, world2.marital)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(w1_cont1, annot=True, cmap="YlGnBu", ax=axes[0])
sns.heatmap(w2_cont1, annot=True, cmap="YlGnBu", ax=axes[1])
plt.show()

c1, p1, dof1, expected1 = chi2_contingency(observed=w1_cont1)
c2, p2, dof2, expected2 = chi2_contingency(observed=w2_cont1)

print(f'Whether you are a man or a woman, this should not affect your privilege in getting married. While it can '
      f'be seen from chi squared test for World1 p.value = {p1} the gender has no effect on marital status, in the '
      f'case of World2, the chi squared test produced a p.value = {p2} that shows being a man or woman has an '
      f'influence on the chances of getting married and I think of this as a bias.')

# Winner: World1
w1_points += 1

#%%

w1_married_by_gen = w1_married.loc[:, ['gender', 'marital']].groupby(['gender', 'marital']).agg('size')
w2_married_by_gen = w2_married.loc[:, ['gender', 'marital']].groupby(['gender', 'marital']).agg('size')

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
w1_married_by_gen.unstack(level=-1).plot.bar(ax=axes[0])
w2_married_by_gen.unstack(level=-1).plot.bar(ax=axes[1])
plt.show()
tmp_w1_married_by_gen = pd.DataFrame(w1_married_by_gen, columns=['Count'])
tmp_w2_married_by_gen = pd.DataFrame(w2_married_by_gen, columns=['Count'])
_, p, _, _ = chi2_contingency(tmp_w1_married_by_gen.join(tmp_w2_married_by_gen,
                                                         on=['gender', 'marital'], how='inner',
                                                         lsuffix=' World1', rsuffix=' World2'))
print(p)

#%%
# Does coming from a particular ethnic has any impact on getting married - part 1

w1_by_eth = w1_married.loc[:, ['ethnic', 'gender', 'marital']].groupby(['ethnic', 'gender']).size()
w2_by_eth = w2_married.loc[:, ['ethnic', 'gender', 'marital']].groupby(['ethnic', 'gender']).size()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
w1_by_eth.unstack(level=-1).plot.bar(ax=axes[0])
w2_by_eth.unstack(level=-1).plot.bar(ax=axes[1])
axes[0].set_ylabel('Count of married')
axes[1].set_ylabel('Count of married')
axes[0].set_title('World1')
axes[1].set_title('World2')
plt.show()

# Winner: World1
w1_points += 1
#%%
plt.figure()
plt.plot(world1.age00, world1.income00, '.b')
plt.show()





















