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
from statsmodels.formula.api import ols
from statsmodels.stats import anova

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

_, p3 = ttest_ind(w1_change.iloc[:, 1], w2_change.iloc[:, 1], equal_var=False)
print(f"Since the p value {p3} also for the ttest for the percentage change between genders in two subgroups shows "
      f"no statistical significance, I conclude the winner is neither of them on the basis of statistical "
      f"results.")
# Winner: Neutral (Although unexpected)

#%%

w1_by_gen = world1.groupby('gender')\
                  .agg(np.mean)
w1_by_gen_mean = w1_by_gen.loc[:, 'income00']

w2_by_gen = world2.groupby('gender')\
                  .agg(np.mean)
w2_by_gen_mean = w2_by_gen_mean = w2_by_gen.loc[:, 'income00']

plt.figure()
plt.title('Income of male and female in both worlds')
bar1 = w1_by_gen_mean.plot.bar(color='black', label='World1', alpha=0.5)
bar2 = w2_by_gen_mean.plot.bar(color='silver', label='World2', alpha=0.5)
plt.ylabel('Income')
plt.legend()
# bar1.bar_label(padding=3)
# bar2.bar_label(padding=3)
plt.show()
# Winner: On the next cell
#%%

ttest_inc = ttest_ind(world1['income00'], world2['income00'], equal_var=False, alternative='two-sided')
print(f'I decided to run t.test on the entire income data between the two worlds to understand if there is'
      f' statistical significance between them and it produced a p value = {round(ttest_inc.pvalue, 4)} that tells '
      f' any difference that exists is not due to more than just chance. Although the average may be similar'
      f' there could still be differences under different levels. Hence, I wish to drill down even further.')

ttest_inc_by_gen2 = ttest_ind(world1.loc[world1['gender']==0, 'income00'], world1.loc[world1['gender']==1, 'income00'],
                             equal_var=False, alternative='two-sided')

ttest_inc_by_gen3 = ttest_ind(world2.loc[world2['gender']==0, 'income00'], world2.loc[world2['gender']==1, 'income00'],
                             equal_var=False, alternative='two-sided')

print(f'The test confirms that there is a gender bias in world1, in terms of income as the pvalue is '
      f'{round(ttest_inc_by_gen2.pvalue, 4)}, whereas in case of World2, there is not much difference between the '
      f'income made by male and female as was confirmed by the t.test that produced a p value of '
      f'{round(ttest_inc_by_gen3.pvalue, 4)}')

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

# Winner: Neutral

#%%
# Analyzing Ethnic bias in income
w1_by_eth = world1.loc[:, ['ethnic', 'gender', 'income00']]\
                  .groupby(['ethnic', 'gender']).agg(np.mean)
w2_by_eth = world2.loc[:, ['ethnic', 'gender', 'income00']]\
                  .groupby(['ethnic', 'gender']).agg(np.mean)


fig, axes = plt.subplots(1, 2, figsize=(10,4))
w1_by_eth.iloc[:, 0].unstack(level=1).plot.bar(ax=axes[0])
w2_by_eth.iloc[:, 0].unstack(level=1).plot.bar(ax=axes[1])
axes[0].set_ylabel('Income')
axes[1].set_ylabel('Income')
axes[0].set_title('World1')
axes[1].set_title('World2')
plt.show()

# Include statistical results (ANOVA perhaps as there are three ethnic groups)
ethnic_inc_w1_model = ols('income00~C(ethnic)', data=world1).fit()
print(f'ANOVA test on whether ethnicity has any effect on the income level in World1')
print(anova.anova_lm(ethnic_inc_w1_model))
print(f'ANOVA test on whether ethnicity has any effect on the income level in World2')
ethnic_inc_w2_model = ols('income00~C(ethnic)', data=world2).fit()
print(anova.anova_lm(ethnic_inc_w2_model))
print(f'As it is clear from plots and also from the ANOVA test result, the World1 has ethnic bias in the income '
      f'people make. Hence the winner for this test would be World2')
# Winner: World2
w2_points += 1

#%%
# There is definitely an issue here. Analyze. Remove this line once fixed

w1_married = world1[world1['marital']>0]
w1_married_by_ed = w1_married.loc[:, ['education', 'marital']]\
                           .groupby('education')\
                           .agg('count')

w2_married = world2[world2['marital']>0]
w2_married_by_ed = w2_married.loc[:, ['education', 'marital']]\
                             .groupby('education')\
                             .agg('count')

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
plt.title('Married count based on education level')
w1_w2_married_by_ed.plot.bar()
plt.ylabel('Married count')
plt.show()
_, mar_by_ed_p, _, _ = chi2_contingency(w1_w2_married_by_ed)
print(f'As was conveyed by the visual plot, the statistical test also confirmed that there is not much '
      f'statistical significance between the two worlds in terms of how education enables one to get married. '
      f'The p.value would be {round(mar_by_ed_p, 4)}')

#%%
# Is there difference in age between the subgroups of marital - this is perhaps a useless test

w1_married_by_age = world1.loc[:, ['age00', 'marital', 'gender']]
w1_married_by_age = w1_married_by_age.pivot_table(index='marital', columns='gender', values='age00', aggfunc=np.mean)
w2_married_by_age = world2.loc[:, ['age00', 'marital', 'gender']]
w2_married_by_age = w2_married_by_age.pivot_table(index='marital', columns='gender', values='age00', aggfunc=np.mean)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
w1_married_by_age.plot.bar(ax=axes[0])
w2_married_by_age.plot.bar(ax=axes[1])
axes[0].set_ylabel('Average age')
axes[1].set_ylabel('Average age')
axes[0].set_title('World1 avg age of marital status')
axes[1].set_title('World2 avg age of marital status')
plt.show()

print('Analyzing if age for different subgroups of marital status has any statistical significance in World1')
age_marital_w1_model = ols('age00~C(marital)', data=world1).fit()
print(anova.anova_lm(age_marital_w1_model))
print('Analyzing if age for different subgroups of marital status has any statistical significance in World2')
age_marital_w2_model = ols('age00~C(marital)', data=world2).fit()
print(anova.anova_lm(age_marital_w2_model))
print('The whole intention of this test was to ensure in particular I do not want to see there is difference '
      'between different subgroups as this means one subgroup suffers delayed married and compare this to world2 so'
      'as to find out which world is better from this perspective. People could have different perception on this, '
      'but this is at least my understanding.')
#%%
# Construct a contingency table for gender, and marital status. Run a chi squared test

gender_marital_cont_w1 = pd.crosstab(world1.gender, world1.marital)
gender_marital_cont_w2 = pd.crosstab(world2.gender, world2.marital)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(gender_marital_cont_w1, annot=True, cmap="YlGnBu", ax=axes[0])
sns.heatmap(gender_marital_cont_w2, annot=True, cmap="YlGnBu", ax=axes[1])
plt.show()

_, p1, _, _ = chi2_contingency(observed=gender_marital_cont_w1)
_, p2, _, _ = chi2_contingency(observed=gender_marital_cont_w2)

print(f'Whether you are a man or a woman, this should not affect your privilege in getting married. While it can '
      f'be seen from chi squared test for World1 p.value = {p1} the gender has no effect on marital status, in the '
      f'case of World2, the chi squared test produced a p.value = {p2} that shows being a man or woman has an '
      f'influence on the chances of getting married and I think of this as a bias.')

# Winner: World1
w1_points += 1

#%%
# Probably a duplicate analysis...
w1_married_by_gen = w1_married.loc[:, ['gender', 'marital']].groupby(['gender', 'marital']).agg('size')
w2_married_by_gen = w2_married.loc[:, ['gender', 'marital']].groupby(['gender', 'marital']).agg('size')

fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4))

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

w1_by_eth = w1_by_eth.unstack(level=-1)
w2_by_eth = w2_by_eth.unstack(level=-1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
w1_by_eth.plot.bar(ax=axes[0])
w2_by_eth.plot.bar(ax=axes[1])
axes[0].set_ylabel('Count of married')
axes[1].set_ylabel('Count of married')
axes[0].set_title('World1')
axes[1].set_title('World2')
plt.show()

# Include hypothesis test to show that the differences are indeed present.
# I don't think it is possible to use ANOVA in this context as we are trying to compare the frequencies and
# not the quantitative variable itself. Hence, a chi-squared appears more appropriate.
# However there is a problem even with that approach. What we have is more than 2 variables in picture...
_, eth_p1, _, _ = chi2_contingency(w1_by_eth)
_, eth_p2, _, _ = chi2_contingency(w2_by_eth)

print(f'Gender bias was already found in one of our previous analysis and this is not surprising to see this again. '
      f'As for the intention of the analysis, I wanted to see if there is an ethnic bias to getting married and '
      f'as confirmed by visual plots, there is no ethnic bias to getting married in both worlds. The p.value for '
      f'world1 would be {round(eth_p1, 4)}, while for world2 the p.value would be {round(eth_p2, 4)}')

# Winner: World1
w1_points += 1
#%%
print(f'w1_points: {w1_points}')
print(f'w2_points: {w2_points}')





















