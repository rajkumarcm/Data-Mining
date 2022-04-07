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
print('I wish to decide which world is an epitome of Utopia by assessing data on various criteria that should '
      'aim at answering some of the SMART questions as follows:')
print('1. Is there a difference between the two worlds such as the trend between industry and income is different.')
print('2. Is there a gender bias in people who work at different industries in either of the worlds')
print('3. Is there a gender bias in income people make in either of the worlds')
print('4. Is there a gender bias in education in either of the worlds')
print('5. Does the income people make is biased by ethnicity in any of the worlds')
print('6. Does any of the world shows sign of delayed marriage - perhaps a useless smart question')
print('7. Is there a gender bias in getting married in any of the worlds')
print('8. Is there an ethnic bias in getting married in any of the worlds')
print('9. Is there an ethnic bias in working at different industries in either of the worlds')
print('10. Is there an ethnic bias in the income people make - this is ignored as considered redundant.')
print('11. Is there an ethnic bias in education in either of the worlds')
print('12. Does income has any effect on the marital status in any of the worlds')
print('Final conclusion will be points based.')
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
# Gender bias in people working in different industries in both worlds
w1_by_gen = world1.groupby(['industry', 'gender']).size().unstack(level=-1)
w2_by_gen = world2.groupby(['industry', 'gender']).size().unstack(level=-1)

w1_change = w1_by_gen.pct_change(periods=1, axis='columns')
w2_change = w2_by_gen.pct_change(periods=1, axis='columns')
change_df = pd.DataFrame({'W1 Change': w1_change.iloc[:, 1], 'W2 Change': w2_change.iloc[:, 1]})

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
w1_by_gen.plot.bar(ax=axes[0, 0])
w2_by_gen.plot.bar(ax=axes[0, 1])
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

# Is there statistical significance in the frequencies of employment count between male and female in
# different industries in World1
_, p1, _, _ = chi2_contingency(w1_by_gen)
_, p2, _, _ = chi2_contingency(w2_by_gen)

print(f"I really want to declare immediately that World2 is the winner here, but the chi-squared test produced a "
      f"p value of {round(p2, 4)} for World2. I think it would make sense to run a t.test on percentage change in genders "
      f"across different industries for both the worlds. If the following test would produce a p.value that shows"
      f"statistical significance in percentage change between genders subgroups in two worlds then we could conclude"
      f"that World2 is the winner here.")

_, p3 = ttest_ind(w1_change.iloc[:, 1], w2_change.iloc[:, 1], equal_var=False)
print(f"Even t.test for the percentage change between genders in two subgroups produced a p.value={round(p3, 4)} "
      f"that represents no statistical significance, I conclude the winner is neither of them on the basis of statistical "
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
# Probably a duplicate analysis... may be not...
tmp_married_w1 = world1.loc[:, ['gender', 'marital']]
tmp_married_w2 = world2.loc[:, ['gender', 'marital']]

w1_married_by_gen = pd.crosstab(tmp_married_w1.gender, tmp_married_w1.marital)
w2_married_by_gen = pd.crosstab(tmp_married_w2.gender, tmp_married_w2.marital)


tmp_w1_married_by_gen = w1_married_by_gen.stack(level=0)
tmp_w2_married_by_gen = w2_married_by_gen.stack(level=0)
w1_w2_married_by_gen = pd.DataFrame({'World1': tmp_w1_married_by_gen,
                                     'World2': tmp_w2_married_by_gen})

w1_w2_married_by_gen2 = w1_w2_married_by_gen.unstack(level=-1).swaplevel(i=0, j=1, axis=1).sort_index(axis=1, level=0)
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
idx = 0
for i in range(2):
    for j in range(2):
        idx = i*2 + j
        w1_w2_married_by_gen2.loc[:, idx].plot.bar(ax=axes[i, j])
        axes[i, j].set_title(f'Marital status {idx}')
plt.show()

input('Press any key once you are done looking at the plot...')
plt.figure()
w1_w2_married_by_gen2.sum(axis=0).unstack(level=-1).plot.bar()
plt.show()
# It would not make sense to run a chi-squared here as difference in row is balanced in the subsequent rows.
# I mean, if world1 has upper hand in one section, the world2 has upper hand in the second section. This counteracts
# the points and makes the winner neutral.

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
# Industry and ethnic
ethnic_ind_w1 = pd.crosstab(world1.industry, world1.ethnic)
ethnic_ind_w2 = pd.crosstab(world2.industry, world2.ethnic)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
ethnic_ind_w1.plot.bar(ax=axes[0])
ethnic_ind_w2.plot.bar(ax=axes[1])
axes[0].set_title('How ethnicity affect employment in World1')
axes[1].set_title('How ethnicity affect employment in World2')
axes[0].set_ylabel('Number of employeed')
axes[1].set_ylabel('Number of employeed')
plt.show()

_, ethnic_ind_p1, _, _ = chi2_contingency(ethnic_ind_w1)
_, ethnic_ind_p2, _, _ = chi2_contingency(ethnic_ind_w2)

print(f'The visual plot shows that there is ethnicity bias in world1 as also confirmed by chi-squared test '
      f'with a p.value={ethnic_ind_p1}, whereas in case of the second world, this is neutral and likewise the '
      f'p.value would be {ethnic_ind_p2}. Based on both the results, I conclude the winner for this test would be'
      f'World2')

# Winner: World2
w2_points += 1
# I wish not to analyse if there is ethnic bias in getting income that I find it being redundant in some sense.
# As you have seen, ethnicity has effect on industry and different industries have different income levels.
# When you say ethnic -> industry and industry -> income then it makes sense to think ethnic -> income
# This assumption is also backed by the correlation plot as it shows there is some connection between ethnic and income.

#%%
# Ethnic and education
ethnic_ed_w1 = pd.crosstab(world1.education, world1.ethnic)
ethnic_ed_w2 = pd.crosstab(world2.education, world2.ethnic)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
ethnic_ed_w1.plot.bar(ax=axes[0])
ethnic_ed_w2.plot.bar(ax=axes[1])
axes[0].set_title('Analyzing the bias of ethnicity in World1')
axes[1].set_title('Analyzing the bias of ethnicity in World2')
axes[0].set_ylabel('Number of students at x grade')
plt.show()

print(f'I am convinced by this result not to run a statistical test as any subtle difference could'
      f' be attributed to mere noise.')

#%%
# Does income influence marriage in either of the worlds
income_marital_w1 = world1.loc[:, ['income00', 'marital']]\
                          .groupby('marital')\
                          .agg(np.mean)

income_marital_w2 = world2.loc[:, ['income00', 'marital']]\
                          .groupby('marital')\
                          .agg(np.mean)

w1_w2_income_marital = income_marital_w1.join(income_marital_w2, on='marital', how='inner',
                                              lsuffix=' World1', rsuffix=' World2')

plt.figure()
w1_w2_income_marital.plot.bar()
plt.title('Does income has any effect on marital status')
plt.ylabel('Mean income')
plt.show()

print('Verifying by ANOVA on whether income has any effect on marital status in World1')
income_marital_model_w1 = ols('income00~C(marital)', data=world1).fit()
print(anova.anova_lm(income_marital_model_w1))

print('Verifying by ANOVA on whether income has any effect on marital status in World2')
income_marital_model_w2 = ols('income00~C(marital)', data=world2).fit()
print(anova.anova_lm(income_marital_model_w2))

print('Not surprised by the results, I can not see any difference in income between the different subgroups'
      'of marital status.')

#%%
print('End of analyses--------------------------------------')
print('Points............')
print(f'World1: {w1_points}')
print(f'World2: {w2_points}')
print('I can make weighted average of the points to draw conclusion, but I believe the coefficients would reflect'
      'my perception that may be subjective. Hence, based on the points and samples provided, I wish to conclude '
      'that World2 can be an ideal place (utopia).')



