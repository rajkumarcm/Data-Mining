#%%
from tkinter.font import families
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster
import dm6103 as dm
from sklearn.linear_model import *

# Part I
titanic = dm.api_dsLand('Titanic', 'id')

# Part II
nfl = dm.api_dsLand('nfl2008_fga')
nfl.dropna(inplace=True)

#%% [markdown]

# # Part I  
# Titanic dataset - statsmodels
# 
# | Variable | Definition | Key/Notes  |  
# | ---- | ---- | ---- |   
# | survival | Survived or not | 0 = No, 1 = Yes |  
# | pclass | Ticket class | 1 = 1st, 2 = 2nd, 3 = 3rd |  
# | sex | Gender / Sex |  |  
# | age | Age in years |  |  
# | sibsp | # of siblings / spouses on the Titanic |  |  
# | parch | # of parents / children on the Titanic |  |  
# | ticket | Ticket number (for superstitious ones) |  |  
# | fare | Passenger fare |  |  
# | embarked | Port of Embarkation | C: Cherbourg, Q: Queenstown, S: Southampton  |  
# 
titanic['embarked'] = titanic.embarked.replace({'C':1, 'Q':2, 'S':3, '':np.nan})
titanic.dropna(inplace=True)
titanic['sex'] = titanic.sex.replace({'male': 1, 'female': 0})
#%%
# ## Question 1  
# With the Titanic dataset, perform some summary visualizations:  
# 
# ### a. Histogram on age. Maybe a stacked histogram on age with male-female as two series if possible
plt.figure()
sns.histplot(x='age', hue='sex', data=titanic, kde=True)
plt.show()

# ### b. proportion summary of male-female, survived-dead  
print(pd.crosstab(titanic.sex, titanic.survived).rename(columns={0:'Dead', 1:'Survived'}))


# ### c. pie chart for “Ticketclass”  
pclass_count = titanic.loc[:, ['pclass', 'ticket']].groupby('pclass').agg('count')
plt.figure()
pclass_count.plot.pie(y='ticket')
plt.show()

# ### d. A single visualization chart that shows info of survival, age, pclass, and sex.  
tmp_cont = titanic.pivot_table(index='pclass', columns=['sex', 'survived'], values='age', aggfunc=np.mean)
plt.figure()
sns.heatmap(tmp_cont)
plt.show()

plt.figure()
tmp_cont.plot.bar()
plt.show()

#%%
# ## Question 2  
# Build a logistic regression model for survival using the statsmodels library.
# As we did before, include the features that you find plausible. Make sure categorical variables are use properly. 
# If the coefficient(s) turns out insignificant, drop it and re-build.  
from statsmodels.formula.api import logit

plt.figure()
sns.heatmap(titanic.loc[:, ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]\
       .corr(method='spearman'), annot=True)
plt.title('Correlation Plot')
plt.show()
# I don't know why sibsp and parch are correlated as they both are two different things
survived_logit = logit(formula='survived~C(pclass)+sibsp+parch+C(sex)+age', data=titanic).fit()
survived_logit.summary()



# ## Question 3  
# Interpret your result. What are the factors and how do they affect the chance of survival (or the survival odds ratio)? 
# What is the predicted probability of survival for a 30-year-old female with a second class ticket, no siblings, 3 parents/children on the trip? 
# Use whatever variables that are relevant in your model.  
print(f'\nCoefficients in log(odds-ratio):\n{survived_logit.params}')
print(f'\nCoefficients in odds-ratio:\n{np.exp(survived_logit.params)}')
#%%[markdown]
# The coefficients are represented by log odds ratio. Hence, any value beyond 0 is considered to have more than 50-50 chances of winning while anything below 0 is considered having more chances of not surviving.  
# pclass: Considering pclass.1 as the baseline, the chances of surviving decreases as we switch from first class ticket to second class and so on as the coeff is below 0.  
# sex: Since we are looking at log odds-ratio and the coeff is ~ -2.73 I would say males have very less chances of surviving. In other words, females are more likely to survive than males.  
# sibsp: In terms of odds-ratio, not log, the chances of surviving decreases by 1-(0.75)^sibsp and this makes sense. For instance, having a sibsp of 2 means the chances of surviving decreases by 42.78%  
# parch: In terms of odds-ratio, as the number of parch increases from 1 to 2, the chances of surviving decreases by 1-(0.979138)^2 = 4.128%  
# age: Increase is age is linked to less chances of surviving. An increase in age means, from odds-ratio perspective, the chances of surviving decrease by 1-(0.981928)^2 = 3.58%  

#%%
x_test = pd.DataFrame({'age':[30], 'sex':[0], 'pclass':[2], 'sibsp':[0], 'parch':[3]})
tmp_prob = survived_logit.predict(x_test, 'survived~C(pclass)+sibsp+parch+C(sex)+age')
print(f'The chances of 30yrs old female with 3 parch holding a 2nd second class ticket is {round(tmp_prob.values[0]*100, 2)}% surviving')

#%%
# ## Question 4  
# Try three different cut-off values at 0.3, 0.5, and 0.7. 
# What are the a) Total accuracy of the model 
# b) The precision of the model (average for 0 and 1), and 
# c) the recall rate of the model (average for 0 and 1)
cutoff_values = [0.3, 0.5, 0.7]
df_metrics = pd.DataFrame(np.zeros([len(cutoff_values), 3]), columns=['Precision', 'Recall', 'Accuracy'])
df_metrics['cutoff'] = cutoff_values
df_metrics.set_index('cutoff', inplace=True)
def get_metrics(cutoff_vals, probs, df_metrics):
       for co_val in cutoff_vals:
              pred = probs.copy()
              pred[probs <= co_val] = 0
              pred[probs > co_val] = 1
              pred = pred.astype(int)
              cm = np.zeros([2, 2])

              # Confusion Matrix
              # actual x predicted
              for i in range(len(pred)):
                     cm[titanic.survived.iloc[i], pred.iloc[i]] = cm[titanic.survived.iloc[i], pred.iloc[i]] + 1
              
              tp = cm[0, 0]
              fp = cm[1, 0]
              tn = cm[1, 1]
              fn = cm[0, 1]
              precision = tp/(tp + fp)
              recall = tp/(tp + fn)
              accuracy = (tp+tn)/(tp+fp+tn+fn)
              df_metrics.loc[co_val] = [precision, recall, accuracy]
       return df_metrics
df_metrics = get_metrics(cutoff_values, survived_logit.predict(titanic), df_metrics)
print(df_metrics)


#%%[markdown]
# # Part II  
# NFL field goal dataset - SciKitLearn
# 
# | Variable | Definition | Key/Notes  |  
# | ---- | ---- | ---- |   
# | AwayTeam | Name of visiting team | |  
# | HomeTeam | Name of home team | |  
# | qtr | quarter | 1, 2, 3, 4 |  
# | min | Time: minutes in the game |  |  
# | sec | Time: seconds in the game |  |  
# | kickteam | Name of kicking team |  |  
# | distance | Distance of the kick, from goal post (yards) |  |  
# | timerem | Time remaining in game (seconds) |  |  
# | GOOD | Whether the kick is good or no good | If not GOOD: |  
# | Missed | If the kick misses the mark | either Missed |  
# | Blocked | If the kick is blocked by the defense | or blocked |  
# 

#%% 
# ## Question 5  
# With the nfl dataset, perform some summary visualizations.  
plt.subplots()
sns.histplot(nfl.loc[:, 'min'], kde=True)
plt.show()
team_kick_perf = nfl.loc[:, ['kickteam', 'distance']].groupby('kickteam').agg(np.mean)

from sklearn.cluster import KMeans
n_clusters = 50
cluster_indices = KMeans(n_clusters=50).fit_predict(X=nfl.loc[:, ['min']])
subset = nfl.copy()
subset['cluster'] = cluster_indices

min_distance_perf = subset.loc[:, ['cluster', 'min', 'distance']].groupby(['cluster']).agg(np.mean)
min_distance_perf.sort_values(by='min', inplace=True)
plt.figure()
plt.plot(min_distance_perf.loc[:, 'min'], min_distance_perf.distance, '.b')
plt.xlabel('Minutes')
plt.ylabel('Distance')
plt.title('Performance at the end')
plt.grid(True)
plt.show()

qtr_gkick = pd.crosstab(nfl.qtr, nfl.GOOD)
plt.figure()
qtr_gkick.plot.bar()
plt.title('Kicks in each quarter')
plt.grid(True)
plt.show()

min_kick = pd.DataFrame(np.zeros([n_clusters, 3]), columns=['minutes', 'Good_kicks', 'Bad_kicks']) # change the 100 to dynamic
for c in range(n_clusters): # Change the 100 to dynamic
       tmp_subset = subset.loc[subset.cluster==c]
       min_kick.iloc[c, 0] = tmp_subset.loc[:, 'min'].mean()
       min_kick.iloc[c, 1] = tmp_subset.loc[:, 'GOOD'].sum()
       min_kick.iloc[c, 2] = tmp_subset.shape[0] - tmp_subset.loc[:, 'GOOD'].sum()

plt.figure()
plt.plot(min_kick.minutes, min_kick.Good_kicks, '.b', linewidth=1, label='Good Kicks')
plt.plot(min_kick.minutes, min_kick.Bad_kicks, '.r', linewidth=1, label='Bad kicks')
plt.xlabel('Minutes')
plt.ylabel('Good kicks')
plt.grid(True)
plt.title('How many good kicks at each average minute')
plt.ylim([0, 50]) # Ignore that one outlier
plt.show()

plt.figure(figsize=(15, 12))
sns.heatmap(nfl.loc[:, nfl.columns!='season'].corr(), annot=True)
plt.show()

#%%[markdown]
# Confidence interval of average minute during which the good kick took place.
#%%
import scipy.stats as st
from sklearn.metrics import f1_score
st.norm.interval(alpha=0.95,
                 loc=nfl.loc[nfl.GOOD==1, 'min'].mean(),
                 scale=st.sem(nfl.loc[nfl.GOOD==1, 'min']))

# ## Question 6  
# Using the SciKitLearn library, build a logistic regression model overall (not individual team or kicker) to predict the chances of a 
# successful field goal. What variables do you have in your model? 

# I don't know what field goal means... I guess a good kick

reg_cnames = ['min', 'sec', 'ydline', 'distance', 'defscore', 'down', 'togo', 'kickdiff', 'offscore']
lr = LogisticRegressionCV(Cs=[0.01], cv=5, scoring='f1').fit(nfl.loc[:, reg_cnames], nfl.GOOD)
plt.figure()
plt.plot(lr.coef_.flatten())
plt.grid(True)
plt.title('Coefficients')
plt.show()

print(f'Scores across k fold:\n{lr.scores_[1]}')
print(f'The variables in the model would be\n{reg_cnames}')

#%%
# ## Question 7  
# Someone has a feeling that home teams are more relaxed and have a friendly crowd, they should kick better field goals. 
# Use your model to find out if that is subtantiated or not. 

# Football is not my area of expertise and for the assignment, I do not find that it is appropriate to ask someone to learn 
# something that is uncommon. If it was for the project, then yes, I certainly would be willing to invest time in understanding
# how the game works. I do not understand what hometeam and awayteams are?
# This does not show whether or not I understand logistic regression instead lack of understanding of the game itself.
subset_ht = nfl.loc[nfl.homekick==1, reg_cnames]
subset_at = nfl.loc[nfl.homekick==0, reg_cnames]
y_ht_true = nfl.loc[nfl.homekick==1, 'GOOD']
y_at_true = nfl.loc[nfl.homekick==0, 'GOOD']

print(f'Yes it has very good chances of making goals as the chances of good kicking is: \
{round(lr.score(subset_ht, y_ht_true)*100, 2)}%')

#%%
#  
# ## Question 8    
# From what you found, do home teams and road teams have different chances of making a successful field goal? 
# If one does, is that true for all distances, or only with a certain range?
# 
print(f'They almost have same chance of making a goal and the probability would be {round(lr.score(subset_at, y_at_true)*100, 2)}%')

# %%
# titanic.dropna()


# %%
