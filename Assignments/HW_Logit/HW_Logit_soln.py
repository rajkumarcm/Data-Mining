#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dm6103 as dm

# Part I
titanic = dm.api_dsLand('Titanic', 'id')

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
#%%
# ## Question 1  
# With the Titanic dataset, perform some summary visualizations:  
# 
# ### a. Histogram on age. Maybe a stacked histogram on age with male-female as two series if possible

# fig, axes = plt.subplots()
plt.hist(x = titanic.age, bins = 20, color = 'steelblue', edgecolor = 'black' )
plt.show()

#%%
# stacked bar chart
# fig, axes = plt.subplots()
sns.histplot( data=titanic, x="age", bins = 20, kde = True, hue="sex", multiple="stack" )
plt.title('Stacked histogram on age for male-female')
plt.show()

#%%
# ### b. proportion summary of male-female, survived-dead  
prop_sex = titanic.sex.value_counts(normalize=True)  #  titanic.groupby('sex').agg({'sex':'count'}) # ['sex'].apply(lambda x: 100 * x / float(x.sum()))
print(type(prop_sex))
print(prop_sex)
print(f"The male-female proportion is { (100*prop_sex['female']).__round__(2)} : { (100*prop_sex['male']).__round__(2)} " )

#%%
prop_survived = titanic.survived.value_counts(normalize=True)  #  titanic.groupby('sex').agg({'sex':'count'}) # ['sex'].apply(lambda x: 100 * x / float(x.sum()))
print(type(prop_survived))
print(prop_survived)
print(f"The survived-dead proportion is { (100*prop_survived[1]).__round__(2)} : { (100*prop_survived[0]).__round__(2)} " )

#%%
# ### c. pie chart for “Ticketclass”  
prop_pclass = titanic.pclass.value_counts(normalize=True)
prop_pclass.plot.pie(y='pclass',  autopct="%.1f%%")
plt.title('Ticket Class')
plt.show()

#%%
# ### d. A single visualization chart that shows info of survival, age, pclass, and sex.  

f, axes = plt.subplots(2, 1, sharex=True,figsize=(8, 10) )
sns.despine(left=True)
sns.violinplot(x='pclass', y="age", hue="sex", data = titanic, split=True, ax=axes[0])
axes[0].set_title('Violinplot of age vs sex/pclass')
sns.violinplot(x='pclass', y="age", hue="survived", data = titanic, palette=['pink','magenta'], split=True, ax=axes[1])
axes[1].set_title('Violinplot of age vs survived/pclass')
f.suptitle("Visualization of 'Survived', 'Age','Pclass','Sex'")
plt.show()



#%%
# ## Question 2  
# Build a logistic regression model for survival using the statsmodels library. As we did before, include the features that you find plausible. Make sure categorical variables are use properly. If the coefficient(s) turns out insignificant, drop it and re-build.  

from statsmodels.formula.api import glm
# model = glm(formula, data, family)
import statsmodels.api as sm  # Importing statsmodels

# 1. Instantiate the model → glm()
# 2. Fit the model → .fit()
# 3. Summarize the model → .summary()
# 4. Make model predictions → .predict()

survival_lr = glm(formula='survived ~ age+C(pclass)+C(sex)+C(sibsp)+C(parch)', data=titanic, family=sm.families.Binomial())
survival_lr_fit = survival_lr.fit()
print( survival_lr_fit.summary() )

#%%
# The model turns out have very high p-value for sibsp and parch for almost all levels. I am going to remove them for simpler model.
# Technically, I can single out the few that has low p-values, but I feel those are just anormaly rather than anything substantial.
# 
# Rebuild: 
survival_lr = glm(formula='survived ~ age+C(pclass)+C(sex)', data=titanic, family=sm.families.Binomial())
survival_lr_fit = survival_lr.fit()
print( survival_lr_fit.summary() )


#%%
# ## Question 3  
# Interpret your result. What are the factors and how do they affect the chance of survival (or the survival odds ratio)? What is the predicted probability of survival for a 30-year-old female with a second class ticket, no siblings, 3 parents/children on the trip? Use whatever variables that are relevant in your model.  

import numpy as np
print(np.exp(survival_lr_fit.params))
print(np.exp(survival_lr_fit.conf_int()))
# According to the model, the chance of survival p is given by 
# Logit(p) = 2.8255 + (-0.0161 * age) -2.6291 (if male) - 0.9289 (if pclass 2) - 2.1722 (if plcass 3)
# OR
# p/(1-p) = 16.87 * 0.0.984062^age *  0.072142 (if male) * 0.394998 (if pclass 2) * 0.113926 (if pclass 3)
#

#%%
# survival_lr_fit.predict(titanic[0:5])
samplepredict = survival_lr_fit.predict( {'age':30, 'pclass':2, 'sex':'female'}) # You can either put in a dataframe or dictionary with all the relevant values here to make a prediction.
print(f'The model prediction of the survival probabilty is {(samplepredict[0]*100).__round__(1)}%')

#%%
# ## Question 4  
# Try three different cut-off values at 0.3, 0.5, and 0.7. What are the a) Total accuracy of the model b) The precision of the model (average for 0 and 1), and c) the recall rate of the model (average for 0 and 1)
# Confusion matrix
# Define cut-off value
cut_offs = [0.3, 0.5, 0.7]
# Compute survival predictions
from sklearn.metrics import classification_report
for cut_off in cut_offs:
    print(f"For cutoff = {cut_off}")
    y_true, y_pred = titanic.survived, np.where(survival_lr_fit.predict() > cut_off, 1, 0) 
    print(classification_report(y_true, y_pred))





#%%[markdown]
# # Part II  
nfl = dm.api_dsLand('nfl2008_fga')
nfl.dropna(inplace=True)

#%%
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
# 
# Use corrplot to get an idea of significant features
import seaborn as sns
import matplotlib.pyplot as plt
X = nfl.iloc[:,0:11]  #independent columns
y = nfl.iloc[:,-3]    #target column 
#get correlations of each features in dataset
corrmat = nfl.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(23,23))
#plot heat map
g=sns.heatmap(nfl[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()


# %%
sns.histplot(data=nfl, x='qtr', hue='GOOD', multiple='stack', bins=np.linspace(0.5,5.5,6) ).set(title='Field Goal Success by Quarter')
plt.show()
# %%
sns.violinplot(x='qtr', y='distance', hue="GOOD", split=True, data=nfl).set(title='Kick Distance vs qtr/success')
plt.show()

#%%
# ## Question 6  
# Using the SciKitLearn library, build a logistic regression model overall (not individual team or kicker) to predict the chances of a successful field goal. What variables do you have in your model? 
# 
from sklearn.linear_model import LogisticRegression

# Variables chosen are distance, down, qtr, homekick, and togo.
xnfl = nfl[['distance','down','qtr','homekick','togo']]
ynfl = nfl['GOOD']

nfllogit = LogisticRegression()  # instantiate
nfllogit.fit(xnfl, ynfl)

print(f'Logit model accuracy (with the test set): { (100*nfllogit.score(xnfl, ynfl) ).__round__(2)}%')


#%%
print("The predicted chance of the field goal at 40 years, 2nd-down, 2nd-quarter, away team, with 5 yards to go: ")
print(nfllogit.predict_proba( pd.DataFrame({'distance':[40], 'down':[2], 'qtr':[2], 'homekick':[0], 'togo':[5] } )))
print("\nNow compared to the same data point except by the home team, the predicted chance of the field goal: ")
print(nfllogit.predict_proba( pd.DataFrame({'distance':[40], 'down':[2], 'qtr':[2], 'homekick':[1], 'togo':[5] } )))

print("\nReady to continue.")

#%%
print(nfllogit.predict_proba(xnfl[:8]))


# ## Question 7  
# Someone has a feeling that home teams are more relaxed and have a friendly crowd, they should kick better field goals. Can you build two different models, one for all home teams, and one for road teams, of their chances of making a successful field goal?
# 
# Our previous calculation shows that according to our model, hometeam is actually less likely to make a field goal! For that particular distance anyway. 
# If we can actually see the coefficients, we can then actually see what is the coefficient for homekick, and what p-value it gets. 
# Let us also quickly check the cross table / contingency table:
homeTeams = nfl[nfl['homekick']==1]
awayTeams = nfl[nfl['homekick']==0]
pd.crosstab(nfl.homekick, nfl.GOOD)

# Indeed, it seems hometeam is actually at a disadvantage!!
# I now realize there is no need to build two different models for home and away teams. Our model already has the variable 'homekick' to study the effect. 
# There is really no need to build the models separately.


#%%
# ## Question 8    
# From what you found, do home teams and road teams have different chances of making a successful field goal? If one does, is that true for all distances, or only with a certain range?
# 
# Without seeing the coefficients overall, we can try more distances.
for dist in [5,10,15,20,25,30,35,40,45,50,55]:
    awaydata={'distance':[dist], 'down':[2], 'qtr':[2], 'homekick':[0], 'togo':[5] }
    homedata={'distance':[dist], 'down':[2], 'qtr':[2], 'homekick':[1], 'togo':[5] }
    print(f"At distance {dist}-yard, \nthe away team's chance is { nfllogit.predict_proba( pd.DataFrame( awaydata ) ) }, and ")
    print(f"the home team's chance is { nfllogit.predict_proba( pd.DataFrame( homedata ) ) }. ")

# It seems the home team's chance is indeed worse than the away team from this model at all distances.
# %%
