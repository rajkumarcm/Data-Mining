# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import dm6103 as dm

world1 = dm.api_dsLand('World1', 'id')
world2 = dm.api_dsLand('World2', 'id')

print("\nReady to continue.")

#%% [markdown]
# # Two Worlds (Continuation from midterm: Part I - 25%)
# 
# In the (midterm) mini-project, we used statistical tests and visualization to 
# studied these two worlds. Now let us use the modeling techniques we now know
# to give it another try. 
# 
# Use appropriate models that we learned in this class or elsewhere, 
# elucidate what these two world looks like. 
# 
# Having an accurate model (or not) however does not tell us if the worlds are 
# utopia or not. Is it possible to connect these concepts together? (Try something called 
# "feature importance"?)
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

#%%
# Since you mentioned feature importance, I thought Decision Tree would be more appropriate
#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

tmp_world1 = world1.copy()
tmp_world2 = world2.copy()

tmp_world = pd.concat([tmp_world1, tmp_world2], axis=0)
tmp_world['world'] = ['World1'] * tmp_world1.shape[0] + ['World2'] * tmp_world2.shape[0]
del tmp_world1
del tmp_world2

X_train, X_test, y_train, y_test = train_test_split(tmp_world.loc[:, tmp_world.columns != 'world'],
                                                    tmp_world.world, test_size=0.2, random_state=1234)

tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.1, random_state=1234)
tree.fit(X_train, y_train) # max_depth=6 gives better score

# evaluation
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

cm = confusion_matrix(y_test, y_test_pred)
importance = pd.DataFrame({'variable':X_train.columns, 'importance':tree.feature_importances_})
plt.figure()
sns.barplot(data=importance, x='variable', y='importance')
plt.title('Bias is very clear')
plt.show()

from sklearn.tree import plot_tree, export_graphviz
fig = plt.figure(figsize=(25,20))
_ = plot_tree(tree,
               feature_names=tree.feature_names_in_,
               class_names=tree.classes_,
               filled=True)
plt.show()

#%%[markdown]
# The bias is very clear from the first bar plot. The Decision tree uses both gender and ethnicity to classify between the two worlds that says these are the two features that distinguishes them.  
# From the tree plot, I can see:  
# 1. At the first level i.e. level after the root node, most of the data that went to gender criter were classified as World1.
# 2. If you notice carefully, at the first level i.e., level after the root node, most of the data that went through ethnic and gender in the following level were classified as World1. Whereas in case of World2, they did not go through the gender criteria instead a terminal node was created at the second level itself classifying observations as World2.  
#
# I do understand there are cases when both World1 and World2 do pass through both gender and ethnic criterias, but talking from the perspective of "majority" of cases, I clearly see gender criteria always classifies World1 predominantly than World2.  

#%% [markdown]
#
# # Free Worlds (Continuation from midterm: Part II - 25%)
# 
# To-do: Complete the method/function predictFinalIncome towards the end of this Part II codes.  
#  
# The worlds are gifted with freedom. Sort of.  
# I have a model built for them. It predicts their MONTHLY income/earning growth, 
# base on the characteristics of the individual. You task is to first examine and 
# understand the model. If you don't like it, build you own world and own model. 
# For now, please help me finish the last piece.  
# 
# My model will predict what is the growth factor for each person in the immediate month ahead. 
# Along the same line, it also calculate what is the expected (average) salary after 1 month with 
# that growth rate. You need to help make it complete, by producing a method/function that will 
# calculate what is the salary after n months. (Method: predictFinalIncome )  
# 
# That's all. Then try this model on people like Plato, and also create some of your favorite 
# people with all sort of different demographics, and see what their growth rates / growth factors 
# are in my worlds. Use the sample codes after the class definition below.  
# 
#%%
class Person:
  """ 
  a person with properties in the utopia 
  """

  def __init__(self, personinfo):
    self.age00 = personinfo['age'] # age at creation or record. Do not change.
    self.age = personinfo['age'] # age at current time. 
    self.income00 = personinfo['income'] # income at creation or record. Do not change.
    self.income = personinfo['income'] # income at current time.
    self.education = personinfo['education']
    self.gender = personinfo['gender']
    self.marital = personinfo['marital']
    self.ethnic = personinfo['ethnic']
    self.industry = personinfo['industry']
    # self.update({'age00': self.age00, 
    #         'age': self.age,
    #         'education': self.education,
    #         'gender': self.gender,
    #         'ethnic': self.ethnic,
    #         'marital': self.marital,
    #         'industry': self.industry,
    #         'income00': self.income00,
    #         'income': self.income})
    return
  
  def update(self, updateinfo):
    for key,val in updateinfo.items():
      if key in self.__dict__ : 
        self.__dict__[key] = val
    return
        
  def __getitem__(self, item):  # this will allow both person.gender or person["gender"] to access the data
    return self.__dict__[item]

  
#%%  
class myModel:
  """
  The earning growth model for individuals in the utopia. 
  This is a simplified version of what a model could look like, at least on how to calculate predicted values.
  """

  # ######## CONSTRUCTOR  #########
  def __init__(self, bias) :
    """
    :param bias: we will use this potential bias to explore different scenarios to the functions of gender and ethnicity

    :param b_0: the intercept of the model. This is like the null model. Or the current average value. 

    :param b_age: (not really a param. it's more a function/method) if the model prediction of the target is linearly proportional to age, this would the constant coefficient. In general, this does not have to be a constant, and age does not even have to be numerical. So we will treat this b_age as a function to convert the value (numerical or not) of age into a final value to be combined with b_0 and the others 
    
    :param b_education: similar. 
    
    :param b_gender: similar
    
    :param b_marital: these categorical (coded into numeric) levels would have highly non-linear relationship, which we typically use seaparate constants to capture their effects. But they are all recorded in this one function b_martial
    
    :param b_ethnic: similar
    
    :param b_industry: similar
    
    :param b_income: similar. Does higher salary have higher income or lower income growth rate as lower salary earners?
    """

    self.bias = bias # bias is a dictionary with info to set bias on the gender function and the ethnic function

    # ##################################################
    # The inner workings of the model below:           #
    # ##################################################

    self.b_0 = 0.0023 # 0.23% MONTHLY grwoth rate as the baseline. We will add/subtract from here

    # Technically, this is the end of the constructor. Don't change the indent

  # The rest of the "coefficients" b_1, b_2, etc are now disguised as functions/methods
  def b_age(self, age): # a small negative effect on monthly growth rate before age 45, and slight positive after 45
    effect = -0.00035 if (age<40) else 0.00035 if (age>50) else 0.00007*(age-45)
    return effect

  def b_education(self, education): 
    effect = -0.0006 if (education < 8) else -0.00025 if (education <13) else 0.00018 if (education <17) else 0.00045 if (education < 20) else 0.0009
    return effect

  def b_gender(self, gender):
    effect = 0
    biasfactor = 1 if ( self.bias["gender"]==True or self.bias["gender"] > 0) else 0 if ( self.bias["gender"]==False or self.bias["gender"] ==0 ) else -1  # for bias, no-bias, and reverse bias
    effect = -0.00045 if (gender<1) else 0.00045  # This amount to about 1% difference annually
    return biasfactor * effect 

  def b_marital(self, marital): 
    effect = 0 # let's assume martial status does not affect income growth rate 
    return effect

  def b_ethnic(self, ethnic):
    effect = 0
    biasfactor = 1 if ( self.bias["ethnic"]==True or self.bias["ethnic"] > 0) else 0 if ( self.bias["ethnic"]==False or self.bias["ethnic"] ==0 ) else -1  # for bias, no-bias, and reverse bias
    effect = -0.0006 if (ethnic < 1) else -0.00027 if (ethnic < 2) else 0.00045 
    return biasfactor * effect

  def b_industry(self, industry):
    effect = 0 if (industry < 2) else 0.00018 if (industry <4) else 0.00045 if (industry <5) else 0.00027 if (industry < 6) else 0.00045 if (industry < 7) else 0.00055
    return effect

  def b_income(self, income):
    # This is the kicker! 
    # More disposable income allow people to invest (stocks, real estate, bitcoin). Average gives them 6-10% annual return. 
    # Let us be conservative, and give them 0.6% return annually on their total income. So say roughly 0.0005 each month.
    # You can turn off this effect and compare the difference if you like. Comment in-or-out the next two lines to do that. 
    # effect = 0
    effect = 0 if (income < 50000) else 0.0001 if (income <65000) else 0.00018 if (income <90000) else 0.00035 if (income < 120000) else 0.00045 
    # Notice that this is his/her income affecting his/her future income. It's exponential in natural. 
    return effect

    # ##################################################
    # end of black box / inner structure of the model  #
    # ##################################################

  # other methods/functions
  def predictGrowthFactor( self, person ): # this is the MONTHLY growth FACTOR
    factor = 1 + self.b_0 + self.b_age( person["age"] ) + self.b_education( person['education'] ) + \
             self.b_ethnic( person['ethnic'] ) + self.b_gender( person['gender'] ) + \
             self.b_income( person['income'] ) + self.b_industry( person['industry'] ) + self.b_marital( ['marital'] )
    # becareful that age00 and income00 are the values of the initial record of the dataset/dataframe. 
    # After some time, these two values might have changed. We should use the current values 
    # for age and income in these calculations.
    return factor

  def predictIncome( self, person ): # perdict the new income one MONTH later. (At least on average, each month the income grows.)
    return person['income']*self.predictGrowthFactor( person )

  def predictFinalIncome( self, n, person ): 
    # predict final income after n months from the initial record.
    # the right codes should be no longer than a few lines.
    # If possible, please also consider the fact that the person is getting older by the month. 
    # The variable age value keeps changing as we progress with the future prediction.
    #return # ??? need to return the income level after n months.
    for i in range(n):
      person.age += 1/12
      person.income = self.predictIncome(person)
    return person.age, person.income

  def helper_predictFinalIncome(self, m_evolved, n, person):
    if n < m_evolved:
      raise ValueError("months evolved cannot be higher than the number of months yet to evolve")
    m = n - m_evolved
    return self.predictFinalIncome(m, person)



print("\nReady to continue.")

#%%
# SAMPLE CODES to try out the model
utopModel = myModel( { "gender": False, "ethnic": False } ) # no bias Utopia model
biasModel = myModel( { "gender": True, "ethnic": True } ) # bias, flawed, real world model

print("\nReady to continue.")

#%%
# Now try the two models on some versions of different people. 
# See what kind of range you can get. Plato is here for you as an example.
# industry: 0-leisure n hospitality, 1-retail , 2- Education 17024, 3-Health, 4-construction,
#           5-manufacturing, 6-professional n business, 7-finance
# gender: 0-female, 1-male
# marital: 0-never, 1-married, 2-divorced, 3-widowed
# ethnic: 0, 1, 2 
# age: 30-60, although there is no hard limit what you put in here.
# income: no real limit here.

months = 12 # Try months = 1, 12, 60, 120, 360
# In the ideal world model with no bias
plato = Person( { "age": 58, "education": 20, "gender": 1, "marital": 0, "ethnic": 2, "industry": 7, "income": 100000 } )
print(f'utop: {utopModel.predictGrowthFactor(plato)}') # This is the current growth factor for plato
print(f'utop: {utopModel.predictIncome(plato)}') # This is the income after 1 month
# Do the following line when your new function predictFinalIncome is ready
print(f'utop: {utopModel.predictFinalIncome(months,plato)[1]}')
#
# If plato ever gets a raise, or get older, you can update the info with a dictionary:
# plato.update( { "age": 59, "education": 21, "marital": 1, "income": 130000 } )

# In the flawed world model with biases on gender and ethnicity 
aristotle = Person( { "age": 58, "education": 20, "gender": 1, "marital": 0, "ethnic": 2, "industry": 7, "income": 100000 } )
print(f'bias: {biasModel.predictGrowthFactor(aristotle)}') # This is the current growth factor for aristotle
print(f'bias: {biasModel.predictIncome(aristotle)}') # This is the income after 1 month
# Do the following line when your new function predictFinalIncome is ready
print(f'bias: {biasModel.predictFinalIncome(months,aristotle)[1]}')

print("\nReady to continue.")


#%% [markdown]
# # Evolution (Part III - 25%)
# 
# We want to let the 24k people in WORLD#2 to evolve, for 360 months. You can either loop them through, and 
# create a new income or incomeFinal variable in the dataframe to store the new income level after 30 years. '
# Or if you can figure out a way to do
# broadcasting the predict function on the entire dataframem that can work too. If you loop through them,
# 'you can also consider
# using Person class to instantiate the person and do the calcuations that way, then destroy it when done to
# 'save memory and resources.
# If the person has life changes, it's much easier to handle it that way, then just tranforming the dataframe directly.
# 
# We have just this one goal, to see what the world look like after 30 years, according to the two models
# (utopModel and biasModel).
# 
# Remember that in the midterm, world1 in terms of gender and ethnic groups, 
# there were not much bias. Now if we let the world to evolve under the 
# utopia model utopmodel, and the biased model biasmodel, what will the income distributions 
# look like after 30 years?
# 
# Answer this in terms of distribution of income only. I don't care about 
# other utopian measures in this question here. 
#%%
tmp_world2 = world2.copy()
tmp_world2.rename(columns={"age00":"age", "income00":"income"}, inplace=True) # We should only not change Person object's income00 or age00

world2.loc[:, ['age', 'incomeUtop']] = list(map(lambda x: utopModel.predictFinalIncome(360, Person(x[1])), tmp_world2.iterrows()))
world2.loc[:, ['age', 'incomeBias']] = list(map(lambda x: biasModel.predictFinalIncome(360, Person(x[1])), tmp_world2.iterrows()))
del tmp_world2

gender_income = world2.melt(id_vars=['gender'], value_vars=['incomeUtop', 'incomeBias'])
ethnic_income = world2.melt(id_vars=['ethnic'], value_vars=['incomeUtop', 'incomeBias'])

from matplotlib import pyplot as plt
import seaborn as sns
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=gender_income, x='variable', y='value', hue='gender', ax=axes[0])
sns.boxplot(data=ethnic_income, x='variable', y='value', hue='ethnic', ax=axes[1])
axes[0].set_title('Gender bias after evolving for 360 months')
axes[1].set_title('Ethnic bias after evolving for 360 months')
axes[0].set_xlabel('Model')
axes[1].set_xlabel('Model')
axes[0].set_ylabel('Income')
axes[1].set_ylabel('Income')
plt.show()

#%%[markdown]
# The bias is very visible from the plots. While utopia model does not show any bias in terms of income after evolving
# for 360 months, bias model clearly shows both ethnic and gender bias.

#%% 
# # Reverse Action (Part IV - 25%)
# 
# Now let us turn our attension to World 1, which you should have found in the midterm that 
# it is far from being fair from income perspective among gender and ethnic considerations. 
# 
# Let us now put in place some policy action to reverse course, and create a revser bias model:
revbiasModel = myModel( { "gender": -1, "ethnic": -1 } ) # revsered bias, to right what is wronged gradually.

# If we start off with Word 1 on this revbiasModel, is there a chance for the world to eventual become fair like
# World #2? If so, how long does it take, to be fair for the different genders? How long for the different ethnic groups?

# If the current model cannot get the job done, feel free to tweak the model with more aggressive intervention
# to change the growth rate percentages on gender and ethnicity to make it work.

#%%
tmp_world1 = world1.copy()
tmp_world1.rename(columns={"age00":"age", "income00":"income"}, inplace=True) # same as I did with tmp_world2
from scipy.stats import ttest_ind, f_oneway

best_d_gender = 0
best_d_ethnic = 0
start_m = 1
m_prev = 0
m_new = start_m
age_gender_hist = {}
for m in range(start_m, 600, 12):
    tmp_world1.loc[:, ['age', 'income']] = list(map(lambda x: revbiasModel.helper_predictFinalIncome(m_prev, m, Person(x[1])),
                                        tmp_world1.iterrows()))
    age_gender_hist[f'{m}'] = pd.DataFrame({'income':tmp_world1.income, 'gender':tmp_world1.gender,
                                            'ethnic':tmp_world1.ethnic})

    _, pvalue_gender = ttest_ind(tmp_world1.loc[tmp_world1.gender==1, 'income'],
                                 tmp_world1.loc[tmp_world1.gender==0, 'income'], equal_var=False)
    if pvalue_gender > 0.05:
        best_d_gender = m
    _, pvalue_ethnic = f_oneway(tmp_world1.loc[tmp_world1.ethnic==0, 'income'],
                              tmp_world1.loc[tmp_world1.ethnic==1, 'income'],
                              tmp_world1.loc[tmp_world1.ethnic==2, 'income'])
    if pvalue_ethnic > 0.05:
        best_d_ethnic = m
    
    if best_d_ethnic > 0 and best_d_gender > 0:
        break
    m_prev = m

#%%[markdown]
# Gender Bias
#%%
keys = list(age_gender_hist.keys())
for i in range(18, 26):
  key = keys[i]
  plt.figure()
  sns.boxplot(x=tmp_world1.gender, y=age_gender_hist[f'{key}'].income, showfliers=False)
  plt.title(f'key: {key}')
plt.show()

#%%[markdown]
# Ethnic Bias
#%%
for key in keys:
  plt.figure()
  sns.boxplot(x=tmp_world1.ethnic, y=age_gender_hist[f'{key}'].income, showfliers=False)
  plt.title(f'key: {key}')
plt.show()

#%%[markdown]
# As you could see from the plots, it takes approximately 301 months for the gender bias to reverse, while in case ethnic bias, it takes about 

#%%[markdown]
# The number of months it takes to reverse the gender bias would be 301, whereas in case of
# reversing ethnic bias it takes approximately 565 months.
#
# As you already mentioned, I was not able to bring all three ethnicities to the same level and I think it is difficult to engineer soft landing for everyone.
#
#%%
del tmp_world1


























