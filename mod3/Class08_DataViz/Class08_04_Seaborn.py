# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
# %pip install seaborn

#%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set(style="ticks")

#%%
# #############################
# The famous Anscombe quadrants
# #############################
df = sns.load_dataset("anscombe")
print(df.head())
print(df.tail())
print(df.info())
print(df.shape)

# Back-story: In 1973, English Statistician Francis Anscombe created four 
# datasets (Anscombe quadrants) that share the same the same mean, variance, 
# correlation, regression line, and coefficient of determination. Looking at 
# these numerical statistics alone, one might conclude the four sets are 
# basically the same. But the plots tell a very different story...

#%%
sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df, col_wrap=2, ci=None, palette="muted", scatter_kws={"s": 50, "alpha": 1})

plt.show()

# Try different options for conf int = 95 / None, "s": 5, "alpha":0.1

print("\nReady to continue.")

#%%
sns.set(style="white", palette="muted", color_codes=True)
rs = np.random.RandomState(10)

f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)

d = rs.normal(size=100)

sns.distplot(d, kde=False, color="b", ax=axes[0, 0])

sns.distplot(d, hist=False, rug=True, color="r", ax=axes[0, 1])

sns.distplot(d, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])

sns.distplot(d, color="m", ax=axes[1, 1])
plt.setp(axes, yticks=[])

plt.tight_layout()
plt.show()

print("\nReady to continue.")

#%%
# #############################
# Heatmap in sns
# #############################
sns.set()

flights_long = sns.load_dataset("flights")
print(flights_long.head())
print(flights_long.tail())
print(flights_long.info())
print(flights_long.shape)

#%%
flights = flights_long.pivot("month", "year", "passengers")
print(flights.head())
print(flights.tail())
print(flights.info())
print(flights.shape)

#%%
f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)

plt.show()

print("\nReady to continue.")

#%%
# #############################
# Violinplot in sns
# #############################

sns.set(style="whitegrid", palette="pastel", color_codes=True)

tips = sns.load_dataset("tips")
print(tips.head())
print(tips.tail())
print(tips.info())
print(tips.shape)

#%%
sns.violinplot(x="day", y="total_bill", hue="smoker", split=True, inner="quart", palette={"Yes": "y", "No": "b"}, data=tips)

sns.despine(left=True)
plt.show()

print("\nReady to continue.")

#%%
# #############################
# Classic Titanic dataset
# #############################
sns.set(style="darkgrid")

df = sns.load_dataset("titanic")
print(df.head())
print(df.tail())
print(df.info())
print(df.shape)

#%%
pal = dict(male="#6495ED", female="#F08080")  # dictionary for color palette in RGB hex code

g = sns.lmplot(x="age", y="survived", col="sex", hue="sex", data=df, palette=pal, y_jitter=.02, logistic=True)

g.set(xlim=(0, 80), ylim=(-.05, 1.05))

plt.show()

print("\nReady to continue.")

#%%
# #############################
# Swarmplot using the 
# Classic Iris dataset 
# #############################

import pandas as pd
sns.set(style="whitegrid", palette="muted")

iris = sns.load_dataset("iris")
print(iris.head())
print(iris.tail())
print(iris.info())
print(iris.shape)

#%%
iris = pd.melt(iris, "species", var_name="measurement")
print(iris.head())
print(iris.tail())
print(iris.info())
print(iris.shape)

#%%
sns.swarmplot(x="measurement", y="value", hue="species", palette=["r", "c", "y"], data=iris)
plt.show()

print("\nReady to continue.")

#%%
# #############################
# pairplot using the 
# Classic Iris dataset 
# #############################

sns.set(style="ticks")

df = sns.load_dataset("iris")
sns.pairplot(df, hue="species")
plt.show()

print("\nReady to continue.")


#%%
# #############################
# residplot 
# #############################

sns.set(style="whitegrid")

rs = np.random.RandomState(7)
x = rs.normal(2, 1, 75)
y = 2 + 1.5 * x + rs.normal(0, 2, 75)

sns.residplot(x, y, lowess=True, color="g")
plt.show()


print("\nReady to continue.")

#%%
#%%
# ########################################################
# joinplot in sns 
# scatterplot (different varieties) + kde/hist on two axes
# ########################################################
sns.set(style="white")

rs = np.random.RandomState(5)
mean = [0, 0]
cov = [(1, .5), (.5, 1)]
x1, x2 = rs.multivariate_normal(mean, cov, 500).T

x1 = pd.Series(x1, name="$X_1$")
x2 = pd.Series(x2, name="$X_2$")

g = sns.jointplot(x1, x2, kind="kde", height=7, space=0)
# g = sns.jointplot(x1, x2, kind="hex", height=7, space=0)
plt.show()

print("\nReady to continue.")

#%%
#%%
# #############################
# Boxplot in sns
# #############################
sns.set(style="whitegrid")

diamonds = sns.load_dataset("diamonds")
print(diamonds.head())
print(diamonds.tail())
print(diamonds.info())
print(diamonds.shape)

#%%
clarity_ranking = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

sns.boxplot(x="clarity", y="carat", color="b", order=clarity_ranking, data=diamonds)
plt.show()

print("\nReady to continue.")

#%%
