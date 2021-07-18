# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 17:44:55 2020

@author: user
"""

import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt
import scipy

data = pandas.read_csv('gapminder.csv', low_memory = False)

data['armedforcesrate'] = pandas.to_numeric(data['armedforcesrate'], errors = 'coerce')
data['internetuserate'] = pandas.to_numeric(data['internetuserate'], errors = 'coerce')
data['polityscore'] = pandas.to_numeric(data['polityscore'], errors = 'coerce')

print('Describing the Armed Forces Personnel Rate as a percentage of the total labour force for different Countries')
desc1 = data['armedforcesrate'].describe()
print(desc1)

print('Describing the Internet User Rate for different Countries')
desc2 = data['internetuserate'].describe()
print(desc2)

print('Describing the Polity Score for different Countries')
desc3 = data['polityscore'].describe()
print(desc3)

scat1 = seaborn.regplot(x = "polityscore", y = "armedforcesrate", data = data)
plt.xlabel('Polity Score for different Countries')
plt.ylabel('Armed Forces Personnel Rate as a percentage of the total labour force for different Countries')
plt.title('Relationship between the Armed forces personnel rate and the Polity score for various Countries')
plt.show(seaborn)

scat2 = seaborn.regplot(x = "internetuserate", y = "armedforcesrate", data = data)
plt.xlabel('Internet User Rate for different Countries')
plt.ylabel('Armed Forces Personnel Rate as a percentage of the total labour force for different Countries')
plt.title('Relationship between the Armed forces personnel rate and the Internet user rate for various Countries')
plt.show(seaborn)

data_clean = data.dropna()

print('Association between Armed Forces Personnel Rate and The Polity Score for different Countries')
print(scipy.stats.pearsonr(data_clean['polityscore'], data_clean['armedforcesrate']))

print('Association between the Armed Forces Personnel Rate and the Internet User Rate for different Countries')
print(scipy.stats.pearsonr(data_clean['internetuserate'], data_clean['armedforcesrate']))
