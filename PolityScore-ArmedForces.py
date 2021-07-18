# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:18:43 2015

@author: jml
"""

import pandas
import numpy
import scipy.stats
import seaborn
import matplotlib.pyplot as plt

data = pandas.read_csv('gapminder.csv', low_memory=False)

data['armedforcesrate'] = pandas.to_numeric(data['armedforcesrate'], errors = 'coerce')
data['incomeperperson'] = pandas.to_numeric(data['incomeperperson'], errors = 'coerce')
data['polityscore'] = pandas.to_numeric(data['polityscore'], errors = 'coerce')
data['incomeperperson']=data['incomeperperson'].replace(' ', numpy.nan)

data_clean=data.dropna()

print (scipy.stats.pearsonr(data_clean['armedforcesrate'], data_clean['polityscore']))

scat = seaborn.regplot(x="polityscore", y="armedforcesrate", data=data_clean)
plt.xlabel('Polity Score')
plt.ylabel('Armed Forces Rate')
plt.title('Scatterplot for the Association Between Polity Score and Armed Forces Rate')
plt.show(seaborn)

def incomegrp (row):
   if row['incomeperperson'] <= 744.239:
      return 1
   elif row['incomeperperson'] <= 9425.326 :
      return 2
   elif row['incomeperperson'] > 9425.326:
      return 3
   
data_clean['incomegrp'] = data_clean.apply (lambda row: incomegrp (row),axis=1)

chk1 = data_clean['incomegrp'].value_counts(sort=False, dropna=False)
print(chk1)

sub1=data_clean[(data_clean['incomegrp']== 1)]
sub2=data_clean[(data_clean['incomegrp']== 2)]
sub3=data_clean[(data_clean['incomegrp']== 3)]

print ('association between armed forces rate and polity score for LOW income countries')
print (scipy.stats.pearsonr(sub1['armedforcesrate'], sub1['polityscore']))
print ('       ')
print ('association between armed forces rate and polity score for MIDDLE income countries')
print (scipy.stats.pearsonr(sub2['armedforcesrate'], sub2['polityscore']))
print ('       ')
print ('association between armed forces rate and polity score for HIGH income countries')
print (scipy.stats.pearsonr(sub3['armedforcesrate'], sub3['polityscore']))

scat1 = seaborn.regplot(x="polityscore", y="armedforcesrate", data=sub1)
plt.xlabel('Polity Score')
plt.ylabel('Armed Forces Rate')
plt.title('Scatterplot for the Association Between Polity Score and Armed Forces Rate for LOW income countries')
plt.show(seaborn)

scat2 = seaborn.regplot(x="polityscore", y="armedforcesrate", fit_reg=False, data=sub2)
plt.xlabel('Polity Score')
plt.ylabel('Armed Forces Rate')
plt.title('Scatterplot for the Association Between Polity Score and Armed Forces Rate for MIDDLE income countries')
plt.show(seaborn)

scat3 = seaborn.regplot(x="polityscore", y="armedforcesrate", data=sub3)
plt.xlabel('Polity Score')
plt.ylabel('Armed Forces Rate')
plt.title('Scatterplot for the Association Between Polity Score and Armed Forces Rate for HIGH income countries')
plt.show(seaborn)
