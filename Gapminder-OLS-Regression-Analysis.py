# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 22:16:35 2020

@author: user
"""

import numpy
import pandas
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi

data = pandas.read_csv('nesarc.csv', low_memory = False)

data['S2AQ5B'] = pandas.to_numeric(data['S2AQ5B'], errors = 'coerce')
data['S1Q2G'] = pandas.to_numeric(data['S1Q2G'], errors = 'coerce')
data['CONSUMER'] = pandas.to_numeric(data['CONSUMER'], errors = 'coerce')
data['S1Q2F'] = pandas.to_numeric(data['S1Q2F'], errors = 'coerce')

sub1 = data[(data['AGE']>=18)&(data['AGE']<=25)&(data['CONSUMER'] == 1)]

#setting missing data
sub1['S2AQ5B'] = sub1['S2AQ5B'].replace(99, numpy.nan)
sub1['S1Q2G'] = sub1['S1Q2G'].replace(9, numpy.nan)

#recoding number of days drank in the past one year
recode1 = {1: 365, 2: 300, 3: 156, 4: 104, 5: 52, 6: 24, 7: 12, 8: 9, 9: 5, 10: 2}
sub1['USFREQYR'] = sub1['S2AQ5B'].map(recode1)

sub1['USFREQYR'] = pandas.to_numeric(sub1['USFREQYR'], errors = 'coerce')

ct1 = sub1.groupby('USFREQYR').size()
print(ct1)

#using ols function for calculating F-statistic and associated p-value
model1 = smf.ols(formula = 'USFREQYR ~ C(S1Q2G)', data = sub1)
results1 = model1.fit()
print(results1.summary())

sub2 = sub1[['USFREQYR', 'S1Q2G']].dropna()

print('Means for usfreqyr by ever lived with step parents before 18 years of age')
m1 = sub2.groupby('S1Q2G').mean()
print(m1)

print('Standard deviation for usfreqyr by ever lived with step parents before 18 years of age')
std1 = sub2.groupby('S1Q2G').std()
print(std1)

#for categorical variable with more than 2 categories
sub3 = sub1[['USFREQYR', 'S1Q2F']].dropna()

model2 = smf.ols(formula = 'USFREQYR ~ C(S1Q2F)', data = sub3)
result2 = model2.fit()
print(result2.summary())

print('Means for usfreqyr by parent lived with after biological or adoptive parents stopped living together')
m2 = sub3.groupby('S1Q2F').mean()
print(m2)

print('Standard deviation for usfreqyr by parent lived after biological or adoptive parent stopped living together')
std2 = sub3.groupby('S1Q2F').std()
print(std2)

#conducting post hoc test as to determine whether all means are not equal or some means are not equal
mc1 = multi.MultiComparison(sub3['USFREQYR'], sub3['S1Q2F'])
res1 = mc1.tukeyhsd()
print(res1.summary())
