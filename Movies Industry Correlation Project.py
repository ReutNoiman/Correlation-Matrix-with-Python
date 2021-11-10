
# coding: utf-8

# In[2]:

# Import libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

plt.style.use('ggplot')

from matplotlib.pyplot import figure


get_ipython().magic('matplotlib inline')

matplotlib.rcParams['figure.figsize'] = (12,8) # Adjusts the configuration of the plot we will create

# read in the data

df = pd.read_csv(r'C:\Users\reut\PycharmProjects\MovieIndustryProject\movies.csv')


# In[3]:

#looking at the data 

df.head()


# In[4]:

#Lets see if there is any missing data 

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{}-{:.0}%'.format(col,pct_missing))


# In[41]:

# Data types for our columns

df.info()


# In[46]:

# displayng values as int

df['budget'].fillna(0).astype('int64')

df['gross'].fillna(0).astype('int64')

df['runtime'].fillna(0).astype('int64')

df['votes'].fillna(0.0).astype('float')


# In[26]:

# spliting the released column to 4 new columns to make it more usefull

Newdf = df['released'].str.split(' ',3,expand = True)

Newdf.rename(columns={0:'Month',1:'Day',2:'Year',3:'Country'},inplace = True)

new=Newdf['Day'].str.split(',',1,expand=True)

new.rename(columns={0:'Day',1:'drop'},inplace=True)

df['correctYear'] = Newdf['Year']

df['correctDay']=new['Day']

df['correctMonth'] = Newdf['Month']

df['correctCountry'] = Newdf['Country']

df.head()


# In[36]:

# ordering the data by gross revenue desc

df.sort_values(by=['gross'],inplace=False,ascending=False)


# In[39]:

# drop duplicats

df['company'].drop_duplicates().sort_values(ascending=False)


# In[ ]:

# hypothesis -

# The budget has high positive corralation with the gross

# The votes has high positive corralation with the budget

# The score has high positive corralation with the budget


# In[42]:

# scatter plot - budget vs. gross revenue

plt.scatter(x=df['budget'], y=df['gross'])

plt.title('Budget vs Gross Revenue')

plt.xlabel('Gross Revenue')

plt.ylabel('Budget for Film')

plt.show()


# In[49]:

# scatter plot - votes vs. budget

plt.scatter(x=df['budget'],y=df['votes'])

plt.title('Votes vs Budget')

plt.xlabel('Budget')

plt.ylabel('Votes')

plt.show()


# In[51]:

# scatter plot - score vs. budget

plt.scatter(x=df['budget'],y=df['score'])
plt.title('Budget vs Score')
plt.xlabel('Budget')
plt.ylabel('Score')
plt.show()


# In[63]:

# plot chart using seaborn for regression line - budget vs gross

sns.regplot(x='budget',y='gross',data = df, scatter_kws={'color':'black'},line_kws={'color':'purple'}).set(title='Budget vs Gross Regression Line')


# In[67]:

# plot chart using seaborn for regression line - budget vs votes

sns.regplot(x='budget',y='votes',data=df,scatter_kws={'color':'grey'},line_kws={'color':'purple'},marker='+').set(title='Budget vs Votes Regression Line')


# In[66]:

# Budget vs score Regression Line

sns.regplot(x='budget',y='score',data=df,marker='*',scatter_kws={'color':'blue'},line_kws={'color':'silver'}).set(title='Budget vs Score Linear Regression')


# In[69]:

# Correlation between Variables in the data

correlation_matrix = df.corr(method = 'pearson')

sns.heatmap(correlation_matrix,annot=True)

plt.title('Correlation Matrix for Numeric Features')

plt.xlabel('Movie Features')

plt.ylabel('Movie Features')

plt.show()


# In[70]:

# Correlation matrix including a numeric representation for non numeric variables

df_num = df

for col_name in df_num.columns:
    if(df_num[col_name].dtype == 'object'):
        df_num[col_name]=df_num[col_name].astype('category')
        df_num[col_name] = df_num[col_name].cat.codes

df_num


# In[71]:

# Correlation matrix for all values (pearson method)

correlation_matrix = df_num.corr(method = 'pearson')

sns.heatmap(correlation_matrix,annot = True)

plt.title('Correlation Matrix for Numeric Features')

plt.xlabel('Movie Features')

plt.ylabel('Movie Features')

plt.show()


# In[72]:

# Correlation matrix for all numerized features - using unstacking for better understanding and view (without sorting)

corr_mat = df_num.corr(method = 'pearson')

corr_pairs = corr_mat.unstack()

corr_pairs


# In[90]:

# Correlation between features - using unstack and sort for usability

sorted_pairs = corr_pairs.sort_values(ascending = False)

sorted_pairs


# In[92]:

# Focusing in high correlation (over 0.5) for usabilty 

# Corelation matrix without duplicates

high_corr = sorted_pairs[((sorted_pairs)>0.5) & ((sorted_pairs)<1)]

high_corr = high_corr.drop_duplicates()

high_corr


# In[ ]:

# By analysing the correlation matrix, it looks as if the preliminary theories where correct


# In[93]:

# Focusing on negative correlation (below -0.1) for further inspection.

# Corelation matrix without duplicates

low_corr = sorted_pairs[(sorted_pairs)<-0.1]

low_corr = low_corr.drop_duplicates()

low_corr


# In[ ]:

# Seems that a large budget won't necessarily gaurantee a high rating

