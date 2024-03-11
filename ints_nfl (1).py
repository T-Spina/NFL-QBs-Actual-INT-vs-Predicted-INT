#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nfl_data_py as nfl 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import urllib.request
import numpy as np
import seaborn as sns 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier


# In[2]:


#Play-by-Play Data (pbp)


# In[3]:


pbp = nfl.import_pbp_data([2019,2020,2021,2022,2023])


# In[4]:


pd.set_option('display.max_column',None)
display(pbp.head(2))


# In[5]:


# Getting NFL Logos


# In[6]:


logos = nfl.import_team_desc()
logos.head()


# In[7]:


logos = logos [['team_abbr','team_logo_espn']]


# In[8]:


logo_paths = []

team_abbr = []

if not os.path.exists('logos'):
    os.makedirs('logos')


# In[9]:


for team in range(len(logos)):
    urllib.request.urlretrieve(logos['team_logo_espn'][team], f"logos/{logos['team_abbr'][team]}.tif")
    logo_paths.append(f"logos/{logos['team_abbr'][team]}.tif")
    team_abbr.append(logos['team_abbr'][team])


# In[10]:


data = {'Team Abbr': team_abbr, 'Logo Path': logo_paths}
logo_df = pd.DataFrame(data)
logo_df.head()


# In[11]:


# Getting Dataset Ready for Modeling


# In[12]:


print(pbp.shape)
pbp_clean = pbp[(pbp['pass']== 1) & (pbp['play_type'] != 'no_play') & (pbp['play_type'] != 'run')]
print(pbp_clean.shape)


# In[13]:


pbp_clean.head()


# In[14]:


# EDA


# In[15]:


sns.countplot (x=pbp_clean["interception"])
plt.show()


# In[16]:


int = pbp_clean[(pbp_clean['interception']==1)]
sns.countplot(x=int['down'])
plt.show()


# In[17]:


int = pbp_clean[(pbp_clean['interception']==1)]
sns.countplot(x=int['pass_length'])
plt.show()


# In[18]:


int = pbp_clean[(pbp_clean['interception']==1)]
sns.countplot(x=int['qb_hit'])
plt.show()


# In[19]:


int = pbp_clean[(pbp_clean['interception']==1)]
sns.countplot(x=int['shotgun'])
plt.show()


# In[20]:


int = pbp_clean[(pbp_clean['interception']==1)]
sns.countplot(x=int['drive_game_clock_start'])
plt.show()


# In[21]:


int = pbp_clean[(pbp_clean['interception']==1)]
sns.countplot(x=int['defenders_in_box'])
plt.show()


# In[22]:


int = pbp_clean[(pbp_clean['interception']==1)]
sns.countplot(x=int['was_pressure'])
plt.show()


# In[23]:


int = pbp_clean[(pbp_clean['interception']==1)]
sns.countplot(x=int['route'])
plt.xticks(rotation=45)
plt.show()


# In[24]:


int = pbp_clean[(pbp_clean['interception']==1)]
sns.countplot(x=int['defense_man_zone_type'])
plt.show()


# In[25]:


int = pbp_clean[(pbp_clean['interception']==1)]
sns.countplot(x=int['defense_coverage_type'])
plt.xticks(rotation=45)
plt.show()


# In[26]:


# Feature Engineering


# In[27]:


#pbp_clean.was_pressure = pbp_clean.was_pressure.replace ({True: 1, False:0})
#pbp_clean.was_pressure = pbp_clean.was_pressure.replace ({False: 0, True: 1}, inplace=True)
#pbp_clean['was_pressure']=pbp_clean['was_pressure'].astype(bool).astype(int)
#pbp_clean.was_pressure


# In[28]:


pbp_clean['obvious_pass'] = np.where((pbp_clean['down'] == 3) & (pbp_clean['ydstogo'] >= 6), 1,0)

pbp_clean['zone_coverage'] = np.where((pbp_clean['defense_man_zone_type'] == 'ZONE_COVERAGE'),1,0)

pbp_clean['man_coverage'] = np.where((pbp_clean['defense_man_zone_type'] == 'MAN_COVERAGE'),1,0)

#pbp_clean['pressured'] = np.where((pbp_clean['time_to_throw'] <=2 ) & (pbp_clean['ngs_air_yards'] >= 5 ), 1,0)


# In[29]:


pre_df = pbp_clean[['game_id','play_id','season','name','posteam','down','yardline_100','game_seconds_remaining',
                    'qb_hit','shotgun','defenders_in_box','obvious_pass','zone_coverage','man_coverage',
                    'time_to_throw','interception']]
pre_df.isna().sum()


# In[30]:


df = pre_df.dropna()
df.isna().sum()


# In[31]:


df.head()


# In[32]:


# Logistic Regression Modeling


# In[33]:


df['down']= df['down'].astype('category')


# In[34]:


df_no_ids = df.drop(columns = ['game_id','play_id','season','name','posteam'])
df_no_ids = pd.get_dummies(df_no_ids, columns = ['down'])


# In[35]:


df_no_ids.columns


# In[36]:


df_no_ids.head()


# In[37]:


sss = StratifiedShuffleSplit(n_splits=1,test_size=0.25, random_state=47)
for train_index, test_index in sss.split(df_no_ids,df_no_ids['interception']):
    strat_train_set = df_no_ids.iloc[train_index]
    strat_test_set = df_no_ids.iloc[test_index]
    
X_train = strat_train_set.drop(columns = ['interception'])
Y_train = strat_train_set['interception']
X_test = strat_test_set.drop(columns = ['interception'])
Y_test = strat_test_set['interception']


# In[38]:


lr = LogisticRegression()
lr.fit(X_train,Y_train)

lr_pred = pd.DataFrame(lr.predict_proba(X_test), columns = ['no_interception','interception'])[['interception']]
lr_pred


# In[39]:


print('Brier Score: ', brier_score_loss(Y_test, lr_pred))


# In[40]:


# Random Forest Model


# In[41]:


rf = RandomForestClassifier()
rf.fit(X_train,Y_train)

rf_pred = pd.DataFrame(rf.predict_proba(X_test), columns = ['no_interception','interception'])[['interception']]
rf_pred


# In[42]:


print('Brier Score:', brier_score_loss(Y_test,rf_pred))


# In[43]:


# XGB Boost Model


# In[44]:


XGB = XGBClassifier(objective = 'binary:logistic', random_state=47)
XGB.fit(X_train,Y_train)

XGB_pred = pd.DataFrame(XGB.predict_proba(X_test), columns = ['no_interception','interception'])[['interception']]
print('Brier Score:', brier_score_loss(Y_test,XGB_pred))


# In[45]:


sorted_index = XGB.feature_importances_.argsort()
plt.barh(X_train.columns[sorted_index], XGB.feature_importances_[sorted_index])
plt.title("XGBClassifier Feature Importance")
plt.show()


# In[46]:


make_ints_preds = df_no_ids.drop('interception',axis=1)
XGB_total_predictions = pd.DataFrame(XGB.predict_proba(make_ints_preds), columns = ['no_interception','int_pred'])[['int_pred']]

int_preds = df.reset_index().drop(columns = ['index'])
int_preds['int_pred'] = XGB_total_predictions

int_preds.head()


# In[47]:


int_preds['int_oe'] = int_preds['interception'] - int_preds['int_pred']
qbs = int_preds[(int_preds['season']==2023)].groupby(['name','posteam']).agg({'interception':'sum','int_pred':'sum','int_oe':'sum'}).reset_index().sort_values('int_oe', ascending= True)
qbs = qbs.rename(columns={'posteam' : 'Team Abbr'})
qbs.head(10) # Top Ten QBs that avoid throwing interceptions 


# In[48]:


qbs.tail(10) # Bottom Ten QBs that avoid throwing interceptions 


# In[49]:


avg_int= qbs.loc[:,'interception'].mean()
print('Avergae INT:',avg_int)

avg_int_pred= qbs.loc[:,'int_pred'].mean()
print('Avergae INT Predicted:',avg_int_pred)

avg_int_oe= qbs.loc[:,'int_oe'].mean()
print('Avergae INT Over Expected:',avg_int_oe)


# In[50]:


avg_int_pred= qbs.loc[:,'int_pred'].mean()
avg_int_pred


# In[51]:


#Export file
#qbs.to_excel('qbs_int.xlsx', index=False)


# In[52]:


sns.boxplot(x=int_preds['interception'], y = int_preds['int_pred'])
plt.show()


# In[53]:


#Plot


# In[54]:


qbs_plot = pd.merge(qbs, logo_df)
qbs_plot


# In[55]:


def getImage(path):
    return OffsetImage(plt.imread(path, format="tif"), zoom=.1)


# In[56]:


# Define plot size and autolayout
plt.rcParams["figure.figsize"] = [20, 14]
plt.rcParams["figure.autolayout"] = True

# Define the x and y variables
x = qbs_plot['interception']
y = qbs_plot['int_pred']
types = qbs_plot.reset_index()['name'].values

# Image paths
paths = qbs_plot['Logo Path']

# Define the plot
fig, ax = plt.subplots()

#Display grid
plt.grid(True)

# Load the data into the plot
for x0, y0, path in zip(x, y, paths):
   ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
   ax.add_artist(ab)

#Annotnations
for i, txt in enumerate(types):
    ax.annotate(txt, (x[i], y[i]), xytext=(30,30), textcoords='offset points')
    
# Plot parameters
plt.xlim(0, 18);
plt.ylim(0, 18);
plt.title("2023 NFL Season - Actual_INTs vs. Predicted_INTs", fontdict={'fontsize':35});
plt.xlabel("INTs", fontdict={'fontsize':21});
plt.ylabel("Predicted INTs", fontdict={'fontsize':21});


# In[57]:


# Median for INT is higher than the play is predicted to not have an INT.
#Further improvements of this model could be to add features such as pressured throws and covergaes faced.
#Cont. further improvements can be to add games played or plays played limitations
#Accuracy calculations can be conducted to provide a more accuracte representation of QB qualirt in regards to INTs. 


# In[58]:


#References:
#https://github.com/tejseth/nfl-tutorials-2022/blob/master/nfl_data_py_2.ipynb - LR, RF, and XGB Modeling code
#https://github.com/tbryan2/NFL-Python-Team-Logo-Viz/blob/main/Team-Logo-Visualizations.ipynb - Visualization for Teams

