import pandas as pd
import numpy as np

def monthly_data(df,indicator_name):
    df = df
    df.rename(columns={'concat':'issue_d'}, inplace=True)
    df = df[['issue_d', indicator_name]]

    return(df)

def mkt_data_trim(df,name):

    df = df[['date', 'value', 'concat']]
    df.rename(columns={'concat':'issue_d'}, inplace=True)
    df.rename(columns={'value': name}, inplace=True)

    return df

def mkt_data_transform(df,name):
    '''function to calculate monthly mean, max and min for daily data and save down to df
    NB:NOT a rolling window'''

    groups = df.groupby('issue_d')
    df_temp = round(groups.mean(),2)
    df_temp.rename(columns={name: name + '_mean'}, inplace=True)
    dfinal = df_temp

    df_temp = groups.min()
    df_temp = df_temp.drop('date', 1)
    df_temp.rename(columns={name:name + '_min'}, inplace=True)
    dfinal = pd.merge(dfinal, df_temp,  on='issue_d')

    df_temp = groups.max()
    df_temp = df_temp.drop('date', 1)
    df_temp.rename(columns={name:name + '_max'}, inplace=True)
    dfinal = pd.merge(dfinal, df_temp, on='issue_d')
    newcol = str(name + '_spread')
    max = name + '_max'
    min = name + '_min'

    dfinal[newcol] = dfinal[max] - dfinal[min]
    return dfinal

# WoE and plotting Notebook 02

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def woe_discrete(df, discrete_variabe_name, good_bad_variable_df):
    '''Calculates the Weight of Explanation for a discrete variable within a dataframe'''

    df = pd.concat([df[discrete_variabe_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

# Plotting functions

def plot_by_woe(df_WoE, rotation_of_x_axis_labels = 0):
    '''A function that takes a dataframe and a label rotation number as inputs,
        returns plot of Weight of Evidence vs Categorical Variable, and
        a subplot of number of observations vs Categorical variable
        '''
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    y = df_WoE['WoE']
    y2 = df_WoE['n_obs']

    plt.figure(figsize=(18, 10))

    plt.subplot(211)
    plt.plot(x, y)
    plt.plot(x, y, marker = 'o', linestyle = '--', color = 'k')
    plt.ylabel('Weight of Evidence')
    plt.title(str('Weight of Evidence by ' + df_WoE.columns[0]))
    plt.xticks(rotation = rotation_of_x_axis_labels)

    plt.subplot(212)
    plt.bar(x, y2)
    plt.ylabel('number of observations')
    plt.xticks(rotation = rotation_of_x_axis_labels)
    plt.show()

def plot_by_woe_marker(df_WoE, xposition=[], rotation_of_x_axis_labels = 0):
    '''Plots WoE weight of evidence vs categorical variable.
        Can add a list of line markers using the xposition[] input'''
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    y = df_WoE['WoE']
    y2 = df_WoE['n_obs']

    plt.figure(figsize=(18, 6))
    plt.subplot(211)
    plt.plot(x, y)
    plt.plot(x, y, marker = 'o', linestyle = '--', color = 'k')
    plt.ylabel('Weight of Evidence')
    plt.title(str('Weight of Evidence by ' + df_WoE.columns[0]))
    plt.axhline(y=0.0, xmin=0.0, xmax=1.0, color='r')
    for xc in xposition:
        plt.axvline(x=xc, color='k', linestyle='--')
        plt.xticks(rotation = rotation_of_x_axis_labels)

    plt.subplot(212)
    plt.bar(x, y2)
    plt.ylabel('number of observations')
    plt.axhline(y=0.0, xmin=0.0, xmax=1.0, color='r')
    for xc in xposition:
        plt.axvline(x=xc, color='k', linestyle='--')
        plt.xticks(rotation = rotation_of_x_axis_labels)

    plt.show()

def plot_by_woe_marker_block(df_WoE, xposition=[], start=0, end=0, rotation_of_x_axis_labels = 0):
    '''Plots WoE weight of evidence vs categorical variable.
        Can add a list of line markers using the xposition[] input,
        can also add a colour blok background, with start and end
        parameter inputs'''
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    y = df_WoE['WoE']
    y2 = df_WoE['n_obs']

    plt.figure(figsize=(18, 6))
    plt.subplot(211)
    plt.plot(x, y, marker = 'o', linestyle = '--', color = 'k')
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel('Weight of Evidence')
    plt.title(str('Weight of Evidence by ' + df_WoE.columns[0]))
    plt.axhline(y=0.0, xmin=0.0, xmax=1.0, color='r')
    # blocks
    plt.axvspan(start,end, facecolor='yellow', alpha=0.5)
    for xc in xposition:
        plt.axvline(x=xc, color='k', linestyle='--')
        plt.xticks(rotation = rotation_of_x_axis_labels)

    plt.subplot(212)
    plt.bar(x, y2)
    plt.ylabel('number of observations')
    plt.axhline(y=0.0, xmin=0.0, xmax=1.0, color='r')
    plt.axvspan(start,end, facecolor='yellow', alpha=0.5)
    for xc in xposition:
        plt.axvline(x=xc, color='k', linestyle='--')
        plt.xticks(rotation = rotation_of_x_axis_labels)

    plt.show()

    
def woe_ordered_continuous(df, discrete_variabe_name, good_bad_variable_df):
    '''Takes 3 arguments: a dataframe, a string, and a dataframe.
        The function returns a dataframe with WoE by discrete variable, in order'''

    df = pd.concat([df[discrete_variabe_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

# Score to Approval Rate Functions # Notebook 03

def score_to_cutoff(cutoff, df):
    '''Fucntion to return the approval and rejection rates for a given credit score'''
    unique_cutoffs = df['Score'].unique()
    if cutoff in unique_cutoffs:
        cutoff_level = df[df['Score']==cutoff]
        print("Approval rate is:  "+ str(round(cutoff_level['Approval Rate'].min(),4)))
        print("Rejection rate is: " + str(round(cutoff_level['Rejection Rate'].max(),4)))
    else:
        print("Cutoff level " + str(cutoff) + " not in table.")
        nearest = unique_cutoffs[min(range(len(unique_cutoffs)), key = lambda i: abs(unique_cutoffs[i]-cutoff))]
        print(str(nearest) + " is the nearest cutoff: Results are")
        cutoff_level = df[df['Score']==nearest]
        print("Approval rate is:  "+ str(round(cutoff_level['Approval Rate'].min(),4)))
        print("Rejection rate is: " + str(round(cutoff_level['Rejection Rate'].max(),4)))

def approval_to_cutoff(approval, df):
    '''Function to return the pobabilty of being good for a given credit score'''
    unique_cutoffs = df['Approval Rate'].unique()
    approval = round(approval,2)
    if approval in unique_cutoffs:
        approval_level = df[df['Approval Rate']==approval]
        print("Credit Score is:  "+ str(round(cutoff_level['Approval Rate'].min(),4)))
        # print("Rejection rate is: " + str(round(cutoff_level['Rejection Rate'].max(),4)))
    else:
        print("Cutoff level " + str(approval) + " not in table.")
        nearest = unique_cutoffs[min(range(len(unique_cutoffs)), key = lambda i: abs(unique_cutoffs[i]-approval))]
        print(str(round(nearest,5)) + " is the nearest cutoff:")
        cutoff_level = df[df['Approval Rate']==nearest]
        print("Score is :  "+ str(round(cutoff_level['Score'].min(),1)))

# EAD & LGD sheet

from sklearn import linear_model
import scipy.stats as stat

class LogisticRegression_with_p_values:
    
    def __init__(self,*args,**kwargs):#,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)#,**args)

    def fit(self,X,y):
        self.model.fit(X,y)
        
        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores] ### two tailed test for p-values
        
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        #self.z_scores = z_scores
        self.p_values = p_values
        #self.sigma_estimates = sigma_estimates
        #self.F_ij = F_ij















