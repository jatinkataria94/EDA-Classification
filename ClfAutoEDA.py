# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 10:29:02 2020

@author: Jatin
"""

import timeit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")




def numericalCategoricalSplit(df):
    numerical_features=df.select_dtypes(exclude=['object']).columns
    categorical_features=df.select_dtypes(include=['object']).columns
    numerical_data=df[numerical_features]
    categorical_data=df[categorical_features]
    return(numerical_data,categorical_data)



def nullFind(df):
    null_numerical=pd.isnull(df).sum().sort_values(ascending=False)
    #null_numerical=null_numerical[null_numerical>=0]
    null_categorical=pd.isnull(df).sum().sort_values(ascending=False)
   # null_categorical=null_categorical[null_categorical>=0]
    return(null_numerical,null_categorical)



def removeNullRows(df,few_null_col_list):
    for col in few_null_col_list:
        df=df[df[col].notnull()]
    return(df)
    
 


def EDA(df,labels,target_variable_name,
        data_summary_figsize=(16,16),corr_matrix_figsize=(16,16),
        data_summary_figcol="Reds_r",corr_matrix_figcol='Blues',
        corr_matrix_annot=False,
        pairplt_col='all',pairplt=False,
        feature_division_figsize=(12,12)):
    
    start_time = timeit.default_timer()
    
    #for converting class labels into integer values
    if df[target_variable_name].dtype=='object':
        class_labels=df[target_variable_name].unique().tolist()
        class_labels=[x for x in class_labels if type(x)==str]
        class_labels=[x for x in class_labels if str(x) != 'nan']
      
        for i in range(len(class_labels)):
            df[target_variable_name][df[target_variable_name]==class_labels[i]]=i
            
            
    df_orig=df
    print('The data looks like this: \n',df_orig.head())
    print('\nThe shape of data is: ',df_orig.shape)
    
    #To check missing values
    print('\nThe missing values in data are: \n',pd.isnull(df_orig).sum().sort_values(ascending=False))
    sns.heatmap(pd.isnull(df_orig))
    plt.title("Missing Values Summary", fontsize=(15), color="red")
    
    
   

    print('\nThe summary of data is: \n',df_orig.describe())
    plt.figure(figsize=data_summary_figsize)
    sns.heatmap(df_orig.describe()[1:].transpose(), annot= True, fmt=".1f",
                linecolor="black", linewidths=0.3,cmap=data_summary_figcol)
    plt.title("Data Summary", fontsize=(15), color="red")
    
      
   

    
    print('\nSome useful data information: \n')
    print(df_orig.info())
    print('\nThe columns in data are: \n',df_orig.columns.values)
    
    
    
   
    null_cutoff=0.5

    numerical=numericalCategoricalSplit(df_orig)[0]
    categorical=numericalCategoricalSplit(df_orig)[1]
    null_numerical=nullFind(numerical)[0]
    null_categorical=nullFind(categorical)[1]
    null=pd.concat([null_numerical,null_categorical])
    null_df=pd.DataFrame({'Null_in_Data':null}).sort_values(by=['Null_in_Data'],ascending=False)
    null_df_many=(null_df.loc[(null_df.Null_in_Data>null_cutoff*len(df_orig))])
    null_df_few=(null_df.loc[(null_df.Null_in_Data!=0)&(null_df.Null_in_Data<null_cutoff*len(df_orig))])

    many_null_col_list=null_df_many.index
    few_null_col_list=null_df_few.index
    
    #remove many null columns
    df_orig.drop(many_null_col_list,axis=1,inplace=True)
    
    df_wo_null=(removeNullRows(df_orig,few_null_col_list))
    
    
    if df_wo_null[target_variable_name].dtype=='object':
        df_wo_null[target_variable_name] =df_wo_null[target_variable_name].astype(str).astype(int)
   
    
    df=df_wo_null[df_wo_null.select_dtypes(exclude=['object']).columns]
   
    
    #Check correlation matrix
    plt.figure(figsize=corr_matrix_figsize)
    sns.heatmap(df.corr(),cmap=corr_matrix_figcol,annot=corr_matrix_annot) 
    
    
    col = df.columns.values
    number_of_columns=len(col)
    number_of_rows = len(col)-1/number_of_columns
    
    
    #To check Outliers
    plt.figure(figsize=(number_of_columns,number_of_rows))
    
    for i in range(0,len(col)):
        #plt.subplot(number_of_rows + 1,number_of_columns,i+1)
        if number_of_columns%2==0:
            plt.subplot(number_of_columns/2,2,i+1)   
            sns.set_style('whitegrid')
            sns.boxplot(df[col[i]],color='green',orient='h')
            plt.tight_layout()
        else:
            plt.subplot((number_of_columns+1)/2,2,i+1)
            sns.set_style('whitegrid')
            sns.boxplot(df[col[i]],color='green',orient='h')
            plt.tight_layout()
    
    
    #To check distribution-Skewness
    for i in range(0,len(col)):
        fig,axis = plt.subplots(1, 2,figsize=(16, 5))
        sns.distplot(df_orig[col[i]],kde=True,ax=axis[0]) 
        axis[0].axvline(df_orig[col[i]].mean(),color = "k",linestyle="dashed",label="MEAN")
        axis[0].legend(loc="upper right")
        axis[0].set_title('distribution of {}. Skewness = {:.4f}'.format(col[i] ,df_orig[col[i]].skew()))
        
        sns.violinplot(x=target_variable_name, y=col[i], data=df_orig, ax=axis[1], inner='quartile')
        axis[1].set_title('violin of {}, split by target'.format(col[i]))
    
       
    
    #to construct pairplot
    if (pairplt==True) and (pairplt_col!='all'):
        sns.pairplot(data=df, vars=pairplt_col, hue=target_variable_name)
    elif (pairplt==True) and (pairplt_col=='all'):
        sns.pairplot(data=df, vars=df.columns.values, hue=target_variable_name)
   
    
    
    #Proportion of target variable in dataset   
    
    st=df[target_variable_name].value_counts().sort_index()
    print('\nThe target variable is divided into: \n',st) #how many belong to each class of target variable
    
    
    
    plt.figure(figsize=feature_division_figsize)
    plt.subplot(121)
    ax = sns.countplot(y = df_orig[target_variable_name],
                     
                       linewidth=1,
                       edgecolor="k"*2)
    for i,j in enumerate(st):
        ax.text(.7,i,j,weight = "bold",fontsize = 27)
    plt.title("Count for target variable in datset")
    
    
    plt.subplot(122)
    plt.pie(st,
            labels=labels,
            autopct="%.2f%%",wedgeprops={"linewidth":2,"edgecolor":"white"})
    my_circ = plt.Circle((0,0),.7,color = "white")
    plt.gca().add_artist(my_circ)
    plt.subplots_adjust(wspace = .2)
    plt.title("Proportion of target variable in dataset")
    
    
    print('\nThe numerical features are: \n',df_wo_null.select_dtypes(exclude=['object']).columns.tolist())
    print('\nThe categorical features are: \n',df_wo_null.select_dtypes(include=['object']).columns.tolist())
    
    #Proportion of categorical variables in dataset   
    if len(df_wo_null.select_dtypes(include=['object']).columns.tolist())>=1:
        for cat_feat in df_wo_null.select_dtypes(include=['object']).columns.tolist():
            
            ct=df_wo_null.select_dtypes(include=['object'])[cat_feat].value_counts().sort_values(ascending=False)
            print('\nThe categorical variable is divided into: \n',ct) #how many belong to each class of target variable
            
            
            if (ct.index.size)<50:
                plt.figure(figsize=feature_division_figsize)
                plt.subplot(121)
                ax = sns.countplot(y = df_wo_null.select_dtypes(include=['object'])[cat_feat],
                                  
                                   linewidth=1,
                                   edgecolor="k"*2)
                for i,j in enumerate(ct):
                    ax.text(.7,i,j,weight = "bold",fontsize = 27)
                plt.title("Count for categorical variable in datset")
                
                
                plt.subplot(122)
                plt.pie(ct,
                        labels=df_wo_null.select_dtypes(include=['object'])[cat_feat].unique().tolist(),
                        autopct="%.2f%%",wedgeprops={"linewidth":2,"edgecolor":"white"})
                my_circ = plt.Circle((0,0),.7,color = "white")
                plt.gca().add_artist(my_circ)
                plt.subplots_adjust(wspace = .2)
                plt.title("Proportion of categorical variable in dataset")
            else:
                print('\nThe categorical variable %s has too many divisions to plot \n'%cat_feat)
            continue
    elapsed = timeit.default_timer() - start_time
    print('\nExecution Time for EDA: %.2f minutes'%(elapsed/60))
    
    
    return df_wo_null,df_wo_null.select_dtypes(exclude=['object']).columns.tolist(),df_wo_null.select_dtypes(include=['object']).columns.tolist()

