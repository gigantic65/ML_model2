import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sklearn.metrics as sm
from sklearn.feature_selection import RFE

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.multioutput import MultiOutputRegressor

import xgboost
import base64
import plotly.graph_objects as go
import pickle
import os.path

import os
from io import BytesIO

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################

# Model building
def feature_s(df3):
    
    x = df3.iloc[:,:-1] # Using all column except for the last column as X
    y = df3.iloc[:,-1] 
 
    r2_train = []
    r2_test = []
    
    for n_comp in range(x.shape[1]):
        n_comp += 1
        # define the method
        rfe = RFE(RandomForestRegressor(), n_features_to_select=n_comp)
        
        # fit the model
        rfe.fit(x, y)
        
        # transform the data
        X3 = rfe.transform(x)
        
        #   Column 값 넣기
        X_rfe = pd.DataFrame(X3)
 
        X_rfe_columns = []
        for i in range(x.shape[1]):
            if rfe.ranking_[i] == 1:
                X_rfe_columns.append(x.columns[i])
        
        X_train, X_test, y_train, y_test = train_test_split(X_rfe, y, test_size=0.2, random_state=7)
            
        scaler = StandardScaler().fit(X_train)
        rescaledX = scaler.transform(X_train)
        rescaledtestX = scaler.transform(X_test)

        model = RandomForestRegressor()
        model.fit(rescaledX, y_train)
        
        predictions = model.predict(rescaledX)
        predictions2 = model.predict(rescaledtestX)
        
        r_squared = sm.r2_score(y_train,predictions)
        r_squared2 = sm.r2_score(y_test,predictions2)
        
        r2_train.append(r_squared)
        r2_test.append(r_squared2)
        
    sns.set(rc={'figure.figsize':(10,5)})
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.write("**CTP 중요도 평가**")
    feat_importances = pd.Series(model.feature_importances_, index=X_rfe_columns)
    feat_importances = feat_importances.sort_values(ascending = True)
    feat_importances.plot(kind='barh')
    st.pyplot()
        
    NumX = r2_train.index(np.max(r2_train))+1
    NumX2 = r2_test.index(np.max(r2_test))+1
    st.write("")
    st.write("**최적 CTP 개수;추천 개수 :**", max(NumX,NumX2))
        
    sns.set(rc={'figure.figsize':(8,5)})
    plt.plot(range(1,x.shape[1]+1),r2_train, color="#0e4194", marker= "o",label='Train')
    plt.plot(range(1,x.shape[1]+1),r2_test, color="red", marker= "o",label='Test')
    plt.xlabel('Number of CTP(X variables)')
    plt.ylabel('Model Accuracy (R2)')
    plt.legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


def F_feature(df3,hobby):
    
    x = df3.iloc[:,:-1] # Using all column except for the last column as X
    y = df3.iloc[:,-1]

    X_rfe_columns = []
    rfe = RFE(RandomForestRegressor(), n_features_to_select=hobby)
        
        # fit the model
    rfe.fit(x, y)
        
    for i in range(x.shape[1]):
        if rfe.ranking_[i] == 1:
            X_rfe_columns.append(x.columns[i])

    return X_rfe_columns


def feature_m(df3,Selected_X,Selected_y):
    
    x = df3[Selected_X] # Using all column except for the last column as X
    y = df3[Selected_y] 

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
        
    rfe = MultiOutputRegressor(RandomForestRegressor())
    
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
      
    # fit the model
    rfe.fit(rescaledX, y_train)
        
    f_importance = pd.DataFrame(rfe.estimators_[0].feature_importances_,columns=['importance'],index=X_train.columns)
    
    f_importance = f_importance.sort_values(by='importance', ascending = True)
    
    sns.set(rc={'figure.figsize':(10,5)})
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.write("**CTP 중요도 평가**")
    
    f_importance.plot(kind='barh')
    
    st.pyplot()
    
    return f_importance
        

def F_feature_m(df3,hobby,Selected_X,Selected_y):
    
    x = df3[Selected_X] # Using all column except for the last column as X
    y = df3[Selected_y] 

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
        
    rfe = MultiOutputRegressor(RandomForestRegressor())
    
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
      
    # fit the model
    rfe.fit(rescaledX, y_train)
        
    f_importance = pd.DataFrame(rfe.estimators_[0].feature_importances_,columns=['importance'],index=X_train.columns)
    
    f_importance = f_importance.sort_values(by='importance', ascending = False)
    
    X_rfe_columns = []
        
    for i in range(hobby):
        X_rfe_columns.append(f_importance.index[i])


    return X_rfe_columns

      
def build_model(df3,Selected_ml):
    
    X = df3.iloc[:,:-1] # Using all column except for the last column as X
    Y = df3.iloc[:,-1] # Selecting the last column as Y

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    st.write("")
    st.markdown("**4.1. 모델 검증 방법 :**  _K-Fold Cross Validation_")
    st.write("")
    st.markdown("**4.2. 머신러닝 모델 비교 결과**")
    ml = pd.DataFrame(Selected_ml)
    
    i=0
    models = []
  
    for i in range(ml.shape[0]):
        if ml.iloc[i].values == 'Linear Regression':
            models.append(('Linear Regression', Pipeline([('Scaler', StandardScaler()),('Linear Regression',LinearRegression())])))
                                                         
        if ml.iloc[i].values == 'Lasso':
            models.append(('LASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())])))
        if ml.iloc[i].values == 'KNN':
            models.append(('KNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor())])))
            
        if ml.iloc[i].values == 'Decision_Tree':
            models.append(('Decision_Tree', Pipeline([('Scaler', StandardScaler()),('Decision_Tree',DecisionTreeRegressor())])))
            
        if ml.iloc[i].values == 'GBM':
            models.append(('GBM', Pipeline([('Scaler', StandardScaler()),('GBM',GradientBoostingRegressor(n_estimators=75))])))
        if ml.iloc[i].values == 'XGBOOST':
            models.append(('XGBOOST', Pipeline([('Scaler', StandardScaler()),('XGBOOST',xgboost.XGBRegressor(booster='gbtree',n_estimators= 100))])))
            
        if ml.iloc[i].values == 'AB':
            models.append(('AB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
            
        if ml.iloc[i].values == 'Extra Trees':
            models.append(('Extra Trees', Pipeline([('Scaler', StandardScaler()),('Extra Trees',ExtraTreesRegressor())])))
        if ml.iloc[i].values == 'RandomForest':
            models.append(('RandomForest', Pipeline([('Scaler', StandardScaler()),('RandomForest',RandomForestRegressor())])))
            
    results = []
    names = []

    msg = []
    mean = []
    max1 = []
    min1 = []
    std = []        

    for name, model in models:

        model = model
        
        kfold = KFold(n_splits=5, random_state=7, shuffle=True)
        cv_results = cross_val_score(model, X, Y, cv=kfold, scoring='r2')
        
        for i, element in enumerate(cv_results):
            if element <= 0.0:
                cv_results[i] = 0.0

        results.append(abs(cv_results))
        
        names.append(name)
        msg.append('%s' % (name))
        mean.append('%f' %  (cv_results.mean()))
        min1.append('%f' %  (cv_results.min()))
        max1.append('%f' %  (cv_results.max()))
        std.append('%f' % (cv_results.std()))
        
    F_result = pd.DataFrame(np.transpose(msg))
    F_result.columns = ['Machine_Learning_Model']
    F_result['R2_Mean'] = pd.DataFrame(np.transpose(mean))
    F_result['R2_Min'] = pd.DataFrame(np.transpose(min1))
    F_result['R2_Max'] = pd.DataFrame(np.transpose(max1))
    F_result['R2_Std'] = pd.DataFrame(np.transpose(std))
    
    #st.write(F_result)
    F2_result = F_result.sort_values(by='R2_Mean', ascending=False, inplace =False)
    F2_result = F2_result.reset_index(drop = True)

    st.markdown('*교차검증 샘플 분할 갯수 : 5,   검증 인자 : $R^2$*')
    st.write(F2_result)
    
    F2_result['R2_Mean'] = F2_result['R2_Mean'].astype('float')
    F2_result['R2_Min'] = F2_result['R2_Min'].astype('float')
    F2_result['R2_Max'] = F2_result['R2_Max'].astype('float')
    F2_result['R2_Std'] = F2_result['R2_Std'].astype('float')

    fig, axs = plt.subplots(ncols=2)
    g = sns.barplot(x="Machine_Learning_Model", y="R2_Mean", data=F2_result, ax=axs[0])
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.set(ylim=(0,1))
   
    g2 = sns.barplot(x="Machine_Learning_Model", y="R2_Std", data=F2_result, ax=axs[1])
    g2.set_xticklabels(g2.get_xticklabels(), rotation=90)
    g2.set(ylim=(0,1))
    
    st.pyplot(plt)
    
    return F2_result


def build_model_m(df3,Selected_ml,Selected_X,Selected_y):
    
    x = df3[Selected_X] # Using all column except for the last column as X
    y = df3[Selected_y] 
    
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    st.write("")
    st.markdown("**4.1. 모델 검증 방법 :**  _K-Fold Cross Validation_")
    st.write("")
    st.markdown("**4.2. 머신러닝 모델 비교 결과**")
    ml = pd.DataFrame(Selected_ml)

    i=0
    models = []
   
    for i in range(ml.shape[0]):
        if ml.iloc[i].values == 'Linear Regression':
            models.append(('Linear Regression', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('Linear Regression',LinearRegression())]))))
                                                         
        if ml.iloc[i].values == 'Lasso':
            models.append(('LASSO', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())]))))
            
        if ml.iloc[i].values == 'KNN':
            models.append(('KNN', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor())]))))
            
        if ml.iloc[i].values == 'Decision_Tree':
            models.append(('Decision_Tree', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('Decision_Tree',DecisionTreeRegressor())]))))
            
        if ml.iloc[i].values == 'GBM':
            models.append(('GBM', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()), ('GBM',GradientBoostingRegressor(n_estimators=75))]))))
            
        if ml.iloc[i].values == 'XGBOOST':
            models.append(('XGBOOST', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('XGBOOST',xgboost.XGBRegressor(n_estimators= 100))]))))
            
        if ml.iloc[i].values == 'AB':
            models.append(('AB', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostRegressor())]))))
            
        if ml.iloc[i].values == 'Extra Trees':
            models.append(('Extra Trees', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('Extra Trees',ExtraTreesRegressor())]))))

        if ml.iloc[i].values == 'RandomForest':
            models.append(('RandomForest', MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('RandomForest',RandomForestRegressor())])))) 
            
    results = []
    names = []

    msg = []
    mean = []
    max1 = []
    min1 = []
    std = []        

    for name, model in models:

        model = model
        
        kfold = KFold(n_splits=5, random_state=7, shuffle=True)
        cv_results = cross_val_score(model, x, y, cv=kfold, scoring='r2')
        
        for i, element in enumerate(cv_results):
            if element <= 0.0:
                cv_results[i] = 0.0

        results.append(abs(cv_results))
        
        names.append(name)
        msg.append('%s' % (name))
        mean.append('%f' %  (cv_results.mean()))
        min1.append('%f' %  (cv_results.min()))
        max1.append('%f' %  (cv_results.max()))
        std.append('%f' % (cv_results.std()))
        
    F_result = pd.DataFrame(np.transpose(msg))
    F_result.columns = ['Machine_Learning_Model']
    F_result['R2_Mean'] = pd.DataFrame(np.transpose(mean))
    F_result['R2_Min'] = pd.DataFrame(np.transpose(min1))
    F_result['R2_Max'] = pd.DataFrame(np.transpose(max1))
    F_result['R2_Std'] = pd.DataFrame(np.transpose(std))

    F2_result = F_result.sort_values(by='R2_Mean', ascending=False, inplace =False)
    F2_result = F2_result.reset_index(drop = True)


    st.markdown('*교차검증 샘플 분할 갯수 : 5,   검증 인자 : $R^2$*')
    st.write(F2_result)

    F2_result['R2_Mean'] = F2_result['R2_Mean'].astype('float')
    F2_result['R2_Min'] = F2_result['R2_Min'].astype('float')
    F2_result['R2_Max'] = F2_result['R2_Max'].astype('float')
    F2_result['R2_Std'] = F2_result['R2_Std'].astype('float')
    
    fig, axs = plt.subplots(ncols=2)
    g = sns.barplot(x="Machine_Learning_Model", y="R2_Mean", data=F2_result, ax=axs[0])
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.set(ylim=(0,1))
   
    g2 = sns.barplot(x="Machine_Learning_Model", y="R2_Std", data=F2_result, ax=axs[1])
    g2.set_xticklabels(g2.get_xticklabels(), rotation=90)
    g2.set(ylim=(0,1))

    st.pyplot(plt)
    
    return F2_result

    
def st_pandas_to_csv_download_link(_df, file_name:str = "dataframe.csv"): 
    csv_exp = _df.to_csv(index=False)
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}" > Download Dataset (CSV) </a>'
    st.markdown(href, unsafe_allow_html=True)
    

def st_pandas_to_csv_download_link2(_df, file_name:str = "dataframe.csv"): 
    csv_exp = _df.to_csv(index=False)
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}" > Download Dataset (CSV) </a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)


def Opti_model(Model,df3,parameter_n_estimators,parameter_max_features,param_grid):
    
    X = df3.iloc[:,:-1] # Using all column except for the last column as X
    Y = df3.iloc[:,-1] # Selecting the last column as Y

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    scaler = StandardScaler().fit(X)
    rescaled = scaler.transform(X)

    if Model == 'GBM':
        model = GradientBoostingRegressor(n_estimators=parameter_n_estimators, max_features=parameter_max_features)
    elif Model == 'Extra Trees':
        model = ExtraTreesRegressor(n_estimators=parameter_n_estimators, max_features=parameter_max_features)
    elif Model == 'RandomForest':
        model = RandomForestRegressor(n_estimators=parameter_n_estimators, max_features=parameter_max_features)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
               
    grid.fit(rescaled, Y)
        
    st.markdown("**· 최적 파라미터 : %s**" %(grid.best_params_))

            #-----Process grid data-----#
    grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])],axis=1)
    # Segment data into groups based on the 2 hyperparameters
    grid_contour = grid_results.groupby(['max_features','n_estimators']).mean()
    # Pivoting the data
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['max_features', 'n_estimators', 'R2']
    grid_pivot = grid_reset.pivot('max_features', 'n_estimators')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values
    
    #-----Plot-----#
    layout = go.Layout(
        xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                        text='n_estimators')
                ),
        yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                        text='max_features')
                ) )
    fig = go.Figure(data= [go.Surface(z=z, y=y, x=x)], layout=layout )
    fig.update_layout(title='하이퍼 파라미터 튜닝 결과',
        scene = dict(
            xaxis_title='n_estimators',
            yaxis_title='max_features',
            zaxis_title='R2'),
            autosize=False,
            width=800, height=800,
            margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)
     
    return grid.best_params_['n_estimators'],grid.best_params_['max_features']


def Opti_model_m(Model,df3,param_grid,Selected_X2,Selected_y):
    
    X = df3[Selected_X2] # Using all column except for the last column as X
    Y = df3[Selected_y] # Selecting the last column as Y

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    scaler = StandardScaler().fit(X)
    rescaled = scaler.transform(X)

    if Model == 'GBM':
        model = GradientBoostingRegressor()
    elif Model == 'Extra Trees':
        model = ExtraTreesRegressor()
    elif Model == 'RandomForest':
        model = RandomForestRegressor()

    grid = GridSearchCV(estimator=MultiOutputRegressor(model), param_grid=param_grid, cv=5)
        
    grid.fit(rescaled, Y)
    
    st.markdown("**· 최적 파라미터 : %s**" %(grid.best_params_))
        
    return grid.best_params_['estimator__n_estimators'],grid.best_params_['estimator__max_features']


def Opti_model2(Model,df3,parameter_n_estimators,parameter_max_depth,param_grid):
    
    X = df3.iloc[:,:-1] # Using all column except for the last column as X
    Y = df3.iloc[:,-1] # Selecting the last column as Y
    
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    scaler = StandardScaler().fit(X)

    model = xgboost.XGBRegressor(booster='gbtree',n_estimators=parameter_n_estimators, max_depth=parameter_max_depth)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    
    rescaled = scaler.transform(X)
        
    grid.fit(rescaled, Y)
                
    y_pred_test = grid.predict(rescaled)

    st.markdown("**· 최적 파라미터 : %s**" %(grid.best_params_))        

            #-----Process grid data-----#
    grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])],axis=1)
    # Segment data into groups based on the 2 hyperparameters
    grid_contour = grid_results.groupby(['n_estimators','max_depth']).mean()
    # Pivoting the data
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['n_estimators', 'max_depth', 'R2']
    grid_pivot = grid_reset.pivot('n_estimators', 'max_depth')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values
    
    #-----Plot-----#
    layout = go.Layout(
        xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                        text='max_depth')
                ),
        yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                        text='n_estimators')
                ) )
    fig = go.Figure(data= [go.Surface(z=z, y=y, x=x)], layout=layout )
    fig.update_layout(title='하이퍼 파라미터 튜닝 결과',
        scene = dict(
            xaxis_title='max_depth',
            yaxis_title='n_estimators',
            zaxis_title='R2'),
            autosize=False,
            width=800, height=800,
            margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)
    
    return grid.best_params_['n_estimators'],grid.best_params_['max_depth']


def Opti_model2_m(Model,df3,param_grid,Selected_X2,Selected_y):
    
    X = df3[Selected_X2] # Using all column except for the last column as X
    Y = df3[Selected_y] # Selecting the last column as Y

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    scaler = StandardScaler().fit(X)

    model = xgboost.XGBRegressor()

    grid = GridSearchCV(estimator=MultiOutputRegressor(model), param_grid=param_grid, cv=5)
    
    rescaled = scaler.transform(X)
        
    grid.fit(rescaled, Y)

    st.markdown("**· 최적 파라미터 : %s**" %(grid.best_params_))    
    
    return grid.best_params_['estimator__n_estimators'],grid.best_params_['estimator__max_depth']


def Opti_model3(Model,df3,parameter_n_estimators,parameter_learning_rate,param_grid):
    
    X = df3.iloc[:,:-1] # Using all column except for the last column as X
    Y = df3.iloc[:,-1] # Selecting the last column as Y
    
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    scaler = StandardScaler().fit(X)

    model = AdaBoostRegressor(n_estimators=parameter_n_estimators, learning_rate=parameter_learning_rate)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    
    rescaled = scaler.transform(X)
        
    grid.fit(rescaled, Y)
        
    y_pred_test = grid.predict(rescaled)

    st.markdown("**· 최적 파라미터 : %s**" %(grid.best_params_))        

            #-----Process grid data-----#
    grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])],axis=1)
    # Segment data into groups based on the 2 hyperparameters
    grid_contour = grid_results.groupby(['n_estimators','learning_rate']).mean()
    # Pivoting the data
    grid_reset = grid_contour.reset_index()
    grid_reset.columns = ['n_estimators', 'learning_rate', 'R2']
    grid_pivot = grid_reset.pivot('n_estimators', 'learning_rate')
    x = grid_pivot.columns.levels[1].values
    y = grid_pivot.index.values
    z = grid_pivot.values
    
    #-----Plot-----#
    layout = go.Layout(
        xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                        text='learning_rate')
                ),
        yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                        text='n_estimators')
                ) )
    fig = go.Figure(data= [go.Surface(z=z, y=y, x=x)], layout=layout )
    fig.update_layout(title='하이퍼 파라미터 튜닝 결과',
        scene = dict(
            xaxis_title='learning_rate',
            yaxis_title='n_estimators',
            zaxis_title='R2'),
            autosize=False,
            width=800, height=800,
            margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)
    
    return grid.best_params_['n_estimators'],grid.best_params_['learning_rate']


def Opti_model3_m(Model,df3,param_grid,Selected_X2,Selected_y):
    
    X = df3[Selected_X2] # Using all column except for the last column as X
    Y = df3[Selected_y] # Selecting the last column as Y

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    scaler = StandardScaler().fit(X)

    model = AdaBoostRegressor()

    grid = GridSearchCV(estimator=MultiOutputRegressor(model), param_grid=param_grid, cv=5)
    
    rescaled = scaler.transform(X)
        
    grid.fit(rescaled, Y)
                
    st.markdown("**· 최적 파라미터 : %s**" %(grid.best_params_))
    
    return grid.best_params_['estimator__n_estimators'],grid.best_params_['estimator__learning_rate']


def Opti_KNN_model(df3, parameter_n_neighbors_knn,parameter_n_neighbors_step_knn,param_grid_knn):

    X = df3.iloc[:,:-1] # Using all column except for the last column as X
    Y = df3.iloc[:,-1] # Selecting the last column as Y

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    scaler = StandardScaler().fit(X)
    rescaled = scaler.transform(X)

    model = KNeighborsRegressor(n_neighbors=parameter_n_neighbors_knn)
        
    grid = GridSearchCV(estimator=model, param_grid=param_grid_knn, cv=5)
            
    grid.fit(rescaled, Y)
    
    st.markdown("**· 최적 파라미터 : %s**" %(grid.best_params_))
    
    return grid.best_params_['n_neighbors']


def Opti_KNN_model_m(df3, param_grid,Selected_X2,Selected_y):

    X = df3[Selected_X2] # Using all column except for the last column as X
    Y = df3[Selected_y] # Selecting the last column as Y

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    scaler = StandardScaler().fit(X)
    rescaled = scaler.transform(X)

    model = KNeighborsRegressor()
        
    grid = GridSearchCV(estimator=MultiOutputRegressor(model), param_grid=param_grid, cv=5)
            
    grid.fit(rescaled, Y)
    
    st.markdown("**· 최적 파라미터 : %s**" %(grid.best_params_))            

    return grid.best_params_['estimator__n_neighbors']


def download_model(k, model):
    if k==0:
        output_model = pickle.dumps(model)
        b64 = base64.b64encode(output_model).decode()
        href = f'<a href="data:file/output_model;base64,{b64}" download="02_Trained_model.pkl">Download Trained Model.pkl</a>'
        st.markdown(href, unsafe_allow_html=True)
    elif k==1:
        model.save('test.h5')
        st.write('Currently, Neural network model save is underway.')
        st.write('If you want to make Neural network model, please contact Simulation team.')

#Excel로 변환 후 다운로드 함수, csv는 sheet 한장밖에 사용 못함. 고로 excel로 변환 적용.
def to_excel(df1, df2):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df1.to_excel(writer, sheet_name="sheet1", index = False)
    df2.to_excel(writer, sheet_name="sheet2", index = False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def download_data_xlsx(df1, df2):
    val = to_excel(df1, df2)
    b64 = base64.b64encode(val)
    st.markdown(f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="01_Data_for_Training.xlsx">Download Training Data.xlsx</a>', unsafe_allow_html=True)
    


#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################
#########################################################################################################################################################

def app(session_in):
    
    
    with st.expander("Stage1. 머신러닝 모델 생성하기 - 가이드"):
        st.markdown("0. 공정인자(CTP), 품질인자(CTQ)를 각 column에 나열해서 csv 파일을 준비합니다.")
        st.markdown("1. 데이터 가공 : 데이터(.csv) 업로드 후 데이터 가공 절차(Step1~4)에 따라 데이터를 수정합니다. Step4 까지 모두 끝나야 다음 절차를 진행할 수 있습니다.")
        st.markdown("2. CTP, CTQ 선정 : CTP, CTQ 인자를 선정하고, Visualization 및 Correlation ratio & Heatmap을 통한 인자 간 관계를 확인합니다.")
        st.markdown("3. 핵심인자(CTP) 선택 : 핵심인자 중요도 평가 결과를 통해 최종 CTP 개수 결정합니다.")
        st.markdown("4. 머신러닝 모델 비교 : 각 ML 모델별 결과를 비교하여 최적 모델을 선택합니다.")
        st.markdown("5. 모델 최적화 : 선택된 모델 최적화 후 학습 모델과 Train 데이터 파일을 저장하거나 Stage2로 바로 넘어갑니다.")
        



    #사용자가 접속한 시간'session_in'을 session_state로 저장한 후 본 앱에 사용할 변수(d11~d14) 선언
        
    d11 = 'df' + session_in + '_1'
    d12 = 'df' + session_in + '_2'
    d13 = 'df' + session_in + '_3'
    d14 = 'df' + session_in + '_4'
    list1 = 'list' + session_in + '_1'
    list2 = 'list' + session_in + '_2'
    output_data = 'output' + session_in + '_1'
    output_model = 'output' + session_in + '_2'
    output_y = 'output' + session_in + '_3'
    
    #st.write(session_in+"build")#확인용
    #st.write(output_model)#확인용
    
    aa = pd.DataFrame()
    
    if d11 not in st.session_state:
        st.session_state[d11] = aa
    
    if d12 not in st.session_state:
        st.session_state[d12] = aa
      
    if d13 not in st.session_state:
        st.session_state[d13] = aa
    
    if d14 not in st.session_state:
        st.session_state[d14] = aa  
        
    if list1 not in st.session_state:
        st.session_state[list1] = aa  
        
    if list2 not in st.session_state:
        st.session_state[list2] = aa  
        
    if output_data not in st.session_state:
        st.session_state[output_data] = aa

    if output_model not in st.session_state:
        st.session_state[output_model] = aa
        
    if output_y not in st.session_state:
        st.session_state[output_y] = aa

#=========================================================================================================================================================================

    st.markdown("<h2 style='text-align: left; color: black;'>Stage1. 머신러닝 모델 생성</h2>", unsafe_allow_html=True)
    st.write("")
    st.write("")

    st.sidebar.subheader('1. 데이터 가공')

    uploaded_file = st.sidebar.file_uploader("데이터 파일(.csv) 업로드", type=["csv"])
    
    example = st.sidebar.selectbox("예제 데이터",('Select ▼','예제 데이터1', '예제 데이터2'))
    if example == 'Select ▼':
        pass
    elif example == '예제 데이터1':
        uploaded_file = './examples/example1.csv'
    elif example == '예제 데이터2':
        uploaded_file = './examples/example2.csv'
    
#=========================================================================================================================================================================



###### Main panel ##########

# Displays the dataset
    
           
    if uploaded_file is not None:
        def load_csv():
            csv = pd.read_csv(uploaded_file)
            return csv
      
        df = load_csv()
        
        
        st.subheader('1. 데이터 가공')
        st.write("")


        st.markdown('**1.1 입력 데이터 세트(Data Set)**')
        st.caption("**행 : 케이스(Sample)  /  열 : 특성(Feature)**")
        st._legacy_dataframe(df)
        st.write("")
        st.write("")
        st.write("")


        st.markdown('**1.2 입력 데이터 현황(Statistics)**')
        st.write(df.describe()) #statistics확인

        unique_col = []         #유일값 column 모음 리스트
        df_check1 = df.copy()   #체크용 dataframe복사
        
        df_check1 = df_check1.dropna()  #빈값지우기
        
        #categorical 데이터 삭제
        le = preprocessing.LabelEncoder()
        for column in df_check1.columns:
            if df_check1[column].dtypes == 'O':
                df_check1[column] = le.fit_transform(df_check1[column])
        
        #유일값 column들 모음 리스트에 넣기
        for column in df_check1.columns:
            if len(df_check1[column].unique()) == 1:
                unique_col.append(column)
        
        if len(unique_col) >= 1:
            st.error("공정인자 %s 가 하나의 값을 가짐 → 학습시 해당 공정인자를 제거하세요!!" %unique_col)
            df_check2 = df[unique_col]
            col1, col2 = st.columns([2,8])
            with col1:
                st.markdown("**_해당 내용 확인 ▶_**")
            with col2:
                st.dataframe(df_check2.style.set_properties(**{'color': 'red'}))
        else:
            pass
        st.write("")
        st.write("")
        st.write("")

        st.markdown('**1.3 데이터 가공 - 범주형데이터, 반복데이터, 결측데이터, 이상치 처리**')

        st.markdown('**Step1) 범주형 데이터(Categorical Data) 확인**')
        df11 = df.copy()
        for column in df.columns:
            if df[column].dtypes == 'O':
                cla = 1
                break
            else:
                cla = 0
                    
        if cla == 0:
            st.success("범주형 데이터 없음")
            if st.button('확인 완료 ', key=1):
                st.write(df11)
                st.session_state[d11] = df11
                #st.session_state[d12] = df11
                #st.session_state[d13] = df11
                #st.session_state[d14] = df11
        elif cla == 1:
            st.warning("범주형 데이터가 있습니다")
            if st.button('숫자형 데이터(Numerical Data)로 변환하기', key=1): 
                le = preprocessing.LabelEncoder()
                for column in df.columns:
                    if df11[column].dtypes == 'O':
                        df11[column] = le.fit_transform(df11[column])
                st.write(df11)
                st.session_state[d11] = df11
                #st.session_state[d12] = df11
                #st.session_state[d13] = df11
                #st.session_state[d14] = df11
        df11 = st.session_state[d11]
        
        
        st.write("")
        st.write("")
        st.write("")
        st.write('**Step2) 반복 데이터(Duplicated Data) 확인**')
        
        dupli = df.duplicated().sum()
        if dupli == 0:
            df12 = st.session_state[d11]
            st.success("반복된 데이터 없음")
            if st.button('확인 완료 ', key=2):
                st.write(df12)
                st.session_state[d12] = df12
                #st.session_state[d13] = df12
                #st.session_state[d14] = df12
        elif dupli != 0:
            df12 = st.session_state[d11]
            st.warning("반복된 데이터가 %d 개 있습니다." %dupli)
            if st.button('반복데이터 제거하기', key=2):
                df12 = df12.drop_duplicates()
                st.write(df12)
                st.session_state[d12] = df12
                #st.session_state[d13] = df12
                #st.session_state[d14] = df12
        df12 = st.session_state[d12]

        
        st.write("")
        st.write("")
        st.write("")
        st.write('**Step3) 결측 데이터(Missing Data) 확인**')
        
        miss = df12.isnull().sum().sum()
        if miss == 0:
            st.success("결측 데이터 없음")
            if st.button('확인 완료   '):
                df13 = st.session_state[d12]
                st.write(df13)
                st.session_state[d13] = df13
                #st.session_state[d14] = df13
        elif miss != 0:
            st.warning("결측 데이터가 %d 개 있습니다." %miss)
            option = st.selectbox('처리 방법은?',('select ▼','① 제거하기', '② 채우기'))
            if option == 'select ▼':
                pass
            elif option == '① 제거하기':
                df13 = st.session_state[d12]
                df13 = df13.dropna().reset_index()
                df13.drop(['index'],axis=1,inplace=True)
                st.write(df13)
                st.session_state[d13] = df13
                #st.session_state[d14] = df13
            elif option == '② 채우기':
                option2 = st.selectbox('채우기 방법은?',('select ▼','0(Zero)으로 채우기','앞의 값으로 채우기', '뒤의 값으로 채우기','보간 방법(Interpolation)으로 채우기','평균 값으로 채우기'))
                if option2 == 'select ▼':
                    pass
                elif option2 == '0(Zero)으로 채우기':
                    df13 = st.session_state[d12]
                    df13 = df13.fillna(0)
                    st.write(df13)
                    st.session_state[d13] = df13
                    #st.session_state[d14] = df13
                elif option2 == '앞의 값으로 채우기':
                    df13 = st.session_state[d12]
                    df13 = df13.fillna(method='ffill')
                    st.write(df13)
                    st.session_state[d13] = df13
                    #st.session_state[d14] = df13
                elif option2 == '뒤의 값으로 채우기':
                    df13 = st.session_state[d12]
                    df13 = df13.fillna(method='bfill')
                    st.write(df13)
                    st.session_state[d13] = df13
                    #st.session_state[d14] = df13
                elif option2 == '보간 방법(Interpolation)으로 채우기':
                    df13 = st.session_state[d12]
                    df13 = df13.fillna(df13.interpolate())
                    st.write(df13)
                    st.session_state[d13] = df13
                    #st.session_state[d14] = df13
                elif option2 == '평균 값으로 채우기':
                    df13 = st.session_state[d12]
                    df13 = df13.fillna(df13.mean())
                    st.write(df13)
                    st.session_state[d13] = df13
                    #st.session_state[d14] = df13
        df13 = st.session_state[d13]
        

        st.write("")
        st.write("")
        st.write("")
        st.write('**Step4) 이상치(Outlier Data) 확인**')
        
        
        #Feature(X인자)의 리스트 발췌 -> 버튼 만들 때 리스트로 사용
        newlist =  ['select ▼']
        for column in df.columns:
            if df[column].dtypes != 'O':
                newlist.append(column)
        
        
        out3 = st.selectbox("데이터 특성(Feature) 선택하기 : ", newlist)
        out2 = st.number_input('이상치 경계 선택하기(Number of Sigma(σ)) : ',0,10,3, format="%d")
        
        
        
        #Outlier Plot
        
        df14 = df13.copy()
        for column in df14.columns:
            if column == out3:
                OUT = df14[column].copy()
                OUT = pd.DataFrame(OUT)
                x = range(OUT.shape[0])

                OUT['max'] = np.mean(OUT)[0] + out2 * np.std(OUT)[0]  #Out = Outlier Criteria (σ)
                OUT['min'] = np.mean(OUT)[0] - out2 * np.std(OUT)[0]
                OUT['mean'] = np.mean(OUT)[0]
                
                sns.set(rc={'figure.figsize':(15,6)})
               
                plt.plot(x, OUT[out3], color = "black")
                plt.plot(x, OUT['max'], color = "red", label='Outlier limit')
                plt.plot(x, OUT['min'], color = "red")
                plt.plot(x, OUT['mean'], color = "#0e4194", label='Mean Value')
                plt.legend(loc='upper left', fontsize=15)
                
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()    
               


    #np.nan 이 있지만, 실제 지우지는 않고 outlier개수 확인 용 코드
        for i in range(len(df14.columns)):
            df_std = np.std(df14.iloc[:,i])
            df_mean = np.mean(df14.iloc[:,i])
            cut_off = df_std * out2
            upper_limit = df_mean + cut_off
            lower_limit = df_mean - cut_off
        
            for j in range(df14.shape[0]):
                if df14.iloc[j,i] > upper_limit or df14.iloc[j,i] <lower_limit:
                    df14.iloc[j,i] = np.nan
                    
        out1 = df14.isna().sum().sum() #out2기준을 넘는 outlier 개수
    
        
        #Outlier Cut
        if out1 == 0:
            st.success("기준을 초과한 이상치 없음")

            if st.button('확인 완료    '):
                df144 = df13.copy()
                st.write(df144)
                st.session_state[d14] = df144
        elif out1 != 0:
            st.warning("이상치 경계(Nσ) 기준을 초과한 이상치가 %d 개 있습니다." %out1)
            option3 = st.selectbox('처리 방법은?',('select ▼','① 모두 제거하기', '② 모두 처리하기', '③ 처리하지 않기'))

            #실제 지우기 위한 작업
            df144 = df13.copy()
            for i in range(len(df144.columns)):
                df_std = np.std(df144.iloc[:,i])
                df_mean = np.mean(df144.iloc[:,i])
                cut_off = df_std * out2
                upper_limit = df_mean + cut_off
                lower_limit = df_mean - cut_off
            
                for j in range(df144.shape[0]):
                    if df144.iloc[j,i] > upper_limit or df144.iloc[j,i] <lower_limit:
                        df144.iloc[j,i] = np.nan
                        
            out11 = df144.isna().sum().sum() # 실제 지우기 작업을 위한 np.nan 결과, 지워질 outlier 개수


            if option3 == 'select ▼':
                pass
            elif option3 == '① 모두 제거하기':
                df1444 = df144.copy()
                df1444 = df1444.dropna().reset_index()
                df1444.drop(['index'],axis=1,inplace=True)
                st.write(df1444)
                st.session_state[d14] = df1444
            elif option3 == '② 모두 처리하기':
                
                option4 = st.selectbox('처리 방법은?',('select ▼','0(Zero)으로 채우기','앞의 값으로 채우기', '뒤의 값으로 채우기','보간 방법(Interpolation)으로 채우기','평균 값으로 채우기'))
                
                if option4 == 'select ▼':
                    pass
                elif option4 == '0(Zero)으로 채우기':
                    df1444 = df144.copy()
                    df1444 = df1444.fillna(0)
                    st.write(df1444)
                    st.session_state[d14] = df1444
                elif option4 == '앞의 값으로 채우기':
                    df1444 = df144.copy()
                    df1444 = df1444.fillna(method='ffill')
                    st.write(df1444)
                    st.session_state[d14] = df1444
                elif option4 == '뒤의 값으로 채우기':
                    df1444 = df144.copy()
                    df1444 = df1444.fillna(method='bfill')
                    st.write(df1444)
                    st.session_state[d14] = df1444
                elif option4 == '보간 방법(Interpolation)으로 채우기':
                    df1444 = df144.copy()
                    df1444 = df1444.fillna(df1444.interpolate())
                    st.write(df1444)
                    st.session_state[d14] = df1444
                elif option4 == '평균 값으로 채우기':
                    df1444 = df144.copy()
                    df1444 = df1444.fillna(df1444.mean())
                    st.write(df1444)
                    st.session_state[d14] = df1444
                    
            elif option3 == '③ 처리하지 않기':
                df144 = df13.copy()
                st.write(df144)
                st.session_state[d14] = df144
                
        df1 = st.session_state[d14]
        st.write("")
        st.write("")
        st.markdown("**_· 가공된 학습 데이터 다운로드_**")
        st_pandas_to_csv_download_link(df1, file_name = "Final_cleaned_data.csv")
        st.caption("**_※저장 폴더 지정 방법 : 마우스 우클릭 → [다른 이름으로 링크 저장]_**")

        #확인용, 지우기
        #st.write(st.session_state[d11])
        #st.write(st.session_state[d12])
        #st.write(st.session_state[d13])
        #st.write(st.session_state[d14])
        
       

#========================================================================================================================
        if len(df1) == 0:
            st.write("")
            st.write("")
            st.error("Step1을 먼저 진행하세요.")
            
            ml = ['Linear Regression','Lasso','KNN','Decision_Tree','GBM','AB','XGBOOST','Extra Trees','RandomForest']
            st.session_state[list2] = ml
            
        else:
                
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.subheader('2. 공정인자(CTP), 품질인자(CTQ) 선정')
            st.caption("**CTP : X 인자  /  CTP : Y 인자**")
    
            st.sidebar.write("")
            st.sidebar.write("")
            st.sidebar.write("")
            st.sidebar.subheader('2. CTP, CTQ 선정')
              
    
            df2 = df1.copy()
    
            x = list(df2.columns)
            Selected_X = st.sidebar.multiselect("공정인자(CTP) 선택하기", x, x)
            y = [a for a in x if a not in Selected_X]
            Selected_y = st.sidebar.multiselect("품질인자(CTQ) 선택하기", y, y)
            Selected_X = np.array(Selected_X)
            Selected_y = np.array(Selected_y)
                
            
            num_y = pd.DataFrame(len(Selected_y), columns = ["num_Y"], index = ["value"])
            #st.write(num_y)#확인용
            
            
            if Selected_y.shape[0] <= 1:
                       
                st.write("")
                st.markdown('**2.1 CTP 개수 : %d**' %Selected_X.shape[0])
                st.info(list(Selected_X))
    
    
    
                st.write("")
                st.write("")
                st.markdown('**2.2 CTQ 개수 : %d**' %Selected_y.shape[0])
                st.info(list(Selected_y))
                
                st.session_state[output_y] = Selected_y.shape[0]
    
                df22 = pd.concat([df2[Selected_X],df2[Selected_y]], axis=1) #########################
                
    
                #st.write(df22.tail())
            
                #Selected_xy = np.array((Selected_X,Selected_y))
                st.write("")
                st.write("")
                st.markdown('**2.3 CTP, CTQ 시각화**')
                vis_col = ['모든 인자 보기']
                vis_col = pd.DataFrame(vis_col)
    
                test = list(df22.columns)
                test = pd.DataFrame(test)
    
                vis_col2 = vis_col.append(test)
    
                visual = st.multiselect('시각화 할 데이터 특성(Feature) 선택',vis_col2)
                if visual == ['모든 인자 보기']:
                    visual = df22.columns
                
            #st.write(visual)
            #st.write(df3.columns)
                if st.button('데이터 시각화하기'):
                    col1,col2 = st.columns([1,1])
                    plt.style.use('classic')
                    #fig, ax = plt.subplots(figsize=(10,10))
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    with col1:
                            #st.write('**X variable distribution**')
                            st.markdown("<h6 style='text-align: center; color: black;'>X variable distribution</h6>", unsafe_allow_html=True)
                    with col2:
                            st.markdown("<h6 style='text-align: center; color: black;'>X - Y Graph</h6>", unsafe_allow_html=True)
                
                    sns.set(font_scale = 0.8,rc={'figure.figsize':(10,5)})
                #plt.figure(figsize=(5,10))
                                
                    for vis in visual:
                    
                        fig, axs = plt.subplots(ncols=2)
                    
                        g = sns.distplot(df22[vis], hist_kws={'alpha':0.5}, bins=8, kde_kws={'color':'xkcd:purple','lw':3}, ax=axs[0])
    
                        g2 = sns.scatterplot(x=df22[vis],y=df22.iloc[:,-1],s=60,color='red', ax=axs[1])
    
                        g2 = sns.regplot(x=df22[vis],y=df22.iloc[:,-1], scatter=False, ax=axs[1])
    
                        st.pyplot()
                    
            
    
            
                if st.button('CTP, CTQ 상관관계 확인하기'):
                
                    df_cor = df22.corr()
                    st.write(df_cor)
            
                    #df22.to_csv('output.csv',index=False)
                    #df22 = pd.read_csv('output.csv')
                    
                    corr = df22.corr()
                    mask = np.zeros_like(corr)
                    mask[np.triu_indices_from(mask)] = True
                    sns.set(rc = {'figure.figsize':(6,6)},font_scale=0.5)
                    #sns.set(font_scale=0.4)
                    with sns.axes_style("white"):
                        f, ax = plt.subplots()
                        #ax = sns.heatmap(corr,mask=mask, vmax=1.0, square=True, annot = True, cbar_kws={"shrink": 0.7}, cmap='coolwarm', linewidths=.5)
                        ax = sns.heatmap(corr,  vmax=1, square=True, cbar_kws={"shrink": 0.7}, annot = True, cmap='coolwarm', linewidths=.5)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
            
                    st.pyplot()
#=========================================================================================================================================================================   
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.subheader('3. 핵심인자(CTP) 선택')
                st.write("")
                st.markdown('**3.1 핵심인자(CTP) 비교 결과**')  
            
            
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.subheader('3. 핵심인자(CTP) 선택')
            
                       
                if st.sidebar.button('핵심인자 확인하기(RFE Method)'):
                    feature_s(df22)
            
                with st.sidebar.markdown("**최종 CTP 선택**"):
                
                    fs_list = []
                    for i in range(1,df2[Selected_X].shape[1]+1):
                        fs_list.append(i)
            
                    hobby = st.sidebar.selectbox("최종 CTP 개수 선택하기 : ", fs_list)
                    X_rfe_columns = F_feature(df22,hobby)
                    X_column = pd.DataFrame(X_rfe_columns,columns=['Variables'])
            
                
        #            count=0
                    Selected_X2 = list(X_column.Variables)
                    Selected_y = list(Selected_y)
    
        #           count +=1
                
                df3 = pd.concat([df22[Selected_X2],df22[Selected_y]], axis=1) #df22 -> df3 생성

                
                st.write("")
                st.write("")
                st.write("")
                st.markdown('**3.2 최종 CTP, CTQ**')
    
                st.write('최종 공정인자(CTP) 개수 : %d' %len(Selected_X2))
                st.info(list(Selected_X2))
    
                st.write('최종 품질인자(CTQ) 개수 : %d' %len(Selected_y))
                st.info(list(Selected_y))
    
    
#=========================================================================================================================================================================
                st.write("")
                st.write("")        
                st.write("")
                st.write("")
                st.write("")
    
                st.subheader('4. 머신러닝 모델 비교')
    
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
    
                with st.sidebar.subheader('4. 머신러닝 모델 비교'):
            
                    ml = ['Linear Regression','Lasso','KNN','Decision_Tree','GBM','AB','XGBOOST','Extra Trees','RandomForest']
                    Selected_ml = st.sidebar.multiselect('알고리즘 선택하기', ml, ml)
    
                if st.sidebar.button('모델 비교하기'):
                    M_list = build_model(df3, Selected_ml)
                    M_list_names = list(M_list['Machine_Learning_Model'])
                    M_list = list(M_list['Machine_Learning_Model'][:3])
                    st.session_state[list1] = M_list
                    st.session_state[list2] = M_list_names
                    
                else:
                    st.write("")
                    st.markdown("**4.1. 모델 검증 방법 :**  _K-Fold Cross Validation_")
                    st.write("")
                    st.markdown("**4.2. 머신러닝 모델 비교 결과**")
    
                
#=========================================================================================================================================================================    
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                with st.sidebar.subheader('5. 모델 최적화'):
                    
                    st.sidebar.markdown('**5.1. 모델 최적화(Voting기법)**')
                    
                    #ml = ['Linear Regression','Lasso','KNN','Decision_Tree','AB','GBM','XGBOOST','Extra Trees','RandomForest']
                    
                    ml2 = st.session_state[list1]
                    
                    if len(ml2) == 0:
                        ml2 = ['GBM','XGBOOST','Extra Trees','RandomForest']
                    
                    else:
                        ml2 = ml2
                    Selected_ml2 = st.sidebar.multiselect('최적 알고리즘 선택하기', ml2, ml2)
                    Selected_ml3 = list(Selected_ml2)
                
            
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.subheader('5. 모델 최적화')
                st.write("")
                st.markdown('**5.1. 모델 최적화(Voting기법)**')
                
                if st.sidebar.button('최종 모델 확인 (Voting)'):
                
                
                    X = df3.iloc[:,:-1] # Using all column except for the last column as X
                    Y = df3.iloc[:,-1] # Selecting the last column as Y
                    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
                    models =[]
                    
                    for model in Selected_ml3:
                        
                    
                        if model == 'Linear Regression':
                            models.append(('Linear Regression', Pipeline([('Scaler', StandardScaler()),('Linear Regression',LinearRegression())])))
                        if model == 'Lasso':
                            models.append(('LASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())])))
                        if model == 'KNN':
                            models.append(('KNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor())])))      
                        if model == 'Decision_Tree':
                            models.append(('Decision_Tree', Pipeline([('Scaler', StandardScaler()),('Decision_Tree',DecisionTreeRegressor())])))   
                        if model == 'GBM':
                            models.append(('GBM', Pipeline([('Scaler', StandardScaler()),('GBM',GradientBoostingRegressor(n_estimators=75))])))
                        if model == 'XGBOOST':
                            models.append(('XGBOOST', Pipeline([('Scaler', StandardScaler()),('XGBOOST',xgboost.XGBRegressor(booster='gbtree',n_estimators= 100))])))
                        if model == 'AB':
                            models.append(('AB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
                        if model == 'Extra Trees':
                            models.append(('Extra Trees', Pipeline([('Scaler', StandardScaler()),('Extra Trees',ExtraTreesRegressor())])))
                        if model == 'RandomForest':
                            models.append(('RandomForest', Pipeline([('Scaler', StandardScaler()),('RandomForest',RandomForestRegressor())])))
                
    
                    k=0
                    
                             
                    
                    model = VotingRegressor(estimators=models).fit(X,Y)
                
    
                    results = []
    
                    msg = []
                    mean = []
                    std = []        
    
                    
                    kfold = KFold(n_splits=5, random_state=7, shuffle=True)
                    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring='r2')
                
                    for i, element in enumerate(cv_results):
                        if element <= 0.0:
                            cv_results[i] = 0.0
                        
                    results.append(abs(cv_results))
                #    names.append(name)
                    msg.append('%s' % model)
                    mean.append('%f' %  (cv_results.mean()))
                    std.append('%f' % (cv_results.std()))
                        
                        
                    F_result3 = pd.DataFrame(np.transpose(msg))
                    F_result3.columns = ['Machine_Learning_Model']
                    F_result3['R2_Mean'] = pd.DataFrame(np.transpose(mean))
                    F_result3['R2_Std'] = pd.DataFrame(np.transpose(std))
                
                #st.write(F_result3)    
            
            
                    st.markdown('**_모델 최적화 결과_**')
                
                    st.write('Voting 모델 정확도 ($R^2$):')
                
                    R2_mean = list(F_result3['R2_Mean'].values)
                    st.info( R2_mean[0] )
                    
                    st.write('모델 정확도 편차 (Standard Deviation):')
                
                    R2_std = list(F_result3['R2_Std'].values)
                    st.info( R2_std[0])
                
                
                    
                #scaler = StandardScaler().fit(X)
                #rescaled = scaler.transform(X)
                
                
                    predictions = model.predict(X)
                    predictions = pd.DataFrame(predictions).values              
            
            
                #st.markdown('*Voting Model Results*')
                #st.write(F_result3)
                
                
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    plt.plot(Y, Y, color='#0e4194', label = 'Actual data')
                    plt.scatter(Y, predictions, color='red', label = 'Prediction')
                    plt.xlabel(df3.columns[-1])
                    plt.ylabel(df3.columns[-1])
                    #plt.ylim(39.7,39.9) 
                    #plt.xlim(39.7, 39.9) 
                    plt.legend()
                    st.pyplot()
                
            
                    st.write("")
                    st.write("")
                    st.write("")
                    
                    st.session_state[output_data] = df3
                    st.session_state[output_model] = model
                    
                    #st.write(st.session_state[output_data])#확인용
                    #st.write(st.session_state[output_model])#확인용
                    
                    st.markdown('**_· Option1) 학습 완료 모델 & 학습 데이터 자동저장 완료 → Stage2로 넘어가기_**')
                    st.caption("**_※ [F5키] 누르지 마세요! 자동 저장된 데이터가 초기화 됩니다._**") 
                    st.write("")
                    st.write("")
                    
                    st.markdown('**_· Option2) 학습 완료 모델(.pkl) & 학습 데이터(.xlsx) 저장하기_**')
                    
                    download_data_xlsx(df3, num_y)
                    download_model(k,model)
                    
                    st.caption("**_※ 저장 폴더 지정 방법 : 마우스 우클릭 → [다른 이름으로 링크 저장]_**") 


                st.sidebar.write("")
    
                with st.sidebar.markdown('**5.2. 하이퍼파라미터 최적화**'):
                    
                    M_list_names = st.session_state[list2]
                    
                    max_num = df3.shape[1]
                    #ml = ['Linear Regression','Lasso','KNN','Decision_Tree','GBM','XGBOOST','Extra Trees','RandomForest','Neural Network']
                    Model = st.sidebar.selectbox('하이퍼파라미터 튜닝 - 알고리즘 선택하기',M_list_names)
                    #if Model == 'Linear Regression'or'Lasso':
                        #    parameter_n_neighbers = st.sidebar.slider('Number of neighbers', 2, 10, 6, 2)
                        # st.sidebar.markdown('No Hyper Parameter)
                    if Model == 'KNN':
                        parameter_n_neighbors_knn = st.sidebar.slider('Number of neighbers', 2, 10, (2,8), 2)
                        parameter_n_neighbors_step_knn = st.sidebar.number_input('Step size for n_neighbors', 1)
                        n_neighbors_range = np.arange(parameter_n_neighbors_knn[0], parameter_n_neighbors_knn[1]+parameter_n_neighbors_step_knn, parameter_n_neighbors_step_knn)
                        param_grid_knn = dict(n_neighbors=n_neighbors_range)
                
                    elif Model == 'GBM' or Model == 'Extra Trees' or Model == 'RandomForest':
                        parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 301, (11,151), 20)
                        parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 20)
                        parameter_max_features = st.sidebar.slider('Max features (max_features)', 1, max_num, (1,3), 1)
                        parameter_max_features_step = st.sidebar.number_input('Step size for max_features', 1)
                        #parameter_max_depth = st.sidebar.slider('Number of max_depth (max_depth)', 10, 100, (30,80), 10)
                        #parameter_max_depth_step = st.sidebar.number_input('Step size for max_depth', 10)
                        n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
                        max_features_range = np.arange(parameter_max_features[0], parameter_max_features[1]+parameter_max_features_step, parameter_max_features_step)
                        #max_depth_range = np.arange(parameter_max_depth[0], parameter_max_depth[1]+parameter_max_depth_step, parameter_max_depth_step)
                        param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)
                    
                    elif Model == 'AB' :
                        parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 1, 301, (11,151), 20)
                        parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 20)
                        parameter_learning_rate = st.sidebar.slider('learning_rate', 0.1, 2.0, (0.1,0.6), 0.2)
                        parameter_learning_rate_step = st.sidebar.number_input('Step size for learing_rate', 0.2)
                        n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
                        learning_rate_range = np.arange(parameter_learning_rate[0], parameter_learning_rate[1]+parameter_learning_rate_step, parameter_learning_rate_step)
                        param_grid = dict(learning_rate=learning_rate_range, n_estimators=n_estimators_range)
                    
                
                    elif Model == 'XGBOOST' :
                        parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 1, 301, (41,101), 20)
                        parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 20)
                        parameter_max_depth = st.sidebar.slider('max_depth', 0, 10, (2,5), 1)
                        parameter_max_depth_step = st.sidebar.number_input('Step size for max_depth', 1)
                        n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
                        max_depth_range = np.arange(parameter_max_depth[0], parameter_max_depth[1]+parameter_max_depth_step, parameter_max_depth_step)
                        param_grid = dict(max_depth=max_depth_range, n_estimators=n_estimators_range)
                
                    elif Model == 'Neural Network':
                        parameter_n_estimators = st.sidebar.slider('Number of first nodes', 10, 100, (10,40), 10)
                        parameter_n_estimators_step = st.sidebar.number_input('Step size for first nodes', 10)
                        parameter_n_estimators2 = st.sidebar.slider('Number of Second nodes', 10, 100, (10,40), 10)
                        parameter_n_estimators_step2 = st.sidebar.number_input('Step size for second nodes', 10)
                        n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
                        n_estimators_range2 = np.arange(parameter_n_estimators2[0], parameter_n_estimators2[1]+parameter_n_estimators_step2, parameter_n_estimators_step2)
                        param_grid = dict(n_estimators=n_estimators_range, n_estimators2=n_estimators_range2)
                
                    elif Model == 'Linear Regression' or Model == 'Lasso' or Model == 'Decision_Tree':
                        st.sidebar.markdown('_해당 알고리즘은 해당되지 않습니다._')
    
    
                   # if model == 'Neural Network' or 'XGBOOST' or 'AB' or 'GBM' or 'Extra Trees' or 'RandomForest' or 'KNN':
    
                       
    
                if st.sidebar.button('최종 모델 확인 (튜닝완료)'):
                    st.write("")
                    st.write("")
                    st.write("")
                    st.markdown('**5.2. 하이퍼파라미터(Hyper Parameter) 최적화**')
                    
                    st.markdown("**_하이퍼파라미터 최적화 결과_**")
                    
                    X = df3.iloc[:,:-1] # Using all column except for the last column as X
                    Y = df3.iloc[:,-1] # Selecting the last column as Y
    
                    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
            
                    #rescaled = scaler.transform(X)
                    #rescaledX = scaler.transform(X_train)
                    #rescaledTestX = scaler.transform(X_test)
                
                    k=0
                    if Model == 'Linear Regression':
                        #    print(X_train, y_train)
                        model = Pipeline([('Scaler', StandardScaler()),('Linear Regression',LinearRegression())])
                        #Pipeline([('Scaler', StandardScaler()),('Linear Regression',LinearRegression())])))
                    
            
                    elif Model == 'Lasso':
                    #    print(X_train, y_train)
                        model = Pipeline([('Scaler', StandardScaler()),('Lasso',Lasso())])
                    #model.fit(rescaled, y_train)
    
                    
                    elif Model == 'Decision_Tree':
                    #    print(X_train, y_train)
                        model = Pipeline([('Scaler', StandardScaler()),('Decision_Tree',DecisionTreeRegressor())])
                    #model.fit(rescaled, y_train)
                    
                                
                    elif Model == 'KNN':
                
                        a = Opti_KNN_model(df3,parameter_n_neighbors_knn,parameter_n_neighbors_step_knn,param_grid_knn)
                    #    print(X_train, y_train)
                        model = Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor(n_neighbors=a))])
                    #model.fit(rescaledX, y_train)
            
                    elif Model == 'GBM':
                        a, b = Opti_model(Model,df3,parameter_n_estimators,parameter_max_features,param_grid)
                    #st.write(a, b)
                    #    print(X_train, y_train)
                        model = Pipeline([('Scaler', StandardScaler()),('GBM',GradientBoostingRegressor(n_estimators=a, max_features=b))])
                    #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                    #model.fit(rescaledX, y_train)
                    
                    elif Model == 'AB':
                        a, b = Opti_model3(Model,df3,parameter_n_estimators,parameter_learning_rate,param_grid)
                    #st.write(a, b)
                    #    print(X_train, y_train)
                        model = Pipeline([('Scaler', StandardScaler()),('AB',AdaBoostRegressor(n_estimators=a, learning_rate=b))])
                    #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                    #model.fit(rescaledX, y_train)
                
                    elif Model == 'XGBOOST':
                        a, b = Opti_model2(Model,df3,parameter_n_estimators,parameter_max_depth,param_grid)
                    #st.write(a, b)
                    #    print(X_train, y_train)
                        model = Pipeline([('Scaler', StandardScaler()),('XGBOOST',xgboost.XGBRegressor(booster='gbtree',n_estimators=a, max_depth=b))])
                    #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                    #model.fit(rescaledX, y_train)
                    
                    elif Model == 'Extra Trees':
                        a, b = Opti_model(Model,df3,parameter_n_estimators,parameter_max_features,param_grid)
                    #st.write(a, b)
                    #    print(X_train, y_train)
                        model = Pipeline([('Scaler', StandardScaler()),('Extra Trees',ExtraTreesRegressor(n_estimators=a, max_features=b))])
                    #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                    #model.fit(rescaledX, y_train)
                    elif Model == 'RandomForest':
                        a, b = Opti_model(Model,df3,parameter_n_estimators,parameter_max_features,param_grid)
                    #st.write(a, b)
                    #    print(X_train, y_train)
                        model = Pipeline([('Scaler', StandardScaler()),('RandomForest',RandomForestRegressor(n_estimators=a, max_features=b))])
                    #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                    #model.fit(rescaledX, y_train)
    
    
                    results = []
    
                    msg = []
                    mean = []
                    std = []        
    
                    
                    kfold = KFold(n_splits=5, random_state=7, shuffle=True)
                    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring='r2')
                
                    for i, element in enumerate(cv_results):
                        if element <= 0.0:
                            cv_results[i] = 0.0
                        
                        
                    results.append(cv_results)
                    #    names.append(name)
                    msg.append('%s' % Model)
                    mean.append('%f' %  (cv_results.mean()))
                    std.append('%f' % (cv_results.std()))
                        
                        
                    F_result3 = pd.DataFrame(np.transpose(msg))
                    F_result3.columns = ['Machine_Learning_Model']
                    F_result3['R2_Mean'] = pd.DataFrame(np.transpose(mean))
                    F_result3['R2_Std'] = pd.DataFrame(np.transpose(std))
                
                    #st.write(F_result3)    
            
            
                                       
                    st.write('최종 모델 정확도 ($R^2$):')
                
                    R2_mean = list(F_result3['R2_Mean'].values)
                    st.info( R2_mean[0] )
                    
                    st.write('모델 정확도 편차 (Standard Deviation):')
                
                    R2_std = list(F_result3['R2_Std'].values)
                    st.info( R2_std[0])
                    
            
                    model.fit(X_train,y_train)
                
                    predictions = model.predict(X)
                    predictions = pd.DataFrame(predictions).values
    
                
                
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    plt.plot(Y, Y, color='#0e4194', label = 'Actual data')
                    plt.scatter(Y, predictions, color='red', label = 'Prediction')
                    plt.xlabel(df3.columns[-1])
                    plt.ylabel(df3.columns[-1])
                    #plt.ylim(39.7,39.9) 
                    #plt.xlim(39.7, 39.9) 
                    plt.legend()
                    st.pyplot()
                
                    st.session_state[output_data] = df3
                    st.session_state[output_model] = model
                    
                    
                    st.write("")
                    st.write("")
                    st.write("")
                    
                    
                    st.markdown('**_· Option1) 학습 완료 모델 & 학습 데이터 자동저장 완료 → Stage2로 넘어가기_**')
                    st.caption("**_※ [F5키] 누르지 마세요! 자동 저장된 데이터가 초기화 됩니다._**") 
                    st.write("")
                    st.write("")
                    
                    st.markdown('**_· Option2) 학습 완료 모델(.pkl) & 학습 데이터(.xlsx) 저장하기_**')
                    #st_pandas_to_csv_download_link(df3, file_name = "01_Train_data.csv")
                    download_data_xlsx(df3, num_y)
                    download_model(k,model)
                    st.caption("**_※ 저장 폴더 지정 방법 : 마우스 우클릭 → [다른 이름으로 링크 저장]_**") 
                    
                    
                    

                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                    
    
#=========================================================================================================================================================================
#=========================================================================================================================================================================
#=========================================================================================================================================================================
    
            else :
                
                
                st.write("")
                st.markdown('**2.1 CTP 개수 : %d**' %Selected_X.shape[0])
                st.info(list(Selected_X))
    
                st.write("")
                st.write("")
                st.markdown('**2.2 CTQ 개수 : %d**' %Selected_y.shape[0])
                st.info(list(Selected_y))
                
                st.session_state[output_y] = Selected_y.shape[0]
    
                df222 = pd.concat([df2[Selected_X],df2[Selected_y]], axis=1)
    
                st.write("")
                st.write("")
                st.markdown('**2.3 CTP, CTQ 시각화**')
                vis_col = ['모든 인자 보기']
                vis_col = pd.DataFrame(vis_col)
    
                test = list(df222.columns)
                test = pd.DataFrame(test)
    
                vis_col2 = vis_col.append(test)
    
                visual = st.multiselect('시각화 할 데이터 특성(Feature) 선택',vis_col2)
                if visual == ['모든 인자 보기']:
                    visual = df222.columns
                
    
                if st.button('데이터 시각화하기', key = 10):
                    col1,col2 = st.columns([1,1])
                    plt.style.use('classic')
                    #fig, ax = plt.subplots(figsize=(10,10))
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    with col1:
                            #st.write('**X variable distribution**')
                            st.markdown("<h6 style='text-align: center; color: black;'>X variable distribution</h6>", unsafe_allow_html=True)
                    with col2:
                            st.markdown("<h6 style='text-align: center; color: black;'>X - Y Graph</h6>", unsafe_allow_html=True)
                
                    sns.set(font_scale = 0.8,rc={'figure.figsize':(10,5)})
                #plt.figure(figsize=(5,10))
                                
                    for vis in visual:
                    
                        fig, axs = plt.subplots(ncols=2)
                    
                        g = sns.distplot(df222[vis], hist_kws={'alpha':0.5}, bins=8, kde_kws={'color':'xkcd:purple','lw':3}, ax=axs[0])
                        color = ['red','#0e4194','green']
                        i=0
                        for feature in Selected_y:
                        #g2 = sns.scatterplot(x=df3[vis],y=df3.iloc[:,-1],s=60,color='red', ax=axs[1])
                            g2 = sns.scatterplot(x=df222[vis],y=df222[feature],s=60,color=color[i],ax=axs[1],label=feature)
                        #g2 = sns.regplot(x=df3[vis],y=df3.iloc[:,-1], scatter=False, ax=axs[1])
                            i+=1
                    
                        st.pyplot()
                    
            
    
            
                if st.button('CTP, CTQ 상관관계 확인하기'):
                
                    df_cor = df222.corr()
                    st.write(df_cor)
            
                    #df222.to_csv('output.csv',index=False)
                    #df222 = pd.read_csv('output.csv')
    
                    corr = df222.corr()
                    mask = np.zeros_like(corr)
                    mask[np.triu_indices_from(mask)] = True
                    sns.set(rc = {'figure.figsize':(6,6)},font_scale=0.5)
                    #sns.set(font_scale=0.4)
                    with sns.axes_style("white"):
                        f, ax = plt.subplots()
                        #ax = sns.heatmap(corr,mask=mask, vmax=1.0, square=True, annot = True, cbar_kws={"shrink": 0.7}, cmap='coolwarm', linewidths=.5)
                        ax = sns.heatmap(corr,  vmax=1, square=True, cbar_kws={"shrink": 0.7}, annot = True, cmap='coolwarm', linewidths=.5)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
            
                    st.pyplot()
            
    
#=========================================================================================================================================================================
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.subheader('3. 핵심인자(CTP) 선택')
                st.write("")
                st.markdown('**3.1 핵심인자(CTP) 비교 결과**')  
            
               
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.subheader('3. 핵심인자(CTP) 선택')
            
                
                if st.sidebar.button('핵심인자 확인하기'):
                    
                    feature_m(df222,Selected_X,Selected_y )
                    
                    
            
                with st.sidebar.markdown('**최종 CTP 선택**'):
                
            
                    fs_list2 = []
                    
                    for i in range(1,df2[Selected_X].shape[1]+1):
                        fs_list2.append(i)
                        
            
                    hobby2 = st.sidebar.selectbox("최종 CTP 개수 선택하기 : ", fs_list2)
                    
                    
                    X_rfe_columns = F_feature_m(df222,hobby2,Selected_X,Selected_y)
                
            
                    X_column = pd.DataFrame(X_rfe_columns,columns=['Variables'])
              
                  #count=0
                    Selected_X2 = list(X_column.Variables)
                    Selected_y = list(Selected_y)
                    #count +=1
         
                df33 = pd.concat([df222[Selected_X2],df222[Selected_y]], axis=1)
                
                st.write("")
                st.write("")
                st.write("")
                st.markdown('**3.2 최종 CTP, CTQ**')
    
            #st.write('**3.1 Selected X, Y Variables**')
                st.markdown('최종 공정인자(CTP) 개수 : %d' %len(Selected_X2))
                st.info(list(Selected_X2))
    
                st.markdown('최종 품질인자(CTQ) 개수 : %d' %len(Selected_y))
                st.info(list(Selected_y))
    
    
    
#=========================================================================================================================================================================
    
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.subheader('4. 머신러닝 모델 비교')
           
            #with st.sidebar.header('2. Feature Selection '):
    
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                with st.sidebar.subheader('4. 머신러닝 모델 비교'):
            
                    ml = ['Linear Regression','Lasso','KNN','Decision_Tree','GBM','AB','XGBOOST','Extra Trees','RandomForest']
                    Selected_ml = st.sidebar.multiselect('알고리즘 선택하기', ml, ml)
    
                if st.sidebar.button('모델 비교하기'):
                    M_list = build_model_m(df33, Selected_ml, Selected_X2,Selected_y)
                    M_list_names = list(M_list['Machine_Learning_Model'])
                    #st.write(M_list_names)
                    st.session_state[list2] = M_list_names
             
                else:
                    st.write("")
                    st.markdown('**4.1. 모델 검증 방법 :**  _K-Fold Cross Validation_')
                    st.write("")
                    st.markdown("**4.2. 머신러닝 모델 비교 결과**")
    
                
#=========================================================================================================================================================================
    
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.subheader('5. 모델 최적화')
            
                
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.subheader('5. 모델 최적화')
    
    
                with st.sidebar.markdown('**5.1. 하이퍼파라미터 최적화**'):
                
                    max_num = df33.shape[1]
                    #ml = ['Linear Regression','Lasso','KNN','Decision_Tree','GBM','XGBOOST','Extra Trees','RandomForest','Neural Network']
                    M_list_names = st.session_state[list2]
                    Model = st.sidebar.selectbox('하이퍼파라미터 튜닝 - 알고리즘 선택하기', M_list_names)
                    #if Model == 'Linear Regression'or'Lasso':
                        #    parameter_n_neighbers = st.sidebar.slider('Number of neighbers', 2, 10, 6, 2)
                        # st.sidebar.markdown('No Hyper Parameter)
                    if Model == 'KNN':
                        parameter_n_neighbors_knn = st.sidebar.slider('Number of neighbers', 2, 10, (2,8), 2)
                        parameter_n_neighbors_step_knn = st.sidebar.number_input('Step size for n_neighbors', 1)
                        n_neighbors_range = np.arange(parameter_n_neighbors_knn[0], parameter_n_neighbors_knn[1]+parameter_n_neighbors_step_knn, parameter_n_neighbors_step_knn)
                        param_grid_knn = dict(estimator__n_neighbors=n_neighbors_range)
                
                    elif Model == 'GBM' or Model == 'Extra Trees' or Model == 'RandomForest':
                        parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 501, (101,251), 30)
                        parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 30)
                        parameter_max_features = st.sidebar.slider('Max features (max_features)', 1, max_num, (1,3), 1)
                        parameter_max_features_step = st.sidebar.number_input('Step size for max_features', 1)
                        #parameter_max_depth = st.sidebar.slider('Number of max_depth (max_depth)', 10, 100, (30,80), 10)
                        #parameter_max_depth_step = st.sidebar.number_input('Step size for max_depth', 10)
                        n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
                        max_features_range = np.arange(parameter_max_features[0], parameter_max_features[1]+parameter_max_features_step, parameter_max_features_step)
                        #max_depth_range = np.arange(parameter_max_depth[0], parameter_max_depth[1]+parameter_max_depth_step, parameter_max_depth_step)
                        param_grid = dict(estimator__max_features=max_features_range, estimator__n_estimators=n_estimators_range)
                    
                    elif Model == 'AB' :
                        parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 1, 501, (101,251), 20)
                        parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 30)
                        parameter_learning_rate = st.sidebar.slider('learning_rate', 0.1, 2.0, (0.1,0.6), 0.2)
                        parameter_learning_rate_step = st.sidebar.number_input('Step size for learing_rate', 0.2)
                        n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
                        learning_rate_range = np.arange(parameter_learning_rate[0], parameter_learning_rate[1]+parameter_learning_rate_step, parameter_learning_rate_step)
                        param_grid = dict(estimator__learning_rate=learning_rate_range, estimator__n_estimators=n_estimators_range)
                    
                
                    elif Model == 'XGBOOST' :
                        parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 1, 301, (41,101), 20)
                        parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 20)
                        parameter_max_depth = st.sidebar.slider('max_depth', 0, 10, (2,5), 1)
                        parameter_max_depth_step = st.sidebar.number_input('Step size for max_depth', 1)
                        n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
                        max_depth_range = np.arange(parameter_max_depth[0], parameter_max_depth[1]+parameter_max_depth_step, parameter_max_depth_step)
                        param_grid = dict(estimator__max_depth=max_depth_range, estimator__n_estimators=n_estimators_range)
                
    
                    elif Model == 'Linear Regression' or Model == 'Lasso' or Model == 'Decision_Tree':
                        st.sidebar.markdown('_해당 알고리즘은 해당되지 않습니다._')
    
    
    
    
    
                st.markdown('**5.1. 모델 하이퍼파라미터 최적화**')        
    
                if st.sidebar.button('최종 모델 확인하기'):
                    
                    st.write('**_모델 최적화 결과_**')
                    
                    
                    X = df33[Selected_X2] # Using all column except for the last column as X
                    Y = df33[Selected_y] # Selecting the last column as Y
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
            
    
                
                    if Model == 'Linear Regression':
                        #    print(X_train, y_train)
                        model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('Linear Regression',LinearRegression())]))
    
            
                    elif Model == 'Lasso':
                    #    print(X_train, y_train)
                        model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())]))
                    #model.fit(rescaled, y_train)
    
                    
                    elif Model == 'Decision_Tree':
                    #    print(X_train, y_train)
                        model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('Decision_Tree',DecisionTreeRegressor())]))
                    #model.fit(rescaled, y_train)
                    
                                
                    elif Model == 'KNN':
                        
                        a = Opti_KNN_model_m(df33,param_grid_knn,Selected_X2,Selected_y)
                
                        model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor(n_neighbors=a))]))
                    #model.fit(rescaledX, y_train)
            
                    elif Model == 'GBM':
                        
                        a, b = Opti_model_m(Model,df33,param_grid,Selected_X2,Selected_y)
                        
                        model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()), ('GBM',GradientBoostingRegressor(n_estimators=a, max_features=b))]))
                    #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                    #model.fit(rescaledX, y_train)
                    
                    elif Model == 'AB':
                        
                        a, b = Opti_model3_m(Model,df33,param_grid,Selected_X2,Selected_y)
                        
                        model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostRegressor(n_estimators=a, learning_rate=b))]))
                    #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                    #model.fit(rescaledX, y_train)
                
                    elif Model == 'XGBOOST':
                        
                        a, b = Opti_model2_m(Model,df33,param_grid,Selected_X2,Selected_y)
                 
                        model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('XGBOOST',xgboost.XGBRegressor(n_estimators=a, max_depth=b))]))
                    #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                    #model.fit(rescaledX, y_train)
                    
                    elif Model == 'Extra Trees':
                        
                        a, b = Opti_model_m(Model,df33,param_grid,Selected_X2,Selected_y)
                       
                        model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('Extra Trees',ExtraTreesRegressor(n_estimators=a, max_features=b))]))
                    #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                    #model.fit(rescaledX, y_train)
                    elif Model == 'RandomForest':
                        
                        a, b = Opti_model_m(Model,df33,param_grid,Selected_X2,Selected_y)
                        
                        model = MultiOutputRegressor(Pipeline([('Scaler', StandardScaler()),('RandomForest',RandomForestRegressor(n_estimators=a, max_features=b))]))
                    #model = Model(n_estimators=grid.best_params_['n_estimators'], max_features=grid.best_params_['max_features'])
                    #model.fit(rescaledX, y_train)
    
    
                    results = []
    
                    msg = []
                    mean = []
                    std = []        
    
                    
                    kfold = KFold(n_splits=5, random_state=7, shuffle=True)
                    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring='r2')
                
                    for i, element in enumerate(cv_results):
                        if element <= 0.0:
                            cv_results[i] = 0.0
                        
                        
                    results.append(cv_results)
                    #    names.append(name)
                    msg.append('%s' % Model)
                    mean.append('%f' %  (cv_results.mean()))
                    std.append('%f' % (cv_results.std()))
                        
                        
                    
                    
                    F_result3 = pd.DataFrame(np.transpose(msg))
                    F_result3.columns = ['Machine_Learning_Model']
                    F_result3['R2_Mean'] = pd.DataFrame(np.transpose(mean))
                    F_result3['R2_Std'] = pd.DataFrame(np.transpose(std))
                
                    #st.write(F_result3)    
            
            
                    
                    
                    st.write('최종 모델 정확도 ($R^2$):')
                
                    R2_mean = list(F_result3['R2_Mean'].values)
                    st.info( R2_mean[0] )
                    
                    st.write('모델 정확도 편차 (Standard Deviation):')
                
                    R2_std = list(F_result3['R2_Std'].values)
                    st.info( R2_std[0])
                    
            
                    model.fit(X_train,y_train)
                
                    predictions = model.predict(X)
                    predictions = pd.DataFrame(predictions)
    
    
                    
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    plt.figure(figsize=(10,6))
                    fig, axs = plt.subplots(ncols=Y.shape[1])
                    fig.subplots_adjust(hspace=1)
                    
                    
    
    
                    for i in range(1,Y.shape[1]+1):
                    
    
                        
                        plt.subplot(1,Y.shape[1],i)
                        
                        plt.plot(Y.iloc[:,i-1], Y.iloc[:,i-1], color='#0e4194', label = 'Actual data')
                        plt.scatter(Y.iloc[:,i-1], predictions.iloc[:,i-1], color='red', label = 'Prediction')
                        plt.title(Y.columns[i-1],fontsize=10)
                        plt.xticks(fontsize=8)
                        plt.yticks(fontsize=8)
                        #ax.set_xlabel('Time', fontsize=16)
                        #plt.ylabel(Y.columns[i-1], fontsize=10)
                        
                    st.pyplot()
                        
                    
                    k=0
                    
                    st.session_state[output_data] = df33
                    st.session_state[output_model] = model
                    
                    
                    st.write("")
                    st.write("")
                    st.write("")
                    
                    
                    st.markdown('**_· Option1) 학습 완료 모델 & 학습 데이터 자동저장 완료 → Stage2로 넘어가기_**')
                    st.caption("**_※ [F5키] 누르지 마세요! 자동 저장된 데이터가 초기화 됩니다._**") 
                    st.write("")
                    st.write("")
                    
                    st.markdown('**_· Option2) 학습 완료 모델(.pkl) & 학습 데이터(.xlsx) 저장하기_**')
                    #st_pandas_to_csv_download_link(df33, file_name = "01_Train_data.csv")
                    download_data_xlsx(df33, num_y)
                    download_model(k,model)
                    st.caption("**_※ 저장 폴더 지정 방법 : 마우스 우클릭 → [다른 이름으로 링크 저장]_**") 
                    
                    st.write("")
                    st.write("")
                    st.write("")
                
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                    
    
    
    
