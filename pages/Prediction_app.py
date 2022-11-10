
import streamlit as st
import pandas as pd
import numpy as np
import sklearn.metrics as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import base64
import random

import os


#import numpy.ndarray

#st.set_page_config(page_title='Prediction_app')


def st_pandas_to_csv_download_link(_df, file_name:str = "dataframe.csv"): 
    csv_exp = _df.to_csv(index=False)
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}" > Download Dataset (CSV) </a>'
    st.markdown(href, unsafe_allow_html=True)
    
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################

os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
    
def app(session_in):
    
    output_data = 'output' + session_in + '_1'
    output_model = 'output' + session_in + '_2'
    output_y = 'output' + session_in + '_3'
    
    aa = pd.DataFrame()

    with st.expander("Stage2. 최적 조건 예측하기 - 가이드"):
        st.markdown("**Option1) 자동 업로드**")
        st.markdown("Stage1. 완료시 학습 결과 자동 저장되어 Stage2.로 자동 연동됩니다.")
        st.markdown("**Option2) 수동 업로드**")
        st.markdown("1. Stage1.에서 저장한 두 파일(학습모델.pkl, Train데이터.xlsx)을 준비합니다.")
        st.markdown("2. Train 데이터 파일(.xlsx)을 업로드 합니다. CTP, CTQ는 학습 때와 동일하게 자동 선정됩니다.")
        st.markdown("3. 학습모델 파일(.pkl)을 업로드 합니다.")
        st.markdown("")
        st.markdown("**공통) CTP, CTQ 예측하기**")
        st.markdown("1. 공정조건(CTP)을 입력하여 품질인자(CTQ)를 예측합니다")
        st.markdown("2. 목표 품질(CTQ)를 입력하고, 공정조건(CTP) 범위를 선정해 최적 공정조건을 예측합니다.")
        st.markdown("3. 여러 케이스의 공정조건(CTP)을 나열한 파일(.csv)을 만들어 업로드 합니다. 파일에 입력한 여러 케이스 각각의 품질(CTQ)을 예측합니다.")


    st.markdown("<h2 style='text-align: left; color: black;'>Stage2. 최적 조건 예측</h2>", unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.sidebar.write("")

    st.sidebar.subheader("★모델 & 데이터 업로드 방법★")
    howtofetch = st.sidebar.radio("Select ▼", ("Option1) 학습 결과 자동 업로드", "Option2) 저장 파일 수동 업로드"))

    if howtofetch == "Option1) 학습 결과 자동 업로드":
        
        if output_data not in st.session_state:
            st.session_state[output_data] = aa
        df = st.session_state[output_data]
            
        if output_model not in st.session_state:
            st.session_state[output_model] = aa
        model = st.session_state[output_model]

        if output_y not in st.session_state:
            st.session_state[output_y] = aa
        outputY = st.session_state[output_y]
        outputY2 = -1 * outputY
        
        if len(df)==0:
            st.error("저장된 파일이 없습니다!!")
            st.markdown("**_※ Stage1. 을 먼저 진행하세요._**")
            st.markdown("**_※ Stage1. [5. 모델 최적화] 완료 시 결과가 자동저장되며, 이후 다시 본 절차를 실행해 주세요._**")
            st.markdown("**_※ [F5]키 누르면 앞 단계에서 저장한 데이터가 모두 초기화됩니다._**")
            
        else:
            
            #st.write(df)#확인용
            #st.write(model)#확인용
    
            st.subheader('**1. 업로드된 학습 데이터**')
            st.write('')
            x = list(df.columns)
            
            st.sidebar.write("")
            st.sidebar.write("")
            st.sidebar.header('1. 학습 데이터(.csv) 업로드')
            st.sidebar.markdown("_**- 자동 업로드 완료**_")
            
            st.sidebar.write("")
            Selected_X = st.sidebar.multiselect('공정인자(CTP) 선택하기', x, x[:outputY2])
            
            y = [a for a in x if a not in Selected_X]
            
            Selected_y = st.sidebar.multiselect('품질인자(CTQ) 선택하기', y, y)
        
            Selected_X = np.array(Selected_X)
            Selected_y = np.array(Selected_y)
            
             
            st.write('**1.1 CTP 개수 :**',Selected_X.shape[0])
            st.info(list(Selected_X))
            st.write('')
        
            st.write('**1.2 CTQ 개수:**',Selected_y.shape[0])
            st.info(list(Selected_y))
        
            df2 = pd.concat([df[Selected_X],df[Selected_y]], axis=1)
            #df2 = df[df.columns == Selected_X]
        
            #Selected_xy = np.array((Selected_X,Selected_y))
            
            st.write('')
            st.write('')
            st.write('')
            st.write('')    
            st.write('')    
            
            st.sidebar.write("")
            st.sidebar.write("")
            st.sidebar.write("")
            
            st.sidebar.header('2. 학습모델(.pkl) 업로드')
            st.sidebar.markdown("**_- 자동 업로드 완료_**")
            
            st.subheader('**2. 업로드한 머신러닝 모델**')
            st.write('')        
    
            st.markdown('**2.1 학습된 머신러닝 모델 :**')
            st.write(model)
            st.write('')
            st.write('')
            st.write('')
            
            st.markdown('**2.2 학습 모델 정확도 :**')
            X_train = df[Selected_X]
            y_train = df[Selected_y]
            
            scaler_y = StandardScaler().fit(y_train)
            predictions = model.predict(X_train)
        
            results = []
    
            msg = []
            mean = []
            std = []        
                
            kfold = KFold(n_splits=5, random_state=7, shuffle=True)
            cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
            
            for i, element in enumerate(cv_results):
                if element <= 0.0:
                    cv_results[i] = 0.0
                    
                    
            results.append(cv_results)
            #    names.append(name)
            msg.append('%s' % model)
            mean.append('%f' %  (cv_results.mean()))
            std.append('%f' % (cv_results.std()))
                    
                    
            F_result3 = pd.DataFrame(np.transpose(msg))
            F_result3.columns = ['Machine_Learning_Model']
            F_result3['R2_Mean'] = pd.DataFrame(np.transpose(mean))
            F_result3['R2_Std'] = pd.DataFrame(np.transpose(std))
            
            #st.write(F_result3)    
    
            st.write('모델 정확도 ($R^2$):')
            
            R2_mean = list(F_result3['R2_Mean'].values)
            st.info( R2_mean[0] )
                
            st.write('모델 정확도 편차 (Standard Deviation):')
            
            R2_std = list(F_result3['R2_Std'].values)
            st.info( R2_std[0])
                
            
            
            
                        
            
            df2 = df[Selected_X]
            columns = df2.columns
    
            test = []
            name = []
            #st.sidebar.write('3.1 Predict Single Condition')
            
    
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    
            st.sidebar.write("")
            st.sidebar.write("")
            st.sidebar.write("")
            
            if Selected_y.shape[0] <= 1:
                st.sidebar.header('3. CTP, CTQ 예측')
                st.sidebar.write('**3.1 품질인자(CTQ) 예측**')
                
                #st.subheader('**3. Model Prediction **')
                #st.write('**3.1 Single Condition Prediction :**')
    
                for column in columns:
                        value = st.sidebar.number_input(column, None, None, df2[column].mean(),format="%.2f") #int(df2[column].mean()))
                
                        name.append(column)
                        test.append(value)
                
     
    
                
                st.write('')
                st.write('')
                st.write('')        
                st.write('')        
                st.write('')        
                              
                st.subheader('**3. CTP, CTQ 예측**')
     
                st.write('')        
                st.write('**3.1 품질인자(CTQ) 예측 - 단일 케이스**')
                st.markdown("<h6 style='text-align: left; color: #0e4194;'> * 사이드바_3.1 에서 입력한 공정조건(CTP)에 대한 품질 결과(CTQ)를 예측합니다. </h6>", unsafe_allow_html=True)
                st.write('')        
    
                
                F_result = pd.DataFrame(name)
                F_result.columns = ['X Variables']
                F_result['Value'] = pd.DataFrame(test)
    
                
                #
     
            #para = st.sidebar.slider(para,mi,ma,cu,inter)
                #st.write('')
            
            
        
                #st.write(F_result)
                #scaler = StandardScaler().fit(X_train)
            
                if st.sidebar.button('CTQ 예측하기'):
            
            #st.write(F_result)
                    F_result = F_result.set_index('X Variables')
                    F_result_T = F_result.T
    
                    #rescaled2 = scaler.transform(F_result_T)
        
                    predictions = model.predict(F_result_T)
        
                    st.markdown("_- 입력한 공정조건(CTP) 확인_")
                    st.write(F_result)                    
    
                    st.write('**▶ CTQ [', Selected_y[0],"]의 예측 결과는  :**" , predictions[0])
                    
                
     
                
                
                
                st.sidebar.write('')
                st.sidebar.write('')    
                st.sidebar.write('**3.2 최적 공정조건(CTP) 예측**')
                
                ctq1 = df[Selected_y].mean()
                ctq2 = ctq1[0]
    
                
            #Target = st.sidebar.number_input(list(Selected_y)[0], 0.00, 1000.00, 0.00,format="%.2f")
                st.sidebar.write('**Step1) CTQ 목표 수치는?**')
                N_sample = 0
                Target = st.sidebar.number_input(Selected_y[0], None, None, ctq2,format="%.2f")
            
                st.sidebar.write("")    
                st.sidebar.write('**Step2) 생성할 샘플 개수는?**')
                N_sample = st.sidebar.number_input("샘플 개수",0, 1000, 50, format="%d")
                
                name2=[]
                test2=[]
                count = 0
                
                st.sidebar.write("")    
                st.sidebar.write('**Step3) CTP 범위 선정하기**')
                
                for column in columns:
                
                    max1 = round(float(df[column].max()),3)
                    min1 = round(float(df[column].min()),3)
                
                    #rag1 = round(min1+((max1-min1)*0.1),3)
                    #rag2 = round(min1+((max1-min1)*0.9),3)
                
                    step = round((max1-min1)/20.0,3)
                
                    value = st.sidebar.slider(column, min1, max1, (min1,max1), step)
                         
                    name2.append(column)
                    test2.append(value)
                #param2.append(para_range)
                #st.write(min1,rag1,rag2,max1)
                #st.write(column)
                #st.write(test2)
                st.write('')        
                st.write('')        
                st.write('')
                st.write('**3.2 최적 공정조건(CTP) 예측**')
                st.markdown("<h6 style='text-align: left; color: #0e4194;'> * 사이드바_3.2 에서 입력한 CTQ 목표에 대한 최적 공정조건(CTP)을 예측합니다. </h6>", unsafe_allow_html=True)
        
                if st.sidebar.button('CTP 예측하기',key = count): 
                
                    para = []
                    para2 = []
                    para4 = []
                
                    #st.write(test2)
                    import itertools
                
                    for para in test2:
                        if para[0] == para[1]:
                            para = itertools.repeat(para[0],100)
                        else:
                            para = np.arange(round(para[0],6), round(para[1]+((para[1]-para[0])/100.0),6), round((para[1]-para[0])/100.0,6))
                    #st.write(para)
                        para2.append(para)
                
               
                    Iter = N_sample
                
                    para2 = pd.DataFrame(para2)
                    para2 = para2.T
                    para2 = para2.dropna().reset_index()
                    para2.drop(['index'],axis=1,inplace=True)
                
                    Iter2 = para2.shape[1]
                
                
    
                #st.write(Iter,Iter2)
                    for i in range(Iter):
                        para3 = []
                        para5 = []
                        for j in range(Iter2):
                            #st.write(i,j,list(para2[j]))
                            para3.append(random.sample(list(para2[j]),1))
                            para5.append(para3[j][0])
                        
                    #para3 = pd.DataFrame(para3).values
                    #para4.append(para3)
                        para4.append(para5)
                    
                    
                #para4 = pd.DataFrame(para4)
                    para4 = pd.DataFrame(para4)
                
                
                    para4.columns=list(Selected_X)
                
                    #st.write('_생성된 샘플 공정조건의 결과 :_')
                    #st.write(para4)
                
                    datafile = para4
    
                    #rescaled = scaler.transform(datafile)
            
                    predictions2 = model.predict(datafile)
    
                    para4['predicted results'] = predictions2
                
                #st.write(para4)
                
                    para4.sort_values(by='predicted results', ascending=True, inplace =True)
                
                    para4 = para4.reset_index()
                    para4.drop(['index'],axis=1,inplace=True)
                
                #st.write(para4)
                
                    def find_nearest(array, value):
                        array = np.asarray(array)
                        idx = (np.abs(array - value)).argmin()
                        return array[idx]
                
                    opt_result = find_nearest(para4['predicted results'],Target)
                
                
                #st.write(opt_result)
    
                    st.write('')
                    st.markdown("_- 입력한 목표 수치 : %.2f_" %Target)
                    st.markdown("_- 입력한 샘플 개수 : %d_" %N_sample)
                    st.write('')               
                
                    df_max = para4[para4['predicted results']==para4['predicted results'].max()]
                    df_min = para4[para4['predicted results']==para4['predicted results'].min()]
                    df_opt = para4[para4['predicted results']==opt_result]
                    st.write('**▶ 최적 공정조건(CTP) :**')
                    st.write(df_opt)
                    st.write("")
                    st.write('**▶ 최대 CTQ 공정조건 :**')
                    st.write(df_max)
                    st.write("")
                    st.write('**▶ 최소 CTQ 공정조건 :**')
                    st.write(df_min)
                    #st.info(list(Selected_X2))
                    st.write("")
                    st.write('**▶ 전체 샘플 예측 결과 :**')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.write(para4)
                    st.write('')
         
                    #sample_list_1 = para4
                    #del sample_list_1["predicted results"]
                    #st.write(sample_list_1)
    
    
                    opt_ctq_best1 = df_opt['predicted results']
    
           
                    if len(opt_ctq_best1.index) == 1:
                        
                        plt.scatter(para4.index, para4.iloc[:,-1],color='#0e4194')
                        plt.scatter(opt_ctq_best1.index, opt_ctq_best1.iloc[0], color='#e30613')
                        sns.lineplot(x=para4.index, y=Target, color='red')
                        
                        st.pyplot()
                        
                        
                    else:
                        list1 = opt_ctq_best1.index
                        list1 = np.mean(list1)
                    
                        plt.scatter(para4.index, para4.iloc[:,-1],color='#0e4194')
                        plt.scatter(list1, opt_ctq_best1.iloc[0], color='#e30613')
                        sns.lineplot(x=para4.index, y=Target, color='red')
                        
                        st.pyplot()
                        
                
                
                    count +=1
    
                    st.write('')
                    st.write('')
                    st.write('')
                    st.markdown('**▶ 예측 결과 다운로드**')
        
                    st_pandas_to_csv_download_link(para4, file_name = "Predicted_CTPs_Results.csv")
                    st.caption("**_※ 저장 폴더 지정 방법 : 마우스 우클릭 → [다른 이름으로 링크 저장]_**")
                    st.write('')
                    st.write('')
    
    
            
        
        
                st.sidebar.write('')
                st.sidebar.write('')    
            
        
                st.sidebar.write("**3.3 품질인자(CTQ) 예측 (Multi-case)**")
                uploaded_file3 = st.sidebar.file_uploader("Multi-case 데이터(.csv) 업로드", type=["csv"])
                if uploaded_file3 is not None:
                    def load_csv():
                        csv = pd.read_csv(uploaded_file3)
                        return csv
                
    
    
    
        
                count +=1
                st.write('')        
                st.write('')        
                st.write('')        
                st.write('**3.3 품질인자(CTQ) 예측 (Multi-case)**')
                
                st.markdown("<h6 style='text-align: left; color: #0e4194;'> * 사이드바_3.3 에서 업로드한 다양한 공정조건 케이스(CTP)에 대한 품질 결과(CTQ)를 예측합니다. </h6>", unsafe_allow_html=True)
            
                if st.sidebar.button('CTQ 예측하기',key = count):
                    df3 = load_csv()            
                    datafile = df3
    
                    #rescaled = scaler.transform(datafile)
                    
                    predictions2 = model.predict(datafile)
    
                    df3['predicted results'] = predictions2
    
                    
                
                    df_max = df3[df3['predicted results']==df3['predicted results'].max()]
                    df_min = df3[df3['predicted results']==df3['predicted results'].min()]
                    st.write("")
                    st.write('**▶ 최대 CTQ 공정조건 :**')
                    st.write(df_max)
                    st.write('**▶ 최소 CTQ 공정조건 :**')
                    st.write(df_min)
                    #st.info(list(Selected_X2))
                
                    st.write('')
                    st.write('**▶ 전체 샘플 예측 결과 :**')
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.write(df3)
                    sns.scatterplot(x=df3.index,y=df3.iloc[:,-1],s=30,color='#e30613')
                    st.write('')
                    st.pyplot()
                    count +=1
        
                    st.write('')
                    st.write('')
                    st.write('')
                    st.markdown('**▶ 예측 결과 다운로드 (Multi-case)**')
        
                    st_pandas_to_csv_download_link(df3, file_name = "Predicted_CTQs_Results.csv")
                    st.caption("**_※ 저장 폴더 지정 방법 : 마우스 우클릭 → [다른 이름으로 링크 저장]_**")
                    
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
    
    
                
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
    
            #st.write(mi)
        #parameter_n_neighbors = st.sidebar.slider('Number of neighbers', 2, 10, (1,6), 2)  
    
        
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    
    
        
            else :
                
                df_y = df[Selected_y]
                
                st.sidebar.header('3. CTP, CTQ 예측')
                st.sidebar.write('**3.1 품질인자(CTQ) 예측**')
                
                #st.subheader('**3. Model Prediction **')
                #st.write('**3.1 Single Condition Prediction :**')
    
                for column in columns:
                        value = st.sidebar.number_input(column, None, None, df2[column].mean(),format="%.2f") #int(df2[column].mean()))
                
                        name.append(column)
                        test.append(value)
                
     
    
                
                st.write('')
                st.write('')
                st.write('')        
                st.write('')        
                st.write('')        
                              
                st.subheader('**3. CTP, CTQ 예측**')
                st.write('')        
                st.write('**3.1 품질인자(CTQ) 예측 - 단일 케이스**')
                st.markdown("<h6 style='text-align: left; color: #0e4194;'> * 사이드바_3.1 에서 입력한 공정조건(CTP)에 대한 품질 결과(CTQ)를 예측합니다. </h6>", unsafe_allow_html=True)
                F_result = pd.DataFrame(name)
                F_result.columns = ['X Variables']
                F_result['Value'] = pd.DataFrame(test)
                
                
                
                #st.write(F_result)
     
            #para = st.sidebar.slider(para,mi,ma,cu,inter)
                st.write('')
            
            
        
                
                
            
                if st.sidebar.button('CTQ 예측하기'):
            
            #st.write(F_result)
                    F_result = F_result.set_index('X Variables')
                    F_result_T = F_result.T
    
      
                    predictions = model.predict(F_result_T)
        
    
                    #predictions = pd.DataFrame(predictions[0],columns=['Value'])
                    #st.write(predictions)
                    
                    predictions2 = pd.DataFrame()
                    predictions2['Y Variable'] = df[Selected_y].columns
                    predictions2['Value'] = pd.DataFrame(predictions[0])
                    
                    
                    #Selected_y_list = Selected_y.tolist()
                    #st.write(Selected_y_list)
                    
                    st.write("_입력한 공정조건(CTP) 확인_")
                    st.write(F_result)
                    
                    st.write("**▶ CTQ(Y variables)의 예측 결과는 :**")
                    st.write(predictions2)
                    
                    
                
                
                
                st.sidebar.write('')
                st.sidebar.write('')    
                st.sidebar.write('**3.2 최적 공정조건(CTP) 예측**')
            
            #Target = st.sidebar.number_input(list(Selected_y)[0], 0.00, 1000.00, 0.00,format="%.2f")
                st.sidebar.write('**Step1) CTQ 목표 수치는?**')
                
                sample_target = df[Selected_y].shape[1]
                                
                Target = []
                
    
                for i in range(sample_target):
    
                    Sel_y_mean = df[Selected_y[i]]
                    Sel_y_mean2 = Sel_y_mean.mean()
     
                    Target.append(st.sidebar.number_input(Selected_y[i], None, None, Sel_y_mean2, format="%.2f"))
            
            
            
                N_sample = 0
                st.sidebar.write('')
                st.sidebar.write('**Step2) 생성할 샘플 개수는?**')
                N_sample = st.sidebar.number_input('샘플 개수',0,1000,50, format="%d")
                
            
                
                name2=[]
                test2=[]
                count = 0
                
                st.sidebar.write('')
                st.sidebar.write('**Step3) CTP 범위 선정하기**')
                for column in columns:
                
                    max1 = round(float(df[column].max()),3)
                    min1 = round(float(df[column].min()),3)
                    
                    
                    #rag1 = round(min1+((max1-min1)*0.1),3)
                    #rag2 = round(min1+((max1-min1)*0.9),3)
                    step = round((max1-min1)/20.0,3)
                    value = st.sidebar.slider(column, min1, max1, (min1,max1), step)
                    name2.append(column)
                    test2.append(value)
                    
                st.write('')
                st.write('')        
                st.write('')        
                st.write('**3.2 최적 공정조건(CTP) 예측**')
                st.markdown("<h6 style='text-align: left; color: #0e4194;'> * 사이드바_3.2 에서 입력한 CTQ 목표에 대한 최적 공정조건(CTP)을 예측합니다. </h6>", unsafe_allow_html=True)
                
                if st.sidebar.button('CTP 예측하기',key = count): 
                
                    para = []
                    para2 = []
                    para4 = []
                
                    #st.write(test2)
                    import itertools
                
                    for para in test2:
                        if para[0] == para[1]:
                            para = itertools.repeat(para[0],100)
                        else:
                            para = np.arange(round(para[0],6), round(para[1]+((para[1]-para[0])/100.0),6), round((para[1]-para[0])/100.0,6))
                    #st.write(para)
                        para2.append(para)
                
               
                    Iter = N_sample
                
                    para2 = pd.DataFrame(para2)
                    para2 = para2.T
                    para2 = para2.dropna().reset_index()
                    para2.drop(['index'],axis=1,inplace=True)
                
                    Iter2 = para2.shape[1]
                
                
    
                #st.write(Iter,Iter2)
                    for i in range(Iter):
                        para3 = []
                        para5 = []
                        for j in range(Iter2):
                            #st.write(i,j,list(para2[j]))
                            para3.append(random.sample(list(para2[j]),1))
                            para5.append(para3[j][0])
                        
                    #para3 = pd.DataFrame(para3).values
                    #para4.append(para3)
                        para4.append(para5)
                    
                    
                #para4 = pd.DataFrame(para4)
                    para4 = pd.DataFrame(para4)
                
                
                    para4.columns=list(Selected_X)
                    st.write('')
    
                    st.write('**▶ 생성된 공정조건(CTP) 샘플 :**')
    
                    st.write(para4)
                    
                    datafile = para4
              
                    predictions2 = model.predict(datafile)
                    
                    predictions2 = pd.DataFrame(predictions2,columns=df[Selected_y].columns)
                
                    
                    para4 = pd.concat([para4, predictions2],axis=1)
                        
    
                
                    #para4.sort_values(by='predicted results', ascending=True, inplace =True)
                
                    para4 = para4.reset_index()
                    para4.drop(['index'],axis=1,inplace=True)
                
                    predictions3 = scaler_y.transform(predictions2)                
                    
                    Target = pd.DataFrame(Target)
                    Target = Target.T
                    Target2 = scaler_y.transform(Target)
                    
                    Target2 = pd.DataFrame(Target2)
                    Target3 = []
                    
                    for i in range(Iter):
                        Target3.append(Target2.values)
                    
                    Target3 = np.reshape(Target3, (Iter, Selected_y.shape[0]))
                    Target3 = pd.DataFrame(Target3)
                    
                    Diff3 = abs(predictions3 - Target3)
    
                    #################제곱넣기################
                    Diff4 = Diff3 * Diff3
                    Diff5 = Diff4.sum(axis=1)
                    #Diff5.sort_values(ascending=True, inplace =True)
                    #para4['sum'] = Diff5.sum(axis=1)                   
                    para4['sum'] = Diff5
                    para4.sort_values(by='sum', ascending=True, inplace =True)
    
    
                   
                    """
                    def find_nearest(array, value):
                        
                        array = np.asarray(array)
                        idx = (np.abs(array - value)).argmin()
                        return array[idx]
                
                    opt = predictions2.shape[1]
                    
                    opt_result = []
                    for i in range(opt):
                        
                        opt_result.append(find_nearest(predictions2.iloc[:,i],Target[i]))
      
    
                    st.write(opt_result)
                    st.write('')
                    st.write('')
                    st.write('')
                    st.markdown("<h6 style='text-align: left; color: #0e4194;'> * Optimizing Condition Results </h6>", unsafe_allow_html=True)
                
                
                    #df_max = para4[para4['predicted results']==para4['predicted results'].max()]
                    #df_min = para4[para4['predicted results']==para4['predicted results'].min()]
                    df_opt = pd.DataFrame()
                    
                    for i in range(opt):   
                        for column in df[Selected_y].columns:
                            df_opt = pd.concat([df_opt, para4[para4[column]==opt_result[i]]],axis=0)
                    """        
                    
                    st.write('**▶ 최적 공정조건(CTP) :**')
                    opt = para4.drop(['sum'],axis=1)
                    st.write(opt.head(5))
                    
    
                    st.write('')
    
                    st.write('**▶ 전체 샘플 예측 결과 :**')
                    st.write(opt)
                    
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    
                    Target5 = []
                    for i in range(Iter):
                        Target5.append(Target.values)
                    Target5 = np.reshape(Target5, (Iter, Selected_y.shape[0]))
                    Target5 = pd.DataFrame(Target5)
                    
                    plt.figure(figsize=(10,6))
                    fig, axs = plt.subplots(ncols=df[Selected_y].shape[1])
                    fig.subplots_adjust(hspace=1)
                
                    opt_best5 = opt.head(5)
                    opt_ctq_best5 = opt_best5[Selected_y]
                    
                    for i in range(1,df[Selected_y].shape[1]+1):
                        
                        plt.subplot(1,df[Selected_y].shape[1],i)
                        
                        sns.lineplot(x=para4.index, y=Target5[i-1], color='red')
                        
                        plt.scatter(para4.index, predictions2.iloc[:,i-1], color='#0e4194', label = 'Prediction')
                        plt.scatter(opt_ctq_best5.index, opt_ctq_best5.iloc[:,i-1], color='#e30613', label = 'Prediction')
                        
                        plt.title(df[Selected_y].columns[i-1],fontsize=10)
                        plt.xticks(fontsize=8)
                        plt.yticks(fontsize=8)
                    #ax.set_xlabel('Time', fontsize=16)
                    #plt.ylabel(Y.columns[i-1], fontsize=10)
                    
                    st.pyplot()
                    
    
                    count +=1
    
                    st.write('')
                    st.write('')
                    st.write('')
                    st.markdown('**▶ 예측 결과 다운로드**')
        
                    st_pandas_to_csv_download_link(para4, file_name = "Predicted_CTPs_Results.csv")
                    st.caption("**_※ 저장 폴더 지정 방법 : 마우스 우클릭 → [다른 이름으로 링크 저장]_**")
                    
                    st.write('')
                    st.write('')
           
        
        
                st.sidebar.write('')
                st.sidebar.write('')    
                st.sidebar.write('')
                st.sidebar.write('**3.3 품질인자(CTQ) 예측 (Mulit-case)**')
                uploaded_file3 = st.sidebar.file_uploader("Mulit-case 데이터(.csv) 업로드", type=["csv"])
                if uploaded_file3 is not None:
                    def load_csv():
                        csv = pd.read_csv(uploaded_file3)
                        return csv
                
    
    
    #    uploaded_file = st.file_uploader("Choose a file")
    #if uploaded_file is not None:
    #    uploaded_file.seek(0)
    #    data = pd.read_csv(uploaded_file, low_memory=False)
    #    st.write(data.shape)
        
                count +=1
                st.write('')
                st.write('')        
                st.write('')        
                st.write('**3.3 품질인자(CTQ) 예측 (Multi-case)**')
                st.markdown("<h6 style='text-align: left; color: #0e4194;'> * 사이드바_3.3 에서 업로드한 다양한 공정조건 케이스(CTP)에 대한 품질 결과(CTQ)를 예측합니다. </h6>", unsafe_allow_html=True)
                
                if st.sidebar.button('CTQ 예측하기', key = count):
                    df3 = load_csv()            
                    datafile = df3
    
                    
                    predictions2 = model.predict(datafile)
    
                    predictions2 = pd.DataFrame(predictions2,columns=df[Selected_y].columns)
    
                    #df3['predicted results'] = predictions2
                    
                    predictions3 = pd.DataFrame()
                    predictions3 = pd.concat([df3, predictions2],axis=1)
                    
                    st.write("")
                    st.write('**▶ 전체 샘플 예측 결과 :**')
                    st.write(predictions3)
    
    
                    
                
                    plt.figure(figsize=(10,6))
                    fig, axs = plt.subplots(ncols=df[Selected_y].shape[1])
                    fig.subplots_adjust(hspace=1)
                
                    for i in range(1,df[Selected_y].shape[1]+1):
                
    
                    
                        plt.subplot(1,df[Selected_y].shape[1],i)
                        
                        #sns.lineplot(x=df3.index,y=Target[i-1], color='red')
                        
                        #plt.plot(df[Selected_y].iloc[:,i-1], df[Selected_y].iloc[:,i-1], color='blue', label = 'Actual data')
                        plt.scatter(df3.index, predictions2.iloc[:,i-1], color='#e30613', label = 'Prediction')
                        plt.title(df[Selected_y].columns[i-1],fontsize=10)
                        plt.xticks(fontsize=8)
                        plt.yticks(fontsize=8)
                    #ax.set_xlabel('Time', fontsize=16)
                    #plt.ylabel(Y.columns[i-1], fontsize=10)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.pyplot()
                    
                
                    #st.write('')
                    #st.write('**Total results:**')
                    #st.set_option('deprecation.showPyplotGlobalUse', False)
                    #st.write(df3)
                    #sns.scatterplot(x=df3.index,y=df3.iloc[:,-1],s=30,color='red')
                    #st.pyplot()
                    count +=1
        
    
    
                    st.write('')
                    st.write('')
                    st.write('')
                    st.markdown('**▶ 예측 결과 다운로드 (Multi-case)**')
            
                    st_pandas_to_csv_download_link(df3, file_name = "Predicted_CTQs_Results.csv")
                    st.caption("**_※ 저장 폴더 지정 방법 : 마우스 우클릭 → [다른 이름으로 링크 저장]_**")
                        
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
    
    
                    
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")
                st.sidebar.write("")


################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################


        # Main panel        
    if howtofetch == "Option2) 저장 파일 수동 업로드":
        #---------------------------------#
        # Sidebar - Collects user input features into dataframe
        with st.sidebar.header('1. 학습 데이터(.xlsx) 업로드'):
            uploaded_file = st.sidebar.file_uploader( 'Train Data File', type=["xlsx"])
        
        if uploaded_file is not None:
            def load_dataset():
                file = pd.read_excel(uploaded_file, sheet_name = 'sheet1')
                return file
            def load_num_y():
                y = pd.read_excel(uploaded_file, sheet_name = 'sheet2')
                return y
            
            st.subheader('**1. 업로드된 학습 데이터**')
            st.write('')
            df = load_dataset()
            num_y = load_num_y()
            num_y = num_y.iloc[0,0]
            #st.write(num_y)
            
            x = list(df.columns)

            Selected_X = st.sidebar.multiselect('공정인자(CTP) 선택하기', x, x[:-num_y])
            
            y = [a for a in x if a not in Selected_X]
            
            Selected_y = st.sidebar.multiselect('품질인자(CTQ) 선택하기', y, y)

            Selected_X = np.array(Selected_X)
            Selected_y = np.array(Selected_y)
            
             
            st.write('**1.1 CTP 개수 :**',Selected_X.shape[0])
            st.info(list(Selected_X))
            st.write('')
        
            st.write('**1.2 CTQ 개수:**',Selected_y.shape[0])
            st.info(list(Selected_y))
        
            df2 = pd.concat([df[Selected_X],df[Selected_y]], axis=1)
            #df2 = df[df.columns == Selected_X]
        
            #Selected_xy = np.array((Selected_X,Selected_y))
            
            st.write('')
            st.write('')
            st.write('')
            st.write('')    
            st.write('')    
            
            st.sidebar.write("")
            st.sidebar.write("")
            st.sidebar.write("")
            with st.sidebar.header('2. 학습모델(.pkl) 업로드'):
                uploaded_file2 = st.sidebar.file_uploader("Trained model file", type=["pkl"])
                
            st.sidebar.write("")
            st.sidebar.write("")
            st.sidebar.write("")
                
            
            if uploaded_file2 is not None:
                def load_model(model):
                    loaded_model = pickle.load(model)
                    return loaded_model
                
                st.subheader('**2. 업로드한 머신러닝 모델**')
                st.write('')

                model = load_model(uploaded_file2)
                

                
                st.markdown('**2.1 학습된 머신러닝 모델 :**')
                st.write(model)
                st.write('')
                
                st.write('')
                st.write('')
                st.markdown('**2.2 학습 모델 정확도 :**')
                X_train = df[Selected_X]
                y_train = df[Selected_y]
                
                scaler_y = StandardScaler().fit(y_train)
        
                #rescaled = scaler.transform(X_train)
                
                predictions = model.predict(X_train)
            
      
            
                results = []

                msg = []
                mean = []
                std = []        

                    
                kfold = KFold(n_splits=5, random_state=7, shuffle=True)
                cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
                
                for i, element in enumerate(cv_results):
                    if element <= 0.0:
                        cv_results[i] = 0.0
                        
                        
                results.append(cv_results)
                #    names.append(name)
                msg.append('%s' % model)
                mean.append('%f' %  (cv_results.mean()))
                std.append('%f' % (cv_results.std()))
                        
                        
                F_result3 = pd.DataFrame(np.transpose(msg))
                F_result3.columns = ['Machine_Learning_Model']
                F_result3['R2_Mean'] = pd.DataFrame(np.transpose(mean))
                F_result3['R2_Std'] = pd.DataFrame(np.transpose(std))
                
                #st.write(F_result3)    

                st.write('모델 정확도 ($R^2$):')
                
                R2_mean = list(F_result3['R2_Mean'].values)
                st.info( R2_mean[0] )
                    
                st.write('모델 정확도 편차 (Standard Deviation):')
                
                R2_std = list(F_result3['R2_Std'].values)
                st.info( R2_std[0])
                    
                
                
                
                            
                
                df2 = df[Selected_X]
                columns = df2.columns
        
                test = []
                name = []
                #st.sidebar.write('3.1 Predict Single Condition')
                

    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
        
        
                if Selected_y.shape[0] <= 1:
                    st.sidebar.header('3. CTP, CTQ 예측')
                    st.sidebar.write('**3.1 품질인자(CTQ) 예측**')
                    
                    #st.subheader('**3. Model Prediction **')
                    #st.write('**3.1 Single Condition Prediction :**')

                    for column in columns:
                            value = st.sidebar.number_input(column, None, None, df2[column].mean(),format="%.2f") #int(df2[column].mean()))
                    
                            name.append(column)
                            test.append(value)
                    
     

                    
                    st.write('')
                    st.write('')
                    st.write('')        
                    st.write('')        
                    st.write('')        
                                  
                    st.subheader('**3. CTP, CTQ 예측**')
         
                    st.write('')        
                    st.write('**3.1 품질인자(CTQ) 예측 - 단일 케이스**')
                    st.markdown("<h6 style='text-align: left; color: #0e4194;'> * 사이드바_3.1 에서 입력한 공정조건(CTP)에 대한 품질 결과(CTQ)를 예측합니다. </h6>", unsafe_allow_html=True)
                    st.write('')        

                    
                    F_result = pd.DataFrame(name)
                    F_result.columns = ['X Variables']
                    F_result['Value'] = pd.DataFrame(test)

                    
                    #
         
                #para = st.sidebar.slider(para,mi,ma,cu,inter)
                    #st.write('')
                
                
            
                    #st.write(F_result)
                    #scaler = StandardScaler().fit(X_train)
                
                    if st.sidebar.button('CTQ 예측하기'):
                
                #st.write(F_result)
                        F_result = F_result.set_index('X Variables')
                        F_result_T = F_result.T
        
                        #rescaled2 = scaler.transform(F_result_T)
            
                        predictions = model.predict(F_result_T)
            
                        st.markdown("_- 입력한 공정조건(CTP) 확인_")
                        st.write(F_result)                    
        
                        st.write('**▶ CTQ [', Selected_y[0],"]의 예측 결과는  :**" , predictions[0])
                        
                    
     
                    
                    
                    
                    st.sidebar.write('')
                    st.sidebar.write('')    
                    st.sidebar.write('**3.2 최적 공정조건(CTP) 예측**')
                    
                    ctq1 = df[Selected_y].mean()
                    ctq2 = ctq1[0]

                    
                #Target = st.sidebar.number_input(list(Selected_y)[0], 0.00, 1000.00, 0.00,format="%.2f")
                    st.sidebar.write('**Step1) CTQ 목표 수치는?**')
                    N_sample = 0
                    Target = st.sidebar.number_input(Selected_y[0], None, None, ctq2,format="%.2f")
                
                    st.sidebar.write("")    
                    st.sidebar.write('**Step2) 생성할 샘플 개수는?**')
                    N_sample = st.sidebar.number_input("샘플 개수",0, 1000, 50, format="%d")
                    
                    name2=[]
                    test2=[]
                    count = 0
                    
                    st.sidebar.write("")    
                    st.sidebar.write('**Step3) CTP 범위 선정하기**')
                    
                    for column in columns:
                    
                        max1 = round(float(df[column].max()),3)
                        min1 = round(float(df[column].min()),3)
                    
                        #rag1 = round(min1+((max1-min1)*0.1),3)
                        #rag2 = round(min1+((max1-min1)*0.9),3)
                    
                        step = round((max1-min1)/20.0,3)
                    
                        value = st.sidebar.slider(column, min1, max1, (min1,max1), step)
                             
                        name2.append(column)
                        test2.append(value)
                    #param2.append(para_range)
                    #st.write(min1,rag1,rag2,max1)
                    #st.write(column)
                    #st.write(test2)
                    st.write('')        
                    st.write('')        
                    st.write('')
                    st.write('**3.2 최적 공정조건(CTP) 예측**')
                    st.markdown("<h6 style='text-align: left; color: #0e4194;'> * 사이드바_3.2 에서 입력한 CTQ 목표에 대한 최적 공정조건(CTP)을 예측합니다. </h6>", unsafe_allow_html=True)
            
                    if st.sidebar.button('CTP 예측하기',key = count): 
                    
                        para = []
                        para2 = []
                        para4 = []
                    
                        #st.write(test2)
                        import itertools
                    
                        for para in test2:
                            if para[0] == para[1]:
                                para = itertools.repeat(para[0],100)
                            else:
                                para = np.arange(round(para[0],6), round(para[1]+((para[1]-para[0])/100.0),6), round((para[1]-para[0])/100.0,6))
                        #st.write(para)
                            para2.append(para)
                    
                   
                        Iter = N_sample
                    
                        para2 = pd.DataFrame(para2)
                        para2 = para2.T
                        para2 = para2.dropna().reset_index()
                        para2.drop(['index'],axis=1,inplace=True)
                    
                        Iter2 = para2.shape[1]
                    
                    

                    #st.write(Iter,Iter2)
                        for i in range(Iter):
                            para3 = []
                            para5 = []
                            for j in range(Iter2):
                                #st.write(i,j,list(para2[j]))
                                para3.append(random.sample(list(para2[j]),1))
                                para5.append(para3[j][0])
                            
                        #para3 = pd.DataFrame(para3).values
                        #para4.append(para3)
                            para4.append(para5)
                        
                        
                    #para4 = pd.DataFrame(para4)
                        para4 = pd.DataFrame(para4)
                    
                    
                        para4.columns=list(Selected_X)
                    
                        #st.write('_생성된 샘플 공정조건의 결과 :_')
                        #st.write(para4)
                    
                        datafile = para4
        
                        #rescaled = scaler.transform(datafile)
                
                        predictions2 = model.predict(datafile)
        
                        para4['predicted results'] = predictions2
                    
                    #st.write(para4)
                    
                        para4.sort_values(by='predicted results', ascending=True, inplace =True)
                    
                        para4 = para4.reset_index()
                        para4.drop(['index'],axis=1,inplace=True)
                    
                    #st.write(para4)
                    
                        def find_nearest(array, value):
                            array = np.asarray(array)
                            idx = (np.abs(array - value)).argmin()
                            return array[idx]
                    
                        opt_result = find_nearest(para4['predicted results'],Target)
                    
                    
                    #st.write(opt_result)
        
                        st.write('')
                        st.markdown("_- 입력한 목표 수치 : %.2f_" %Target)
                        st.markdown("_- 입력한 샘플 개수 : %d_" %N_sample)
                        st.write('')               
                    
                        df_max = para4[para4['predicted results']==para4['predicted results'].max()]
                        df_min = para4[para4['predicted results']==para4['predicted results'].min()]
                        df_opt = para4[para4['predicted results']==opt_result]
                        st.write('**▶ 최적 공정조건(CTP) :**')
                        st.write(df_opt)
                        st.write("")
                        st.write('**▶ 최대 CTQ 공정조건 :**')
                        st.write(df_max)
                        st.write("")
                        st.write('**▶ 최소 CTQ 공정조건 :**')
                        st.write(df_min)
                        #st.info(list(Selected_X2))
                        st.write("")
                        st.write('**▶ 전체 샘플 예측 결과 :**')
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.write(para4)
                        st.write('')
             
                        #sample_list_1 = para4
                        #del sample_list_1["predicted results"]
                        #st.write(sample_list_1)


                        opt_ctq_best1 = df_opt['predicted results']

               
                        '''
                        ax = sns.scatterplot(x=para4.index,y=para4.iloc[:,-1],s=30,color='#0e4194')
                        #ax1 = sns.scatterplot(x=opt_ctq_best1.index,y=opt_ctq_best1.iloc[:,-1],s=30,color='#e30613')
                        sns.lineplot(x=para4.index,y=Target,ax=ax.axes, color='red')
                        '''
                        plt.scatter(para4.index, para4.iloc[:,-1],color='#0e4194')
                        plt.scatter(opt_ctq_best1.index, opt_ctq_best1.iloc[0], color='#e30613')
                        sns.lineplot(x=para4.index, y=Target, color='red')
                        
                        st.pyplot()
                    
                    
                        count +=1

                        st.write('')
                        st.write('')
                        st.write('')
                        st.markdown('**▶ 예측 결과 다운로드**')
            
                        st_pandas_to_csv_download_link(para4, file_name = "Predicted_CTPs_Results.csv")
                        st.caption("**_※ 저장 폴더 지정 방법 : 마우스 우클릭 → [다른 이름으로 링크 저장]_**")
                        st.write('')
                        st.write('')


                
            
            
                    st.sidebar.write('')
                    st.sidebar.write('')    
                
            
                    st.sidebar.write("**3.3 품질인자(CTQ) 예측 (Multi-case)**")
                    uploaded_file3 = st.sidebar.file_uploader("Multi-case 데이터(.csv) 업로드", type=["csv"])
                    if uploaded_file3 is not None:
                        def load_csv():
                            csv = pd.read_csv(uploaded_file3)
                            return csv
                    
        
        

            
                    count +=1
                    st.write('')        
                    st.write('')        
                    st.write('')        
                    st.write('**3.3 품질인자(CTQ) 예측 (Multi-case)**')
                    
                    st.markdown("<h6 style='text-align: left; color: #0e4194;'> * 사이드바_3.3 에서 업로드한 다양한 공정조건 케이스(CTP)에 대한 품질 결과(CTQ)를 예측합니다. </h6>", unsafe_allow_html=True)
                
                    if st.sidebar.button('CTQ 예측하기',key = count):
                        df3 = load_csv()            
                        datafile = df3
        
                        #rescaled = scaler.transform(datafile)
                        
                        predictions2 = model.predict(datafile)
        
                        df3['predicted results'] = predictions2
        
                        
                    
                        df_max = df3[df3['predicted results']==df3['predicted results'].max()]
                        df_min = df3[df3['predicted results']==df3['predicted results'].min()]
                        st.write("")
                        st.write('**▶ 최대 CTQ 공정조건 :**')
                        st.write(df_max)
                        st.write('**▶ 최소 CTQ 공정조건 :**')
                        st.write(df_min)
                        #st.info(list(Selected_X2))
                    
                        st.write('')
                        st.write('**▶ 전체 샘플 예측 결과 :**')
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.write(df3)
                        sns.scatterplot(x=df3.index,y=df3.iloc[:,-1],s=30,color='#e30613')
                        st.write('')
                        st.pyplot()
                        count +=1
            
                        st.write('')
                        st.write('')
                        st.write('')
                        st.markdown('**▶ 예측 결과 다운로드 (Multi-case)**')
            
                        st_pandas_to_csv_download_link(df3, file_name = "Predicted_CTQs_Results.csv")
                        st.caption("**_※ 저장 폴더 지정 방법 : 마우스 우클릭 → [다른 이름으로 링크 저장]_**")
                        
                        st.write('')
                        st.write('')
                        st.write('')
                        st.write('')
                        st.write('')


                    
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")

                #st.write(mi)
            #parameter_n_neighbors = st.sidebar.slider('Number of neighbers', 2, 10, (1,6), 2)  

            
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################
    ################################################################################################################################################################################################


            
                else :
                    
                    df_y = df[Selected_y]
                    
                    st.sidebar.header('3. CTP, CTQ 예측')
                    st.sidebar.write('**3.1 품질인자(CTQ) 예측**')
                    
                    #st.subheader('**3. Model Prediction **')
                    #st.write('**3.1 Single Condition Prediction :**')

                    for column in columns:
                            value = st.sidebar.number_input(column, None, None, df2[column].mean(),format="%.2f") #int(df2[column].mean()))
                    
                            name.append(column)
                            test.append(value)
                    
     

                    
                    st.write('')
                    st.write('')
                    st.write('')        
                    st.write('')        
                    st.write('')        
                                  
                    st.subheader('**3. CTP, CTQ 예측**')
                    st.write('')        
                    st.write('**3.1 품질인자(CTQ) 예측 - 단일 케이스**')
                    st.markdown("<h6 style='text-align: left; color: #0e4194;'> * 사이드바_3.1 에서 입력한 공정조건(CTP)에 대한 품질 결과(CTQ)를 예측합니다. </h6>", unsafe_allow_html=True)
                    F_result = pd.DataFrame(name)
                    F_result.columns = ['X Variables']
                    F_result['Value'] = pd.DataFrame(test)
                    
                    
                    
                    #st.write(F_result)
         
                #para = st.sidebar.slider(para,mi,ma,cu,inter)
                    st.write('')
                
                
            
                    
                    
                
                    if st.sidebar.button('CTQ 예측하기'):
                
                #st.write(F_result)
                        F_result = F_result.set_index('X Variables')
                        F_result_T = F_result.T
        
          
                        predictions = model.predict(F_result_T)
            

                        #predictions = pd.DataFrame(predictions[0],columns=['Value'])
                        #st.write(predictions)
                        
                        predictions2 = pd.DataFrame()
                        predictions2['Y Variable'] = df[Selected_y].columns
                        predictions2['Value'] = pd.DataFrame(predictions[0])
                        
                        
                        #Selected_y_list = Selected_y.tolist()
                        #st.write(Selected_y_list)
                        
                        st.write("_입력한 공정조건(CTP) 확인_")
                        st.write(F_result)
                        
                        st.write("**▶ CTQ(Y variables)의 예측 결과는 :**")
                        st.write(predictions2)
                        
                        
                    
                    
                    
                    st.sidebar.write('')
                    st.sidebar.write('')    
                    st.sidebar.write('**3.2 최적 공정조건(CTP) 예측**')
                
                #Target = st.sidebar.number_input(list(Selected_y)[0], 0.00, 1000.00, 0.00,format="%.2f")
                    st.sidebar.write('**Step1) CTQ 목표 수치는?**')
                    
                    sample_target = df[Selected_y].shape[1]
                                    
                    Target = []
                    

                    for i in range(sample_target):

                        Sel_y_mean = df[Selected_y[i]]
                        Sel_y_mean2 = Sel_y_mean.mean()
     
                        Target.append(st.sidebar.number_input(Selected_y[i], None, None, Sel_y_mean2, format="%.2f"))
                
                
                
                    N_sample = 0
                    st.sidebar.write('')
                    st.sidebar.write('**Step2) 생성할 샘플 개수는?**')
                    N_sample = st.sidebar.number_input('샘플 개수',0,1000,50, format="%d")
                    
                
                    
                    name2=[]
                    test2=[]
                    count = 0
                    
                    st.sidebar.write('')
                    st.sidebar.write('**Step3) CTP 범위 선정하기**')
                    for column in columns:
                    
                        max1 = round(float(df[column].max()),3)
                        min1 = round(float(df[column].min()),3)
                        
                        
                        #rag1 = round(min1+((max1-min1)*0.1),3)
                        #rag2 = round(min1+((max1-min1)*0.9),3)
                        step = round((max1-min1)/20.0,3)
                        value = st.sidebar.slider(column, min1, max1, (min1,max1), step)
                        name2.append(column)
                        test2.append(value)
                        
                    st.write('')
                    st.write('')        
                    st.write('')        
                    st.write('**3.2 최적 공정조건(CTP) 예측**')
                    st.markdown("<h6 style='text-align: left; color: #0e4194;'> * 사이드바_3.2 에서 입력한 CTQ 목표에 대한 최적 공정조건(CTP)을 예측합니다. </h6>", unsafe_allow_html=True)
                    
                    if st.sidebar.button('CTP 예측하기',key = count): 
                    
                        para = []
                        para2 = []
                        para4 = []
                    
                        #st.write(test2)
                        import itertools
                    
                        for para in test2:
                            if para[0] == para[1]:
                                para = itertools.repeat(para[0],100)
                            else:
                                para = np.arange(round(para[0],6), round(para[1]+((para[1]-para[0])/100.0),6), round((para[1]-para[0])/100.0,6))
                        #st.write(para)
                            para2.append(para)
                    
                   
                        Iter = N_sample
                    
                        para2 = pd.DataFrame(para2)
                        para2 = para2.T
                        para2 = para2.dropna().reset_index()
                        para2.drop(['index'],axis=1,inplace=True)
                    
                        Iter2 = para2.shape[1]
                    
                    

                    #st.write(Iter,Iter2)
                        for i in range(Iter):
                            para3 = []
                            para5 = []
                            for j in range(Iter2):
                                #st.write(i,j,list(para2[j]))
                                para3.append(random.sample(list(para2[j]),1))
                                para5.append(para3[j][0])
                            
                        #para3 = pd.DataFrame(para3).values
                        #para4.append(para3)
                            para4.append(para5)
                        
                        
                    #para4 = pd.DataFrame(para4)
                        para4 = pd.DataFrame(para4)
                    
                    
                        para4.columns=list(Selected_X)
                        st.write('')

                        st.write('**▶ 생성된 공정조건(CTP) 샘플 :**')

                        st.write(para4)
                        
                        datafile = para4
                  
                        predictions2 = model.predict(datafile)
                        
                        predictions2 = pd.DataFrame(predictions2,columns=df[Selected_y].columns)
                    
                        
                        para4 = pd.concat([para4, predictions2],axis=1)
                            

                    
                        #para4.sort_values(by='predicted results', ascending=True, inplace =True)
                    
                        para4 = para4.reset_index()
                        para4.drop(['index'],axis=1,inplace=True)
                    
                        predictions3 = scaler_y.transform(predictions2)                
                        
                        Target = pd.DataFrame(Target)
                        Target = Target.T
                        Target2 = scaler_y.transform(Target)
                        
                        Target2 = pd.DataFrame(Target2)
                        Target3 = []
                        
                        for i in range(Iter):
                            Target3.append(Target2.values)
                        
                        Target3 = np.reshape(Target3, (Iter, Selected_y.shape[0]))
                        Target3 = pd.DataFrame(Target3)
                        
                        Diff3 = abs(predictions3 - Target3)

                        #################제곱넣기################
                        Diff4 = Diff3 * Diff3
                        Diff5 = Diff4.sum(axis=1)
                        #Diff5.sort_values(ascending=True, inplace =True)
                        #para4['sum'] = Diff5.sum(axis=1)                   
                        para4['sum'] = Diff5
                        para4.sort_values(by='sum', ascending=True, inplace =True)


                       
                        """
                        def find_nearest(array, value):
                            
                            array = np.asarray(array)
                            idx = (np.abs(array - value)).argmin()
                            return array[idx]
                    
                        opt = predictions2.shape[1]
                        
                        opt_result = []
                        for i in range(opt):
                            
                            opt_result.append(find_nearest(predictions2.iloc[:,i],Target[i]))
      
        
                        st.write(opt_result)
                        st.write('')
                        st.write('')
                        st.write('')
                        st.markdown("<h6 style='text-align: left; color: #0e4194;'> * Optimizing Condition Results </h6>", unsafe_allow_html=True)
                    
                    
                        #df_max = para4[para4['predicted results']==para4['predicted results'].max()]
                        #df_min = para4[para4['predicted results']==para4['predicted results'].min()]
                        df_opt = pd.DataFrame()
                        
                        for i in range(opt):   
                            for column in df[Selected_y].columns:
                                df_opt = pd.concat([df_opt, para4[para4[column]==opt_result[i]]],axis=0)
                        """        
                        
                        st.write('**▶ 최적 공정조건(CTP) :**')
                        opt = para4.drop(['sum'],axis=1)
                        st.write(opt.head(5))
                        

                        st.write('')

                        st.write('**▶ 전체 샘플 예측 결과 :**')
                        st.write(opt)
                        
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        
                        Target5 = []
                        for i in range(Iter):
                            Target5.append(Target.values)
                        Target5 = np.reshape(Target5, (Iter, Selected_y.shape[0]))
                        Target5 = pd.DataFrame(Target5)
                        
                        plt.figure(figsize=(10,6))
                        fig, axs = plt.subplots(ncols=df[Selected_y].shape[1])
                        fig.subplots_adjust(hspace=1)
                    
                        opt_best5 = opt.head(5)
                        opt_ctq_best5 = opt_best5[Selected_y]
                        
                        for i in range(1,df[Selected_y].shape[1]+1):
                            
                            plt.subplot(1,df[Selected_y].shape[1],i)
                            
                            sns.lineplot(x=para4.index, y=Target5[i-1], color='red')
                            
                            plt.scatter(para4.index, predictions2.iloc[:,i-1], color='#0e4194', label = 'Prediction')
                            plt.scatter(opt_ctq_best5.index, opt_ctq_best5.iloc[:,i-1], color='#e30613', label = 'Prediction')
                            
                            plt.title(df[Selected_y].columns[i-1],fontsize=10)
                            plt.xticks(fontsize=8)
                            plt.yticks(fontsize=8)
                        #ax.set_xlabel('Time', fontsize=16)
                        #plt.ylabel(Y.columns[i-1], fontsize=10)
                        
                        st.pyplot()
                        

                        count +=1

                        st.write('')
                        st.write('')
                        st.write('')
                        st.markdown('**▶ 예측 결과 다운로드**')
            
                        st_pandas_to_csv_download_link(para4, file_name = "Predicted_CTPs_Results.csv")
                        st.caption("**_※ 저장 폴더 지정 방법 : 마우스 우클릭 → [다른 이름으로 링크 저장]_**")
                        
                        st.write('')
                        st.write('')
               
            
            
                    st.sidebar.write('')
                    st.sidebar.write('')    
                    st.sidebar.write('')
                    st.sidebar.write('**3.3 품질인자(CTQ) 예측 (Mulit-case)**')
                    uploaded_file3 = st.sidebar.file_uploader("Mulit-case 데이터(.csv) 업로드", type=["csv"])
                    if uploaded_file3 is not None:
                        def load_csv():
                            csv = pd.read_csv(uploaded_file3)
                            return csv
                    
        
        
    #    uploaded_file = st.file_uploader("Choose a file")
    #if uploaded_file is not None:
    #    uploaded_file.seek(0)
    #    data = pd.read_csv(uploaded_file, low_memory=False)
    #    st.write(data.shape)
            
                    count +=1
                    st.write('')
                    st.write('')        
                    st.write('')        
                    st.write('**3.3 품질인자(CTQ) 예측 (Multi-case)**')
                    st.markdown("<h6 style='text-align: left; color: #0e4194;'> * 사이드바_3.3 에서 업로드한 다양한 공정조건 케이스(CTP)에 대한 품질 결과(CTQ)를 예측합니다. </h6>", unsafe_allow_html=True)
                    
                    if st.sidebar.button('CTQ 예측하기', key = count):
                        df3 = load_csv()            
                        datafile = df3
        
                        
                        predictions2 = model.predict(datafile)
        
                        predictions2 = pd.DataFrame(predictions2,columns=df[Selected_y].columns)
        
                        #df3['predicted results'] = predictions2
                        
                        predictions3 = pd.DataFrame()
                        predictions3 = pd.concat([df3, predictions2],axis=1)
                        
                        st.write("")
                        st.write('**▶ 전체 샘플 예측 결과 :**')
                        st.write(predictions3)
        
        
                        
                    
                        plt.figure(figsize=(10,6))
                        fig, axs = plt.subplots(ncols=df[Selected_y].shape[1])
                        fig.subplots_adjust(hspace=1)
                    
                        for i in range(1,df[Selected_y].shape[1]+1):
                    

                        
                            plt.subplot(1,df[Selected_y].shape[1],i)
                            
                            #sns.lineplot(x=df3.index,y=Target[i-1], color='red')
                            
                            #plt.plot(df[Selected_y].iloc[:,i-1], df[Selected_y].iloc[:,i-1], color='blue', label = 'Actual data')
                            plt.scatter(df3.index, predictions2.iloc[:,i-1], color='#e30613', label = 'Prediction')
                            plt.title(df[Selected_y].columns[i-1],fontsize=10)
                            plt.xticks(fontsize=8)
                            plt.yticks(fontsize=8)
                        #ax.set_xlabel('Time', fontsize=16)
                        #plt.ylabel(Y.columns[i-1], fontsize=10)
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot()
                        
                    
                        #st.write('')
                        #st.write('**Total results:**')
                        #st.set_option('deprecation.showPyplotGlobalUse', False)
                        #st.write(df3)
                        #sns.scatterplot(x=df3.index,y=df3.iloc[:,-1],s=30,color='red')
                        #st.pyplot()
                        count +=1
            
        

                        st.write('')
                        st.write('')
                        st.write('')
                        st.markdown('**▶ 예측 결과 다운로드 (Multi-case)**')
                
                        st_pandas_to_csv_download_link(df3, file_name = "Predicted_CTQs_Results.csv")
                        st.caption("**_※ 저장 폴더 지정 방법 : 마우스 우클릭 → [다른 이름으로 링크 저장]_**")
                            
                        st.write('')
                        st.write('')
                        st.write('')
                        st.write('')
                        st.write('')


                        
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")
                    st.sidebar.write("")

                #st.write(mi)
            #parameter_n_neighbors = st.sidebar.slider('Number of neighbers', 2, 10, (1,6), 2)  
      

