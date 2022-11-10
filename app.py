import streamlit as st
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Custom imports 
from multipage import MultiPage
#from pages import home, Prediction_app # import your pages here
#from pages.Build_model_app import app
from pages import home, Build_model_app, Prediction_app


#현재접속 시간 기록 → session_state 만들것임, if때문에 안바뀜
#time 함수쓰면, 버튼 누를때마다 시간변함
if 'key' not in st.session_state:
    nowtime1 = time.strftime("%Y%m%d%H%M%S")
    st.session_state['key'] = nowtime1

st.set_page_config(page_title = "KCC 머신러닝 프로그램")



# Create an instance of the app  
appl = MultiPage()
 
# Title of the main page
st.markdown("<h1 style='text-align: center; background-color:#0e4194; color: white;'>머신러닝 데이터분석</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; background-color:#0e4194; color: white;'>[Machine Learning Application]</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: right; color: black;'>KCC Ver. 2.1</h5>", unsafe_allow_html=True)



# Add all your applications (pages) here
appl.add_page("Home", home.app)
appl.add_page("Stage1. 머신러닝 모델 생성하기", Build_model_app.app)##
appl.add_page("Stage2. 최적 조건 예측하기", Prediction_app.app)

#if Build_model_app is 

# The main app
appl.run(st.session_state.key)
