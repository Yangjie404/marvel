# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 19:34:07 2024

@author: Nancy
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #划分训练集测试集
import joblib
from sklearn.linear_model import LogisticRegression#逻辑回归模型
import streamlit as st


#导入数据
csv_data = pd.read_csv('https://raw.githubusercontent.com/Yangjie404/marvel/refs/heads/main/MARVEL.csv') # 读取data1.csv
all_data = csv_data[["sex", "age", "af","glucose","NIHSS","ASPECTS","ASITN","OCCT","TOAST"]]
all_data["sex"] = all_data["sex"].replace(["Male", "Female"], [1, 0])
all_data["af"] = all_data["af"].replace(["yes", "no"], [1, 0])
all_data["OCCT"] = all_data["OCCT"].replace(["ICA", "M1","M2","others"], [1,2,3,4])
all_data["TOAST"] = all_data["TOAST"].replace(["LAA", "CE","others"], [1,2,3])
labels = np.array(csv_data.iloc[:,0])

train_data, test_data, train_labels, test_labels = train_test_split(all_data, labels, test_size=0.2, random_state=2000) 
#数据标准化
# scaler = MinMaxScaler(feature_range=(-1,1))
# scaler.fit(train_data)
# train_data = scaler.transform(train_data)
# test_data = scaler.transform(test_data)
#开始训练模型（logistic回归）
reg_model = LogisticRegression(random_state=42) #声明一个线性模型
reg_model.fit(train_data,train_labels) #训练模型
p_target = reg_model.predict(test_data)     
p_target_proba = reg_model.predict_proba(test_data)
p_target_glm = np.squeeze(p_target)
p_target_proba_glm = np.squeeze(p_target_proba)



joblib.dump(reg_model, "reg_model.pkl")



# Title
st.set_page_config(page_title=('Prediction model for large ischemic core stroke'),page_icon=(":tiger:"),layout='wide')
st.header("Large ischemic core stroke prognostic model")

# Input bar 1
sex = st.selectbox("Select Sex", ("Female", "Male"))
age = st.number_input("Enter Age", step=1)
af = st.selectbox("Select Atrial fibrillation ", ("no", "yes"))
glucose = st.number_input("Enter Serum glucose")
NIHSS = st.number_input("Enter Baseline NIHSS score",min_value=0, max_value=42, step=1)
ASPECTS  = st.number_input("Enter Baseline ASPECTS score",min_value=0, max_value=5, step=1)
ASITN = st.number_input("Enter ASITN/SIR",min_value=0, max_value=4, step=1)
OCCT = st.selectbox("Select Occlusion site", ("ICA", "M1", "M2","others"))
TOAST = st.selectbox("Select Stroke etiology", ("LAA", "CE", "others"))
# Input bar 2


if st.button("Submit"):
    
    # Unpickle classifier
    clf = joblib.load("reg_model.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[sex, age, af,glucose,NIHSS,ASPECTS,ASITN,OCCT,TOAST]], 
                     columns = ["sex", "age", "af","glucose","NIHSS","ASPECTS","ASITN","OCCT","TOAST"])
    X["sex"] = X["sex"].replace(["Male", "Female"], [1, 0])
    X["af"] = X["af"].replace(["yes", "no"], [1, 0])
    X["OCCT"] = X["OCCT"].replace(["ICA", "M1","M2","others"], [1,2,3,4])
    X["TOAST"] = X["TOAST"].replace(["LAA", "CE","others"], [1,2,3])

    
    # Get prediction
    # prediction = reg_model.predict(X)
    prediction = reg_model.predict_proba(X)[:, 1]
    #prediction = prediction * 100
    # Output prediction
    # 设置文本字体大小和颜色
    # 设置文本字体大小
    # 设置文本字体大小
    st.write(
    f"<p style='font-size: 40px;'>Based on feature values,predicted probability of function independence is {prediction} !</p>",
    unsafe_allow_html=True,
    )





