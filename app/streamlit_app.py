import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Path append logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import PhishingDetector

# Page configuration
st.set_page_config(page_title="Phishing Detection", layout="wide")
st.title("Network Phishing Detection")

# Sidebar block
with st.sidebar:
    st.header("About Model")
    st.info("Model: 1D CNN\nFeatures: 30\nAccuracy: ~97%")
    st.metric("Total Samples", "11,055")

# Tabs setup
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Performance", "Data Analysis"])

# List of all 30 features
feature_names = [
    'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol', 'double_slash_redirecting',
    'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 'Favicon',
    'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email',
    'Abnormal_URL', 'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
    'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page', 'Statistical_report'
]

# TAB 1: Prediction Interface
with tab1:
    cols = st.columns(5)
    features = []
    
    for i, name in enumerate(feature_names):
        with cols[i % 5]:
            val = st.selectbox(name, [-1, 0, 1], index=1, key=f"f_{i}")
            features.append(val)
            
    if st.button("Analyze URL", type="primary", use_container_width=True):
        res = PhishingDetector().predict_single(features)
        c1, c2, c3 = st.columns(3)
        
        if res['status'] == 'Phishing':
            c1.error(f"Status: {res['status']}")
        else:
            c1.success(f"Status: {res['status']}")
            
        c2.metric("Confidence", f"{res['confidence']*100:.1f}%")
        c3.metric("Probability", f"{res['probability']*100:.1f}%")
        
        fig, ax = plt.subplots(figsize=(8, 1))
        plot_color = 'red' if res['probability'] > 0.5 else 'green'
        ax.barh(['Phishing Probability'], [res['probability']], color=plot_color)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color='black', linestyle='--')
        st.pyplot(fig)

# TAB 2: Performance Evaluation Graphs
with tab2:
    col_left, col_right = st.columns(2)
    col_left.image('static/images/confusion_matrix.png', caption='Confusion Matrix')
    col_right.image('static/images/training_history.png', caption='Training History')
    
    metrics_dict = {'Accuracy': '97.2%', 'Precision': '96.8%', 'Recall': '97.5%', 'F1-Score': '97.1%'}
    metrics_cols = st.columns(4)
    for i, (m, v) in enumerate(metrics_dict.items()):
        metrics_cols[i].metric(m, v)

# TAB 3: Dataset Analytics
with tab3:
    df = pd.read_csv('data/phisingData.csv')
    st.dataframe(df.head(5))
    st.dataframe(df.describe())
    
    fig, ax = plt.subplots(figsize=(4, 3))
    class_counts = df['Result'].replace(-1, 0).value_counts().values
    ax.pie(class_counts, labels=['Legitimate', 'Phishing'], autopct='%1.1f%%', colors=['green', 'red'])
    
    analysis_cols = st.columns(2)
    analysis_cols[0].pyplot(fig)
