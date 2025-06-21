import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from datetime import datetime

# Load models
@st.cache_resource
def load_all_models():
    bert_model = BertForSequenceClassification.from_pretrained('model/bert_expense_model')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    lstm_model = load_model('model/lstm_model.h5')
    amount_scaler = pickle.load(open('model/amount_scaler.pkl', 'rb'))
    city_encoder = pickle.load(open('model/city_encoder.pkl', 'rb'))
    return bert_model, tokenizer, lstm_model, amount_scaler, city_encoder

bert_model, tokenizer, lstm_model, amount_scaler, city_encoder = load_all_models()

st.title("Smart Expense Predictor & Budget Planner")

uploaded_file = st.file_uploader("Upload your CSV transaction file", type="csv")

if uploaded_file is not None:
    cc = pd.read_csv(uploaded_file)

    # Preprocessing
    cc['description'] = cc['Expense_Type'] + " using " + cc['Card_Type'] + " in " + cc['City']
    # Encode city column
    cc['City'] = city_encoder.transform(cc['City'])
    cc['Date'] = pd.to_datetime(cc['Date'])
    cc = cc.sort_values(by='Date')
    cc['month'] = cc['Date'].dt.to_period('M').dt.to_timestamp()

    # BERT
    inputs = tokenizer(list(cc['description']), truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**inputs)
        preds = torch.argmax(outputs.logits, axis=1)
    cc['Predicted Category'] = preds.numpy()

    st.subheader("📊 Transaction Categories")
    st.dataframe(cc[['description', 'Predicted Category']])

    # LSTM Forecasting
    monthly = cc.groupby('month')['Amount'].sum().reset_index()
    monthly['amount_scaled'] = amount_scaler.transform(monthly[['amount']])
    sequence = monthly['Amount'].values[-3:].reshape(1, 3, 1)
    prediction = lstm_model.predict(sequence)
    forecast = amount_scaler.inverse_transform([[prediction[0][0]]])[0][0]

    st.subheader("Forecast")
    st.write(f"Predicted next month’s spending: ₹{forecast:.2f}")

    # Plot
    st.line_chart(monthly.set_index('month')['Amount'])

    
