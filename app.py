from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib  # To save/load model

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Load model if exists, otherwise train
MODEL_FILE = "stock_model.pkl"
SCALER_FILE = "scaler.pkl"

try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("✅ Model and scaler loaded successfully")
except:
    print("⚠️ No pre-trained model found. Training a new model...")

    # Fetch data for training (example using AAPL)
    df = yf.download("AAPL", period="1y")
    
    if not df.empty:
        features = ['Open', 'High', 'Low', 'Volume']
        df.dropna(inplace=True)
        X = df[features]
        y = df['Close']
        
        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Save model and scaler
        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        print("✅ Model trained and saved")

# Function to fetch stock data
def get_stock_data(ticker):
    stock = yf.download(ticker, period="1y")  # Get last 1 year of data
    if stock.empty:
        return None
    return stock

@app.route('/predict', methods=['POST'])
def predict_stock_trend():
    try:
        data = request.json
        ticker = data['ticker'].upper()

        # Fetch stock data
        df = get_stock_data(ticker)
        if df is None:
            return jsonify({'error': 'Invalid stock ticker'}), 400

        # Feature selection
        features = ['Open', 'High', 'Low', 'Volume']
        df.dropna(inplace=True)
        X = df[features]

        # Predict trend
        latest_data = X.iloc[-1].values.reshape(1, -1)
        latest_scaled = scaler.transform(latest_data)

        predicted_price = float(model.predict(latest_scaled).item())  # ✅ Fix 1
        last_close_price = df['Close'].iloc[-1].item()  # ✅ Fix 2
        trend = 'Upward' if predicted_price > last_close_price else 'Downward'

        # Generate trend graph
        img = io.BytesIO()
        plt.figure(figsize=(8, 4))
        plt.plot(df.index, df['Close'], label='Closing Price', color='blue')
        plt.title(f"Stock Price Trend: {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.savefig(img, format='png')  # ✅ Fix 3
        plt.close()
        img.seek(0)

        # Convert to base64
        graph_url = base64.b64encode(img.getvalue()).decode()

        return jsonify({
            'ticker': ticker,
            'predicted_trend': trend,
            'predicted_price': round(predicted_price, 2),
            'graph_url': f"data:image/png;base64,{graph_url}"  # ✅ Sending graph
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
