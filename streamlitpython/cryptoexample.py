import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# Title and description
st.title("90-Day Cryptocurrency Dashboard with Trend Analysis")
st.write("Track Bitcoin price trends and patterns over the last 90 days.")

# API endpoint for cryptocurrency data
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {
    'vs_currency': 'usd',
    'days': '360'  # Get data for the last 360 days
}
headers = {
    "User-Agent": "Mozilla/5.0"
}

# Fetch data from the API
response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    data = response.json()
    if 'prices' in data:
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['Timestamp', 'Price'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
        df.to_csv("crypto_data.csv", index=False)

        st.subheader("Price over the last 360 days")
        fig = px.line(df, x='Timestamp', y='Price', title='Bitcoin Price (USD)')
        st.plotly_chart(fig)

        if st.checkbox("Show raw data"):
            st.write(df)
    else:
        st.error("Data format from API is not as expected.")
else:
    st.error(f"Failed to retrieve data: {response.status_code} - {response.reason}")
# Load the data from CSV
try:
    df = pd.read_csv("crypto_data.csv")
    # Modified datetime parsing to be more flexible
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
    
    # Calculate Weekly and Monthly Trends
    df.set_index('Timestamp', inplace=True)
    df = df.resample('D').mean()  # Resample to daily mean prices
    df['Weekly Change (%)'] = df['Price'].pct_change(periods=7) * 100  # Weekly percentage change
    df['Monthly Change (%)'] = df['Price'].pct_change(periods=30) * 100  # Monthly percentage change

    # Define Patterns
    def define_pattern(change):
        if change > 5:
            return 'Significant Up'
        elif change < -5:
            return 'Significant Down'
        else:
            return 'Stable'

    df['Weekly Pattern'] = df['Weekly Change (%)'].apply(define_pattern)
    df['Monthly Pattern'] = df['Monthly Change (%)'].apply(define_pattern)

    # Visualization of Weekly and Monthly Trends
    st.subheader("Weekly and Monthly Price Patterns")
    
    # Weekly Patterns
    fig_weekly = px.line(df, x=df.index, y='Price', title='Bitcoin Weekly Patterns')
    for pattern, color in zip(['Significant Up', 'Significant Down', 'Stable'], ['green', 'red', 'gray']):
        pattern_data = df[df['Weekly Pattern'] == pattern]
        fig_weekly.add_scatter(x=pattern_data.index, y=pattern_data['Price'], mode='markers', marker=dict(color=color), name=f"Weekly {pattern}")

    st.plotly_chart(fig_weekly)

    # Monthly Patterns
    fig_monthly = px.line(df, x=df.index, y='Price', title='Bitcoin Monthly Patterns')
    for pattern, color in zip(['Significant Up', 'Significant Down', 'Stable'], ['blue', 'orange', 'gray']):
        pattern_data = df[df['Monthly Pattern'] == pattern]
        fig_monthly.add_scatter(x=pattern_data.index, y=pattern_data['Price'], mode='markers', marker=dict(color=color), name=f"Monthly {pattern}")

    st.plotly_chart(fig_monthly)

    # Display Summary
    st.write("Weekly and Monthly Summary")
    st.write(df[['Price', 'Weekly Change (%)', 'Weekly Pattern', 'Monthly Change (%)', 'Monthly Pattern']].tail(30))

    def calculate_trading_signals(df):
        # Calculate additional technical indicators
        df['SMA_20'] = df['Price'].rolling(window=20).mean()
        df['SMA_50'] = df['Price'].rolling(window=50).mean()
        
        # Calculate price momentum
        df['Price_Change'] = df['Price'].pct_change()
        df['Momentum'] = df['Price_Change'].rolling(window=14).mean()
        
        # Calculate volatility
        df['Volatility'] = df['Price_Change'].rolling(window=20).std()
        
        # Generate trading signals
        df['Signal'] = 'Hold'
        
        # Buy signals: Price above SMA20, positive momentum, relatively low volatility
        buy_condition = (
            (df['Price'] > df['SMA_20']) & 
            (df['Momentum'] > 0) & 
            (df['Volatility'] < df['Volatility'].rolling(window=50).mean())
        )
        
        # Sell signals: Price below SMA50, negative momentum, high volatility
        sell_condition = (
            (df['Price'] < df['SMA_50']) & 
            (df['Momentum'] < 0) & 
            (df['Volatility'] > df['Volatility'].rolling(window=50).mean())
        )
        
        # Short signals: Price below both SMAs, strong negative momentum
        short_condition = (
            (df['Price'] < df['SMA_20']) & 
            (df['Price'] < df['SMA_50']) & 
            (df['Momentum'] < -0.02)
        )
        
        df.loc[buy_condition, 'Signal'] = 'Buy'
        df.loc[sell_condition, 'Signal'] = 'Sell'
        df.loc[short_condition, 'Signal'] = 'Short'
        
        return df

    # Add this after loading and processing your dataframe
    df = calculate_trading_signals(df)

    # Visualize trading signals
    st.subheader("Trading Signals Analysis")
    fig_signals = px.line(df, x=df.index, y='Price', title='Bitcoin Price with Trading Signals')

    # Add signals as scatter points
    signal_colors = {'Buy': 'green', 'Sell': 'red', 'Short': 'orange'}
    for signal, color in signal_colors.items():
        mask = df['Signal'] == signal
        fig_signals.add_scatter(
            x=df[mask].index,
            y=df[mask]['Price'],
            mode='markers',
            marker=dict(size=10, color=color),
            name=signal
        )

    st.plotly_chart(fig_signals)

    # Display current recommendation
    if len(df) > 0:
        latest_signal = df['Signal'].iloc[-1]
        latest_price = df['Price'].iloc[-1]
        st.subheader("Current Trading Recommendation")
        st.write(f"Latest Signal: {latest_signal}")
        st.write(f"Current Price: ${latest_price:,.2f}")
        
        # Calculate performance metrics
        signal_changes = df.groupby('Signal')['Price_Change'].mean()
        st.subheader("Historical Signal Performance")
        st.write("Average price change by signal type:")
        for signal, change in signal_changes.items():
            st.write(f"{signal}: {change*100:.2f}%")

except FileNotFoundError:
    st.write("No data found. Run the fetch script first.")
