import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

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

    def create_features(df):
        # Create lagged features and technical indicators
        df['Lag1'] = df['Price'].shift(1)
        df['Lag7'] = df['Price'].shift(7)
        df['Returns'] = df['Price'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=30).std()
        df['RSI'] = calculate_rsi(df['Price'], periods=14)
        df['MA20'] = df['Price'].rolling(window=20).mean()
        df['MA50'] = df['Price'].rolling(window=50).mean()
        
        # Create target variable (next day's price)
        df['Target'] = df['Price'].shift(-1)
        
        return df.dropna()

    def calculate_rsi(prices, periods=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def market_efficiency_test(df):
        # Run Augmented Dickey-Fuller test
        adf_result = adfuller(df['Returns'].dropna())
        
        # Run Ljung-Box test for autocorrelation
        lb_result = acorr_ljungbox(df['Returns'].dropna(), lags=[10], return_df=True)
        
        return adf_result, lb_result

    def predict_prices(df):
        # Prepare features
        feature_cols = ['Lag1', 'Lag7', 'Volatility', 'RSI', 'MA20', 'MA50']
        X = df[feature_cols]
        y = df['Target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Initialize models
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=100)
        }
        
        predictions = {}
        metrics = {}
        
        # Train and evaluate models
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions[name] = pred
            metrics[name] = {
                'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
                'R2': r2_score(y_test, pred)
            }
        
        return predictions, metrics, X_test.index

    # Add this after your existing data processing
    df = create_features(df)

    # Run market efficiency tests
    adf_result, lb_result = market_efficiency_test(df)

    # Get predictions
    predictions, metrics, test_dates = predict_prices(df)

    # Visualize results
    st.subheader("Market Efficiency Analysis")
    st.write("1. Augmented Dickey-Fuller Test Results:")
    st.write(f"ADF Statistic: {adf_result[0]:.4f}")
    st.write(f"p-value: {adf_result[1]:.4f}")
    st.write("Critical values:")
    for key, value in adf_result[4].items():
        st.write(f"\t{key}: {value:.4f}")

    st.write("\n2. Ljung-Box Test Results (tests for random walk):")
    st.write(f"Test Statistic: {lb_result['lb_stat'].iloc[0]:.4f}")
    st.write(f"p-value: {lb_result['lb_pvalue'].iloc[0]:.4f}")

    # Interpret results
    st.write("\nMarket Efficiency Interpretation:")
    if adf_result[1] < 0.05:
        st.write("✓ Price series is stationary (potentially predictable)")
    else:
        st.write("✗ Price series follows random walk")

    if lb_result['lb_pvalue'].iloc[0] < 0.05:
        st.write("✓ Significant autocorrelation detected (potentially predictable)")
    else:
        st.write("✗ No significant autocorrelation (more random)")

    # Add confidence score based on test results
    confidence_score = (
        (1 if adf_result[1] < 0.05 else 0) + 
        (1 if lb_result['lb_pvalue'].iloc[0] < 0.05 else 0)
    ) / 2 * 100

    st.write(f"\nPredictability Confidence Score: {confidence_score}%")

    # Create prediction visualization
    fig_pred = px.line(df, x=df.index, y='Price', title='Bitcoin Price Predictions')

    # Add prediction lines for each model
    for model_name, preds in predictions.items():
        fig_pred.add_scatter(
            x=test_dates,
            y=preds,
            name=f"{model_name} Prediction",
            line=dict(dash='dash')
        )

    st.plotly_chart(fig_pred)

    # Display model metrics
    st.subheader("Model Performance Metrics")
    metrics_df = pd.DataFrame(metrics).T
    st.dataframe(metrics_df)

    # Calculate confidence intervals for latest prediction
    latest_predictions = {model: pred[-1] for model, pred in predictions.items()}
    mean_prediction = np.mean(list(latest_predictions.values()))
    std_prediction = np.std(list(latest_predictions.values()))
    confidence_interval = stats.norm.interval(0.95, mean_prediction, std_prediction)

    st.subheader("Price Prediction Analysis")
    st.write(f"Consensus Prediction: ${mean_prediction:,.2f}")
    st.write(f"95% Confidence Interval: ${confidence_interval[0]:,.2f} to ${confidence_interval[1]:,.2f}")

    # Risk assessment
    st.subheader("Risk Assessment")
    volatility = df['Volatility'].iloc[-1]
    risk_level = "High" if volatility > df['Volatility'].quantile(0.75) else "Medium" if volatility > df['Volatility'].quantile(0.25) else "Low"

    st.write(f"Current Volatility: {volatility:.4f}")
    st.write(f"Risk Level: {risk_level}")

    # Model disagreement as uncertainty indicator
    model_disagreement = std_prediction / mean_prediction
    st.write(f"Model Uncertainty: {model_disagreement:.2%}")

    def analyze_temporal_patterns(df):
        # Add temporal features
        df['Day_of_Month'] = df.index.day
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Day_of_Week'] = df.index.dayofweek  # 0 = Monday, 6 = Sunday
        
        # Calculate daily returns
        df['Daily_Return'] = df['Price'].pct_change()
        
        # Define time periods
        df['Period'] = 'Mid_Month'
        df.loc[df['Day_of_Month'] <= 5, 'Period'] = 'Month_Start'
        df.loc[df['Day_of_Month'] >= 25, 'Period'] = 'Month_End'
        
        # Calculate pattern metrics
        patterns = {
            'Monthly': {},
            'Quarter': {},
            'Pay_Periods': {},
            'Day_of_Week': {}
        }
        
        # Monthly patterns (Start, Mid, End)
        monthly_stats = df.groupby('Period')['Daily_Return'].agg(['mean', 'std', 'count'])
        patterns['Monthly'] = monthly_stats.to_dict()
        
        # Quarterly patterns
        quarterly_stats = df.groupby('Quarter')['Daily_Return'].agg(['mean', 'std', 'count'])
        patterns['Quarter'] = quarterly_stats.to_dict()
        
        # Common pay period patterns (1st and 15th of month +/- 2 days)
        pay_period_mask = (df['Day_of_Month'].isin(range(1, 4))) | (df['Day_of_Month'].isin(range(14, 17)))
        df['Is_Pay_Period'] = pay_period_mask
        pay_period_stats = df.groupby('Is_Pay_Period')['Daily_Return'].agg(['mean', 'std', 'count'])
        patterns['Pay_Periods'] = pay_period_stats.to_dict()
        
        return patterns, df

    # Analyze patterns
    patterns, df_with_patterns = analyze_temporal_patterns(df)

    # Visualize results
    st.subheader("Temporal Pattern Analysis (Last 360 Days)")

    # Monthly pattern visualization
    monthly_returns = df_with_patterns.groupby('Period')['Daily_Return'].mean() * 100
    fig_monthly = px.bar(
        x=monthly_returns.index,
        y=monthly_returns.values,
        title="Average Returns by Month Period",
        labels={'x': 'Period', 'y': 'Average Return (%)'}
    )
    st.plotly_chart(fig_monthly)

    # Quarterly pattern visualization
    quarterly_returns = df_with_patterns.groupby('Quarter')['Daily_Return'].mean() * 100
    fig_quarterly = px.bar(
        x=quarterly_returns.index,
        y=quarterly_returns.values,
        title="Average Returns by Quarter",
        labels={'x': 'Quarter', 'y': 'Average Return (%)'}
    )
    st.plotly_chart(fig_quarterly)

    # Pay period analysis
    st.subheader("Pay Period Impact Analysis")
    pay_period_returns = df_with_patterns.groupby('Is_Pay_Period')['Daily_Return'].mean() * 100
    st.write("Average Returns:")
    st.write(f"During Pay Periods (1st & 15th ± 2 days): {pay_period_returns[True]:.2f}%")
    st.write(f"Non-Pay Periods: {pay_period_returns[False]:.2f}%")

    # Day of week analysis
    dow_returns = df_with_patterns.groupby('Day_of_Week')['Daily_Return'].mean() * 100
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    fig_dow = px.bar(
        x=dow_names,
        y=dow_returns.values,
        title="Average Returns by Day of Week",
        labels={'x': 'Day', 'y': 'Average Return (%)'}
    )
    st.plotly_chart(fig_dow)

    # Statistical significance
    st.subheader("Pattern Significance")

    # Perform ANOVA test for monthly periods
    from scipy import stats
    monthly_groups = [group['Daily_Return'].values for name, group in df_with_patterns.groupby('Period')]
    f_stat, p_value = stats.f_oneway(*monthly_groups)

    st.write("Statistical Significance of Patterns:")
    st.write(f"Monthly Pattern p-value: {p_value:.4f}")
    significant = p_value < 0.05
    st.write(f"Monthly patterns are {'statistically significant' if significant else 'not statistically significant'}")

    # Trading recommendations based on patterns
    st.subheader("Trading Recommendations Based on Temporal Patterns")

    current_day = pd.Timestamp.now()
    current_period = 'Month_End' if current_day.day >= 25 else 'Month_Start' if current_day.day <= 5 else 'Mid_Month'
    current_quarter = current_day.quarter
    is_pay_period = current_day.day in range(1, 4) or current_day.day in range(14, 17)

    st.write(f"Current Period: {current_period}")
    st.write(f"Current Quarter: Q{current_quarter}")
    st.write(f"Pay Period: {'Yes' if is_pay_period else 'No'}")

    # Generate trading suggestion
    best_period = monthly_returns.idxmax()
    best_quarter = quarterly_returns.idxmax()
    st.write(f"\nHistorical Best Performance:")
    st.write(f"- Best Period: {best_period} ({monthly_returns[best_period]:.2f}% avg return)")
    st.write(f"- Best Quarter: Q{best_quarter} ({quarterly_returns[best_quarter]:.2f}% avg return)")

except FileNotFoundError:
    st.write("No data found. Run the fetch script first.")
