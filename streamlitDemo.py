import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import requests
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set page title
st.set_page_config(page_title="BTC Correlation Analysis", layout="wide")
st.title("Bitcoin Price Correlation Analysis (Last Year)")

# Function: Get Bitcoin price data
# @st.cache_data(ttl=3600)
def get_btc_data():
    btc = yf.Ticker("BTC-USD")
    btc_data = btc.history(period="1y")
    st.write("BTC data fetched:", btc_data.tail())  # Log the last few rows of BTC data

    # Ensure the index is a DatetimeIndex
    if not isinstance(btc_data.index, pd.DatetimeIndex):
        btc_data.index = pd.to_datetime(btc_data.index)

    if btc_data.index.tz is not None:
        btc_data.index = btc_data.index.tz_localize(None)

    btc_data['MA30'] = btc_data['Close'].rolling(window=30).mean()
    btc_data['MA90'] = btc_data['Close'].rolling(window=90).mean()
    btc_data['RSI'] = calculate_rsi(btc_data['Close'])
    st.write("BTC data after processing:", btc_data.tail())  # Log the last few rows after processing
    return btc_data[['Close', 'MA30', 'MA90', 'RSI']]

# Function: Calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function: Get USDC market cap data
# @st.cache_data(ttl=3600)
def get_usdc_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    url = f"https://api.coingecko.com/api/v3/coins/usd-coin/market_chart/range?vs_currency=usd&from={int(start_date.timestamp())}&to={int(end_date.timestamp())}"
    st.write("USDC API URL:", url)  # Log the API URL
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'market_caps' not in data:
            st.error(f"No 'market_caps' key in API response. Response content: {data}")
            return pd.Series(dtype=float)
        df = pd.DataFrame(data['market_caps'], columns=['Date', 'Market Cap'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace=True)
        st.write("USDC data fetched:", df.tail())  # Log the last few rows of USDC data
        return df['Market Cap']
    except requests.RequestException as e:
        st.error(f"Error occurred while fetching USDC data: {e}")
        return pd.Series(dtype=float)

# Function: Get Fear & Greed Index data
# @st.cache_data(ttl=3600)
def get_fear_greed_index():
    url = "https://api.alternative.me/fng/?limit=365&format=json"
    response = requests.get(url)
    data = response.json()['data']
    df = pd.DataFrame(data, columns=['value', 'value_classification', 'timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    df['value'] = df['value'].astype(float)
    st.write("Fear & Greed Index data fetched:", df.head())  # Log the first few rows of Fear & Greed Index data
    return df['value']

# Function: Get NASDAQ-100 Index data
# @st.cache_data(ttl=3600)
def get_nasdaq_data():
    nasdaq = yf.Ticker("^NDX")
    nasdaq_data = nasdaq.history(period="1y")
    nasdaq_data.index = nasdaq_data.index.tz_localize(None)
    st.write("NASDAQ data fetched:", nasdaq_data.tail())  # Log the last few rows of NASDAQ data
    return nasdaq_data['Close']

# Function: Get Gold price data
# @st.cache_data(ttl=3600)
def get_gold_data():
    gold = yf.Ticker("GC=F")
    gold_data = gold.history(period="1y")
    gold_data.index = gold_data.index.tz_localize(None)
    st.write("Gold data fetched:", gold_data.tail())  # Log the last few rows of Gold data
    return gold_data['Close']

# Function: Load LSTM model
@st.cache_resource
def load_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    st.write(f"Model input shape: {input_shape}")
    return model

# Function: Prepare data for LSTM
# @st.cache_data
def prepare_data_for_lstm(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    st.write(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y, scaler

# Function: Predict future prices using LSTM
# @st.cache_data
def predict_future_prices(model, last_sequence, _scaler, days=7):
    st.write(f"Last sequence shape: {last_sequence.shape}")
    predicted_prices = []
    current_sequence = last_sequence.copy()

    for _ in range(days):
        predicted_price = model.predict(current_sequence)
        predicted_prices.append(predicted_price[0, 0])
        current_sequence = np.append(current_sequence[:, 1:, :], predicted_price.reshape((1, 1, 1)), axis=1)

    predicted_prices = _scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
    return predicted_prices

# Function to fill missing dates for any data source
def fill_missing_dates(df):
    all_dates = pd.date_range(start=df.index.min(), end=datetime.now(), freq='D')
    df = df.reindex(all_dates, method='ffill')
    return df

# Get data
btc_data = get_btc_data()
usdc_mcap = get_usdc_data()
fear_greed = get_fear_greed_index()
nasdaq_data = get_nasdaq_data()
gold_data = get_gold_data()

# Fill missing dates for all data sources
btc_data = fill_missing_dates(btc_data)
usdc_mcap = fill_missing_dates(usdc_mcap)
fear_greed = fill_missing_dates(fear_greed)
nasdaq_data = fill_missing_dates(nasdaq_data)
gold_data = fill_missing_dates(gold_data)

# Merge data
df = pd.concat([btc_data, usdc_mcap, fear_greed, nasdaq_data, gold_data], axis=1).dropna()
df.columns = ['BTC Price', 'MA30', 'MA90', 'RSI', 'USDC Market Cap', 'Fear & Greed Index', 'NASDAQ-100', 'Gold Price']

# Log the merged data
st.write("Merged data (last few rows):", df.tail())  # Log the last few rows of merged data
st.write("Merged data (first few rows):", df.head())  # Log the first few rows of merged data

# Check if data was successfully retrieved
if df.empty:
    st.error("Unable to retrieve data. Please check your network connection or try again later.")
    st.stop()

# Ensure all index are tz-naive
df.index = df.index.tz_localize(None)

# Check for missing values in the merged data
if df.isnull().any().any():
    st.warning("The merged data contains missing values. Please check the data sources.")
    st.stop()

# Calculate new correlation
correlation_matrix = df.corr()

# Display correlation matrix
st.write("Correlation Matrix:")
st.dataframe(correlation_matrix)

# Add indicator selector
st.sidebar.header("Select Indicators to Display")
show_ma30 = st.sidebar.checkbox("Show MA30", value=True)
show_ma90 = st.sidebar.checkbox("Show MA90", value=True)
show_usdc = st.sidebar.checkbox("Show USDC Market Cap", value=True)
show_nasdaq = st.sidebar.checkbox("Show NASDAQ-100", value=True)
show_gold = st.sidebar.checkbox("Show Gold Price", value=True)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_fear_greed = st.sidebar.checkbox("Show Fear & Greed Index", value=True)

# Create 5 independent charts
def create_figure(title):
    return go.Figure(layout=go.Layout(title=title, xaxis_title="Date", yaxis_title="Value"))

# Helper function to create hover template
def create_hover_template(df, columns):
    hover_template = "<br>".join([
        "Date: %{x|%Y-%m-%d}",
        *[f"{col}: %{{customdata[{i}]:,.2f}}" for i, col in enumerate(columns)]
    ])
    return hover_template, df[columns].values

# 1. BTC Price and Moving Averages
fig1 = create_figure("BTC Price and Moving Averages")
hover_cols = ['BTC Price']
if show_ma30:
    hover_cols.append('MA30')
if show_ma90:
    hover_cols.append('MA90')
hover_template, customdata = create_hover_template(df, hover_cols)

fig1.add_trace(go.Scatter(x=df.index, y=df['BTC Price'], mode='lines', name='BTC Price',
                          hovertemplate=hover_template, customdata=customdata))
if show_ma30:
    fig1.add_trace(go.Scatter(x=df.index, y=df['MA30'], mode='lines', name='MA30', line=dict(dash='dash'),
                              hovertemplate=hover_template, customdata=customdata))
if show_ma90:
    fig1.add_trace(go.Scatter(x=df.index, y=df['MA90'], mode='lines', name='MA90', line=dict(dash='dot'),
                              hovertemplate=hover_template, customdata=customdata))

# 2. BTC Price and USDC Market Cap
fig2 = create_figure("BTC Price and USDC Market Cap")
hover_cols = ['BTC Price', 'USDC Market Cap'] if show_usdc else ['BTC Price']
hover_template, customdata = create_hover_template(df, hover_cols)

fig2.add_trace(go.Scatter(x=df.index, y=df['BTC Price'], mode='lines', name='BTC Price',
                          hovertemplate=hover_template, customdata=customdata))
if show_usdc:
    fig2.add_trace(go.Scatter(x=df.index, y=df['USDC Market Cap'], mode='lines', name='USDC Market Cap', yaxis='y2',
                              hovertemplate=hover_template, customdata=customdata))
    fig2.update_layout(yaxis2=dict(title="USDC Market Cap", overlaying="y", side="right"))

# 3. BTC Price, RSI and Fear & Greed Index
fig3 = create_figure("BTC Price, RSI and Fear & Greed Index")
fig3.update_layout(height=600)

hover_cols = ['BTC Price']
if show_rsi:
    hover_cols.append('RSI')
if show_fear_greed:
    hover_cols.append('Fear & Greed Index')
hover_template, customdata = create_hover_template(df, hover_cols)

fig3.add_trace(go.Scatter(x=df.index, y=df['BTC Price'], mode='lines', name='BTC Price',
                          hovertemplate=hover_template, customdata=customdata))

if show_rsi:
    fig3.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI', yaxis='y2', line=dict(color='blue'),
                              hovertemplate=hover_template, customdata=customdata))
    fig3.add_hrect(y0=70, y1=100, line_width=0, fillcolor="red", opacity=0.3, yref='y2')
    fig3.add_hrect(y0=0, y1=30, line_width=0, fillcolor="green", opacity=0.3, yref='y2')
    fig3.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, yref='y2')
    fig3.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, yref='y2')
    fig3.add_annotation(x=df.index[-1], y=70, text="Overbought (RSI > 70)", showarrow=False, yref='y2', xref='x',
                        xanchor='right', yanchor='bottom', font=dict(color="white", size=10),
                        bgcolor="rgba(255,0,0,0.7)", bordercolor="rgba(255,0,0,0.7)", borderwidth=1, borderpad=4)
    fig3.add_annotation(x=df.index[-1], y=30, text="Oversold (RSI < 30)", showarrow=False, yref='y2', xref='x',
                        xanchor='right', yanchor='top', font=dict(color="white", size=10),
                        bgcolor="rgba(0,128,0,0.7)", bordercolor="rgba(0,128,0,0.7)", borderwidth=1, borderpad=4)

if show_fear_greed:
    fig3.add_trace(go.Scatter(x=df.index, y=df['Fear & Greed Index'], mode='lines', name='Fear & Greed Index', yaxis='y2', line=dict(color='red'),
                              hovertemplate=hover_template, customdata=customdata))
    fig3.add_hrect(y0=0, y1=25, line_width=0, fillcolor="red", opacity=0.1, yref='y2')
    fig3.add_hrect(y0=25, y1=45, line_width=0, fillcolor="orange", opacity=0.1, yref='y2')
    fig3.add_hrect(y0=45, y1=55, line_width=0, fillcolor="yellow", opacity=0.1, yref='y2')
    fig3.add_hrect(y0=55, y1=75, line_width=0, fillcolor="lime", opacity=0.1, yref='y2')
    fig3.add_hrect(y0=75, y1=100, line_width=0, fillcolor="green", opacity=0.1, yref='y2')
    fig3.add_annotation(x=df.index[-1], y=12.5, text="Extreme Fear (0-24)", showarrow=False, yref='y2', xref='x', xanchor='right', yanchor='middle', font=dict(color="red", size=10))
    fig3.add_annotation(x=df.index[-1], y=35, text="Fear (25-44)", showarrow=False, yref='y2', xref='x', xanchor='right', yanchor='middle', font=dict(color="red", size=10))
    fig3.add_annotation(x=df.index[-1], y=50, text="Neutral (45-55)", showarrow=False, yref='y2', xref='x', xanchor='right', yanchor='middle', font=dict(color="red", size=10))
    fig3.add_annotation(x=df.index[-1], y=65, text="Greed (56-75)", showarrow=False, yref='y2', xref='x', xanchor='right', yanchor='middle', font=dict(color="red", size=10))
    fig3.add_annotation(x=df.index[-1], y=87.5, text="Extreme Greed (76-100)", showarrow=False, yref='y2', xref='x', xanchor='right', yanchor='middle', font=dict(color="red", size=10))

fig3.update_layout(
    yaxis=dict(title="BTC Price"),
    yaxis2=dict(title="RSI / FGI", overlaying="y", side="right", range=[0, 100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=60, b=50, l=50, r=50)
)

# 4. NASDAQ-100
fig4 = create_figure("NASDAQ-100")
hover_cols = ['BTC Price', 'NASDAQ-100'] if show_nasdaq else ['BTC Price']
hover_template, customdata = create_hover_template(df, hover_cols)

fig4.add_trace(go.Scatter(x=df.index, y=df['BTC Price'], mode='lines', name='BTC Price',
                          hovertemplate=hover_template, customdata=customdata))
if show_nasdaq:
    fig4.add_trace(go.Scatter(x=df.index, y=df['NASDAQ-100'], mode='lines', name='NASDAQ-100', yaxis='y2',
                              hovertemplate=hover_template, customdata=customdata))
    fig4.update_layout(yaxis2=dict(title="NASDAQ-100", overlaying="y", side="right"))

# 5. Gold Price
fig5 = create_figure("Gold Price")
hover_cols = ['BTC Price', 'Gold Price'] if show_gold else ['BTC Price']
hover_template, customdata = create_hover_template(df, hover_cols)

fig5.add_trace(go.Scatter(x=df.index, y=df['BTC Price'], mode='lines', name='BTC Price',
                          hovertemplate=hover_template, customdata=customdata))
if show_gold:
    fig5.add_trace(go.Scatter(x=df.index, y=df['Gold Price'], mode='lines', name='Gold Price', yaxis='y2',
                              hovertemplate=hover_template, customdata=customdata))
    fig5.update_layout(yaxis2=dict(title="Gold Price", overlaying="y", side="right"))

# Add range selector for each chart
def add_range_selector(fig):
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

for fig in [fig1, fig2, fig3, fig4, fig5]:
    add_range_selector(fig)

# Display charts
st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)
st.plotly_chart(fig3, use_container_width=True)
st.plotly_chart(fig4, use_container_width=True)
st.plotly_chart(fig5, use_container_width=True)

# Display raw data
st.write("Raw Data:")
st.dataframe(df)

# Display data information
st.write("BTC Price (last few days):")
st.write(df['BTC Price'].tail())
st.write("USDC Market Cap (last few days):")
st.write(df['USDC Market Cap'].tail())

# Add download button
# @st.cache_data
def convert_df(df):
    return df.to_csv(index=True).encode('utf-8')

csv = convert_df(df)
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='btc_usdc_data.csv',
    mime='text/csv',
)

# LSTM Price Prediction
st.header("LSTM Price Prediction")

# Prepare data for LSTM
sequence_length = 60
X, y, scaler = prepare_data_for_lstm(df['BTC Price'].values, sequence_length)

# Load and train the LSTM model
lstm_model = load_lstm_model((sequence_length, 1))
lstm_model.fit(X, y, epochs=25, batch_size=32, verbose=0)

# Make predictions
future_days = 7
last_sequence = df['BTC Price'].values[-sequence_length:]
st.write(f"Original last sequence shape: {last_sequence.shape}")
last_sequence = scaler.transform(last_sequence.reshape(-1, 1)).reshape(1, -1, 1)
st.write(f"Transformed last sequence shape: {last_sequence.shape}")

predicted_prices = predict_future_prices(lstm_model, last_sequence, _scaler=scaler, days=future_days)

# Display predictions
st.write(f"Predicted BTC prices for the next {future_days} days:")
future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]
for date, price in zip(future_dates, predicted_prices):
    st.write(f"Date: {date.strftime('%Y-%m-%d')}, Predicted Price: ${price[0]:.2f}")

# Create a plot for the predictions
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=df.index, y=df['BTC Price'], mode='lines', name='Historical Price'))
fig_pred.add_trace(go.Scatter(x=future_dates, y=predicted_prices.flatten(), mode='lines+markers', name='Predicted Price'))
fig_pred.update_layout(title="BTC Price Prediction", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_pred, use_container_width=True)
