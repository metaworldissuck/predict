import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import requests
from datetime import datetime, timedelta

# 设置页面标题
st.set_page_config(page_title="BTC-USDC 相关性分析", layout="wide")
st.title("比特币价格与USDC市值相关性分析（最近一年）")

# 函数：获取比特币价格数据
@st.cache_data(ttl=3600)  # 设置缓存过期时间为1小时
def get_btc_data():
    btc = yf.Ticker("BTC-USD")
    btc_data = btc.history(period="1y")
    btc_data.index = btc_data.index.tz_localize(None)  # 移除时区信息
    return btc_data['Close']

# 函数：获取USDC市值数据
@st.cache_data(ttl=3600)  # 设置缓存过期时间为1小时
def get_usdc_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1年的数据
    url = f"https://api.coingecko.com/api/v3/coins/usd-coin/market_chart/range?vs_currency=usd&from={int(start_date.timestamp())}&to={int(end_date.timestamp())}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # 如果请求不成功，这将引发一个异常
        data = response.json()

        if 'market_caps' not in data:
            st.error(f"API 响应中没有 'market_caps' 键。响应内容: {data}")
            return pd.Series(dtype=float)

        df = pd.DataFrame(data['market_caps'], columns=['Date', 'Market Cap'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace=True)
        return df['Market Cap']

    except requests.RequestException as e:
        st.error(f"获取 USDC 数据时发生错误: {e}")
        return pd.Series(dtype=float)

# 获取数据
btc_price = get_btc_data()
usdc_mcap = get_usdc_data()

# 检查是否成功获取了数据
if btc_price.empty or usdc_mcap.empty:
    st.error("无法获取数据。请检查您的网络连接或稍后再试。")
    st.stop()

# 确保两个序列的索引都是 tz-naive 的
btc_price.index = btc_price.index.tz_localize(None)
usdc_mcap.index = usdc_mcap.index.tz_localize(None)

# 合并数据
df = pd.concat([btc_price, usdc_mcap], axis=1).dropna()
df.columns = ['BTC Price', 'USDC Market Cap']

# 检查合并后的数据是否有缺失值
if df.isnull().any().any():
    st.warning("合并后的数据包含缺失值，请检查数据源。")
    st.stop()

# 计算相关性
correlation = df['BTC Price'].corr(df['USDC Market Cap'])

# 显示相关性
st.write(f"BTC价格和USDC市值的皮尔逊相关系数: {correlation:.2f}")

# 创建动态图表
fig = go.Figure()

# 添加 BTC 价格线图
fig.add_trace(go.Scatter(x=df.index, y=df['BTC Price'], mode='lines', name='BTC Price', yaxis='y1'))

# 添加 USDC 市值线图
fig.add_trace(go.Scatter(x=df.index, y=df['USDC Market Cap'], mode='lines', name='USDC Market Cap', yaxis='y2'))

# 设置图表布局
fig.update_layout(
    title='BTC Price and USDC Market Cap Over Time (Last Year)',
    xaxis=dict(title='Date'),
    yaxis=dict(title='BTC Price (USD)', side='left'),
    yaxis2=dict(title='USDC Market Cap (USD)', side='right', overlaying='y'),
    legend=dict(x=0, y=1),
    hovermode='x unified'
)

# 显示动态图表
st.plotly_chart(fig, use_container_width=True)

# 创建散点图
scatter_fig = go.Figure()

scatter_fig.add_trace(go.Scatter(x=df['BTC Price'], y=df['USDC Market Cap'], mode='markers', name='BTC Price vs USDC Market Cap'))

scatter_fig.update_layout(
    title='BTC Price vs USDC Market Cap (Last Year)',
    xaxis=dict(title='BTC Price (USD)'),
    yaxis=dict(title='USDC Market Cap (USD)'),
    hovermode='closest'
)

# 显示散点图
st.plotly_chart(scatter_fig, use_container_width=True)

# 显示原始数据
st.write("原始数据:")
st.dataframe(df)

# 显示数据信息
st.write("BTC Price 数据:")
st.write(btc_price.head())
st.write("USDC Market Cap 数据:")
st.write(usdc_mcap.head())

# 添加下载按钮
@st.cache_data
def convert_df(df):
    return df.to_csv(index=True).encode('utf-8')

csv = convert_df(df)
st.download_button(
    label="下载数据为 CSV",
    data=csv,
    file_name='btc_usdc_data.csv',
    mime='text/csv',
)