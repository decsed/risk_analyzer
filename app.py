import yfinance as yf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

@st.cache_data(show_spinner="Adatok letöltése...")
def get_financial_data(tickers_tuple, history):
    tickers_list = list(tickers_tuple)
    
    if tickers_list:
        df = yf.download(tickers_list, period=history)['Close']
        if isinstance(df, pd.Series):
            df = df.to_frame(tickers_list[0])
    else:
        df = pd.DataFrame()

    benchmark_data = yf.download("^GSPC", period=history)['Close']
    rf_data = yf.download("^TNX", period=history)["Close"]
    
    return df, benchmark_data, rf_data

left, center, right = st.columns([1,2,1])

with left:
    ticker_input = st.text_input("Add tickers (szóközzel elválasztva)", "AAPL MSFT NVDA TSLA")
    tickers = [t.upper() for t in ticker_input.split() if t]
    history = st.text_input("Define history (pl. 1y, 5y, ytd)", "1y")

df, benchmark_data, rf_data = get_financial_data(tuple(tickers), history)

shares = {}
with left:
    st.subheader("Részvények darabszáma")
    for ticker in tickers:
        shares[ticker] = st.number_input(
            f"{ticker} shares",
            min_value=0.0,
            value=10.0,
            step=1.0
        )

if not df.empty and sum(shares.values()) > 0:
    
    portfolio_value_df = pd.DataFrame(index=df.index)
    for ticker in tickers:
        if ticker in df.columns:
            portfolio_value_df[ticker] = df[ticker] * shares[ticker]
            
    total_portfolio_value = portfolio_value_df.sum(axis=1)
    
    portfolio_daily_returns = total_portfolio_value.pct_change().dropna()
    rel_daily_returns = df.pct_change().dropna()
    
    # JAVÍTOTT RÉSZ: numpy.ravel() használata a stabil számkinyeréshez
    rf_annual = float(np.ravel(rf_data.dropna())[-1]) / 100

    daily_volatility = portfolio_daily_returns.std()
    portfolio_volatility = daily_volatility * np.sqrt(252)
    annualized_portfolio_return = portfolio_daily_returns.mean() * 252
    sharpe = (annualized_portfolio_return - rf_annual) / portfolio_volatility
    
    cumulative_returns = (1 + portfolio_daily_returns).cumprod() - 1
    
    benchmark_daily_returns = benchmark_data.pct_change().dropna()
    benchmark_cumulative_returns = (1 + benchmark_daily_returns).cumprod() - 1

    portfolio_value_index = 1 + cumulative_returns
    running_max = portfolio_value_index.cummax()
    drawdown = (portfolio_value_index - running_max) / running_max
    max_drawdown = drawdown.min()

    with right:
        st.subheader("Portfólió Metrikák")
        st.write(f"**Sharpe-mutató:** {sharpe:.4f}")
        st.write(f"**Évesített Volatilitás:** {portfolio_volatility * 100:.2f}%")
        st.write(f"**Max Drawdown:** {max_drawdown * 100:.2f}%")
        st.write(f"**Évesített Hozam:** {annualized_portfolio_return * 100:.2f}%")

    with center:
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(cumulative_returns.index, cumulative_returns * 100, label="Saját Portfólió", color='blue', linewidth=2)
        ax1.plot(benchmark_cumulative_returns.index, benchmark_cumulative_returns * 100, label="S&P 500", color='red', linewidth=1.5, alpha=0.8)
        ax1.set_title("Saját Portfólió vs. S&P 500", fontsize=16)
        ax1.set_xlabel("Dátum", fontsize=12)
        ax1.set_ylabel("Kumulált hozam (%)", fontsize=12)
        ax1.axhline(0, color='black', linewidth=1, linestyle='--')
        ax1.grid(True, linestyle=':', alpha=0.7)
        ax1.legend(fontsize=12, loc="upper left")
        st.pyplot(fig1)

    with right:
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown * 100, color='darkred', linewidth=1)
        ax2.set_title("Drawdown", fontsize=16)
        ax2.set_xlabel("Dátum", fontsize=12)
        ax2.set_ylabel("Esés a csúcstól (%)", fontsize=12)
        ax2.set_ylim(drawdown.min() * 100 - 2, 0)
        ax2.axhline(0, color='black', linewidth=1)
        ax2.grid(True, linestyle=':', alpha=0.7)
        st.pyplot(fig2)

    with left:
        if len(tickers) > 1:
            st.subheader("Korrelációs Mátrix")
            correlation_matrix = rel_daily_returns.corr()
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax3)
            st.pyplot(fig3)

    with center:
        st.subheader("Kumulált Portfólió Hozamok Adatsor")
        st.dataframe((cumulative_returns * 100).round(2), use_container_width=True)

else:
    st.warning("Adj meg legalább egy tickert és állítsd be a darabszámokat!")