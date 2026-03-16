import yfinance as yf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(layout="wide")

left, center, right = st.columns([1,2,1])
with left:
    tickers = [t.upper() for t in st.text_input("Add tickers","AAPL MSFT nvda TsLa").split()]

weights = {}
with left:
    for ticker in tickers:
        weights[ticker] = st.slider(
            f"{ticker} weight",
            0,100,25
        )

with left:
    history = st.text_input("define history","1y")

total_weight = sum(weights.values())

with left:
    if total_weight != 100:
        st.write(f"Sum of weights is {total_weight}. Normalizing to percentages...")
normalized_weights = [w / total_weight for w in weights.values()]

df = yf.download(tickers, period=history, auto_adjust=False)['Adj Close']

#df = df[tickers]

#st.write(df.head())

# relative daily returns (today - yesterday) / yesterday

rel_daily_returns  = (df - df.shift(1)) / df.shift(1)
portfolio_daily_returns = (rel_daily_returns *  normalized_weights).sum(axis=1)
#st.write("Daily Portfolio Returns")
#st.write(portfolio_daily_returns.dropna())


# annualized volatility per stock

daily_volatility = rel_daily_returns.std()
annualized_volatility = daily_volatility *  np.sqrt(252)
with right:
    st.write("Portfolio Annualized Volatitlity")
    st.write(annualized_volatility * 100)

# annualized volatility portfolio 

portfolio_volatility = portfolio_daily_returns.std() * np.sqrt(252)

# cumulative daily returns
cumulative_returns = (1 + portfolio_daily_returns).cumprod() - 1



# sharpe-ratio (portfolio profit - risk free profit) / portfolio volatility

rf_data = yf.download("^TNX", period=history)["Close"]

rf_annual = float(rf_data.iloc[-1].iloc[0])/100

annualized_portfolio_return = portfolio_daily_returns.mean() * 252

sharpe = (annualized_portfolio_return - rf_annual) / portfolio_volatility
with right:
    st.write("Metrics")
    st.write(f"Sharpe of Portfolio: {sharpe:.4f}")
    st.write(f"Volatility of Portfolio: {portfolio_volatility * 100:.4f}")


# correlation matrix

correlation_matrix = rel_daily_returns.corr()

plt.figure(figsize = (6,4))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Portfoli Correlation Matrix")
with left:
    st.pyplot(plt)

benchmark_data = yf.download("^GSPC", period=history, auto_adjust=False)['Adj Close']

benchmark_daily_returns = benchmark_data.pct_change()
benchmark_cumulative_returns = (1 + benchmark_daily_returns).cumprod() - 1

plt.figure(figsize=(12, 6))

plt.plot(cumulative_returns.index, cumulative_returns * 100, label="My Portfolio", color='blue', linewidth=2)
plt.plot(benchmark_cumulative_returns.index, benchmark_cumulative_returns * 100, label="S&P 500", color='red', linewidth=1.5, alpha=0.8)

plt.title("My Portfolio vs. S&P 500", fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Cumulative Returns (%)", fontsize=12)

plt.axhline(0, color='black', linewidth=1, linestyle='--')

plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=12, loc="upper left")

with center:
    st.pyplot(plt)

# maximum drawdown (underwater)

portfolio_value = 1 + cumulative_returns
running_max = portfolio_value.cummax()

drawdown = (portfolio_value - running_max) / running_max
max_drawdown = drawdown.min()
with right:
    st.write(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")

plt.figure(figsize=(12,4))
plt.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
plt.plot(drawdown.index, drawdown * 100, color='darkred', linewidth=1)
plt.title("Drawdown", fontsize=16)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Drawdown from peak (%)", fontsize=12)

plt.ylim(drawdown.min() * 100 - 2, 0)

plt.axhline(0, color='black', linewidth=1)
plt.grid(True, linestyle=':', alpha=0.7)

with right:
    st.pyplot(plt)
with center:
    st.write("Cumulative Portfolio Returns")
    st.write(cumulative_returns * 100)