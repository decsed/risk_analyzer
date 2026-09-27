import yfinance as yf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import datetime

st.set_page_config(layout="wide")

@st.cache_data(show_spinner="Adatok letöltése...")
def get_financial_data(tickers_tuple, start_date):
    tickers_list = list(tickers_tuple)
    
    if tickers_list:
        df = yf.download(tickers_list, start=start_date, auto_adjust=False)['Adj Close']
        if isinstance(df, pd.Series):
            df = df.to_frame(tickers_list[0])
    else:
        df = pd.DataFrame()

    benchmark_data = yf.download("^GSPC", start=start_date, auto_adjust=False)['Adj Close']
    rf_data = yf.download("^TNX", start=start_date)["Close"]
    
    return df, benchmark_data, rf_data

def parse_portfolio_txt(file_content):
    portfolio = {}
    errors = []

    for line_number, raw_line in enumerate(file_content.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue

        parts = [part.strip() for part in line.replace(';', ',').split(',')]
        if len(parts) == 1:
            parts = line.split()

        if len(parts) != 3:
            errors.append(f"{line_number}. sor: ticker, mennyiség és dátum szükséges")
            continue

        ticker, shares_text, buy_date_text = parts
        try:
            shares = float(shares_text.replace(',', '.'))
            buy_date = datetime.datetime.strptime(buy_date_text, '%Y%m%d').date()
            if not ticker or shares <= 0:
                raise ValueError
        except ValueError:
            errors.append(f"{line_number}. sor: hibás adat ({raw_line.strip()})")
            continue

        portfolio[ticker.upper()] = {
            'shares': shares,
            'buy_date': buy_date
        }

    return portfolio, errors

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}

left, center, right = st.columns([1,2,1])

with left:
    st.subheader("Részvény hozzáadása")
    with st.form("add_stock_form", clear_on_submit=True):
        new_ticker = st.text_input("Ticker (pl. AAPL)").upper()
        new_shares = st.number_input("Darabszám", min_value=0.0, step=1.0, value=10.0)
        new_buy_date = st.date_input("Vásárlás dátuma", value=datetime.date(2023, 1, 1))
        submit_button = st.form_submit_button("Hozzáadás / Módosítás")
        
        if submit_button and new_ticker:
            st.session_state.portfolio[new_ticker] = {
                'shares': new_shares,
                'buy_date': new_buy_date
            }
            st.rerun()

    st.subheader("Portfólió importálása TXT-ből")
    st.caption("Formátum soronként: TICKER, mennyiség, YYYYMMDD")
    uploaded_file = st.file_uploader("TXT-fájl kiválasztása", type=['txt'])
    import_button = st.button("Portfólió importálása", disabled=uploaded_file is None)

    if import_button and uploaded_file is not None:
        imported_portfolio, import_errors = parse_portfolio_txt(
            uploaded_file.getvalue().decode('utf-8-sig')
        )
        if import_errors:
            st.error("Az import sikertelen:")
            for error in import_errors:
                st.write(f"- {error}")
        elif not imported_portfolio:
            st.error("A TXT-fájl nem tartalmaz importálható adatot.")
        else:
            st.session_state.portfolio = imported_portfolio
            st.success(f"{len(imported_portfolio)} részvény sikeresen importálva.")
            st.rerun()

    st.subheader("Jelenlegi Portfólió")
    if not st.session_state.portfolio:
        st.info("Még nincs részvény a portfólióban.")
    else:
        for t, data in list(st.session_state.portfolio.items()):
            col1, col2 = st.columns([4, 1])
            col1.write(f"**{t}**: {data['shares']} db ({data['buy_date']})")
            if col2.button("🗑️", key=f"del_{t}"):
                del st.session_state.portfolio[t]
                st.rerun()

tickers = list(st.session_state.portfolio.keys())

if tickers:
    earliest_date = min([data['buy_date'] for data in st.session_state.portfolio.values()])
    df, benchmark_data, rf_data = get_financial_data(tuple(tickers), earliest_date)
else:
    st.warning("Adj hozzá legalább egy részvényt a bal oldali sávban!")
    
if tickers and not df.empty and sum(data['shares'] for data in st.session_state.portfolio.values()) > 0:
    missing_data_tickers = [ticker for ticker in df.columns if df[ticker].isnull().any()]
    if missing_data_tickers:
        st.warning(f"⚠️ **Figyelem:** Az alábbi részvényeknek nincs meg a teljes adatsora a választott időtávon (pl. frissebb IPO): **{', '.join(missing_data_tickers)}**. A pontos számítások érdekében a portfólió elemzése a legfiatalabb részvény indulásához lett igazítva!")
    df = df.dropna()
    
    portfolio_value_df = pd.DataFrame(index=df.index)
    for ticker in tickers:
        if ticker in df.columns:
            portfolio_value_df[ticker] = df[ticker] * st.session_state.portfolio[ticker]['shares']
            
    total_portfolio_value = portfolio_value_df.sum(axis=1)
    
    portfolio_daily_returns = total_portfolio_value.pct_change().dropna()
    rel_daily_returns = df.pct_change().dropna()
    
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
        st.subheader("Kumulált Portfólió Hozamok")
        st.line_chart((cumulative_returns * 100).round(2))

else:
    st.warning("Adj meg legalább egy tickert és állítsd be a darabszámokat!")