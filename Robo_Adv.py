import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go

# Liste d'actions euronext et global stable
euronext_stable = [
    'MC.PA', 'ASML.AS', 'TTE.PA', 'OR.PA', 'RMS.PA', 'AIR.PA', 'SU.PA', 'SAN.PA', 'BNP.PA', 'ADYEN.AS',
    'ORAN.PA', 'SAF.PA', 'EL.PA', 'CAP.PA', 'ORA.PA', 'ENGI.PA', 'BN.PA', 'EN.PA', 'ALO.PA', 'PUB.PA',
    'ULVR.L', 'SAP.DE', 'SIE.DE', 'NOK.HE', 'NESN.SW', 'ROG.SW'
]

global_stable = [
    'MSFT', 'AAPL', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JPM', 'UNH',
    'V', 'XOM', 'PG', 'JNJ', 'HD'
]

all_tickers = euronext_stable + global_stable

st.title("🤖 Robot Advisor : Optimisation Markowitz pour 50k€")

budget = st.sidebar.number_input("Budget (€)", value=50000.0, min_value=1000.0, step=1000.0)
risk_level = st.sidebar.selectbox("Niveau de risque", ["Conservateur", "Modéré", "Agressif"])
target_return_input = st.sidebar.slider("Rendement cible (%)", 5.0, 15.0, 8.0) / 100

# Ajustement cible selon niveau de risque
if risk_level == "Conservateur":
    target_return = 0.06
elif risk_level == "Agressif":
    target_return = 0.12
else:
    target_return = target_return_input

selected_tickers = st.sidebar.multiselect(
    "Sélectionnez les actifs",
    all_tickers,
    default=['AAPL', 'MSFT', 'MC.PA', 'ASML.AS', 'TTE.PA']
)

@st.cache_data(show_spinner=False)
def load_data(tickers):
    if not tickers:
        return pd.DataFrame(), []

    valid_returns = []
    failed_tickers = []

    for ticker in tickers:
        try:
            data = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
            if data.empty:
                data = yf.download(ticker, period="1mo", progress=False, auto_adjust=True)
                st.warning(f"{ticker}: Fallback à 1 mois (5y indisponible)")

            if not data.empty and len(data) > 20:
                close_prices = data['Close'].dropna()
                if len(close_prices) > 20:
                    returns = close_prices.pct_change().dropna()
                    if len(returns) > 10:
                        valid_returns.append(returns.rename(ticker))
                        st.success(f"✅ {ticker}")
                        continue
                    else:
                        failed_tickers.append(f"{ticker} (peu de rendements)")
                else:
                    failed_tickers.append(f"{ticker} (peu de prix)")
            else:
                failed_tickers.append(f"{ticker} (données vides)")
        except Exception as e:
            error_msg = str(e)[:50]
            failed_tickers.append(f"{ticker} (erreur: {error_msg}...)")
            st.error(f"{ticker}: Erreur - {error_msg}")

    if not valid_returns:
        return pd.DataFrame(), failed_tickers

    returns_df = pd.concat(valid_returns, axis=1).dropna()
    return returns_df, failed_tickers

def backtest(weights, returns):
    if returns.empty:
        return 0.0, 0.0, 0.0

    port_returns = np.dot(returns, weights)
    port_returns_series = pd.Series(port_returns, index=returns.index)

    if len(port_returns_series) == 0:
        return 0.0, 0.0, 0.0

    cum_ret = (1 + port_returns_series).cumprod()
    drawdown = (cum_ret / cum_ret.cummax() - 1).min()

    downside = port_returns_series[port_returns_series < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = (port_returns_series.mean() * 252 - 0.02) / downside_std

    total_ret = cum_ret.iloc[-1] - 1
    return total_ret, drawdown, sortino

if st.sidebar.button("Charger Données & Optimiser"):
    if len(selected_tickers) < 1:
        st.error("Sélectionnez au moins 1 actif.")
        st.stop()

    returns, failed_tickers = load_data(selected_tickers)

    if returns.empty:
        st.error("❌ AUCUNE DONNÉE VALIDE. Essayez les defaults.")
        if failed_tickers:
            st.warning("Échecs : " + "; ".join(failed_tickers))
        st.stop()

    if failed_tickers:
        st.warning(f"⚠️ {len(failed_tickers)} échecs : {', '.join(failed_tickers[:3])}{'...' if len(failed_tickers)>3 else ''}")

    selected_tickers = list(returns.columns)
    st.success(f"📊 {len(selected_tickers)} actifs chargés !")

    mu = returns.mean() * 252
    cov = returns.cov() * 252

    def portfolio_perf(weights, mu, cov):
        ret = np.dot(weights, mu)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        return ret, vol

    def optimize_portfolio(mu, cov, target_ret):
        n = len(mu)
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, mu) - target_ret}
        )
        bounds
