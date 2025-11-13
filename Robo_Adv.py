import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go

# ========================================
# LISTE NETTOYÉE : 30 TICKERS STABLES (US + Euronext)
# ========================================
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

# ========================================
# INTERFACE
# ========================================
st.title("🤖 Robot Advisor : Optimisation Markowitz pour 50k€")

budget = st.sidebar.number_input("Budget (€)", value=50000.0, min_value=1000.0, step=1000.0)
risk_level = st.sidebar.selectbox("Niveau de risque", ["Conservateur", "Modéré", "Agressif"])
target_return = st.sidebar.slider("Rendement cible (%)", 5.0, 15.0, 8.0) / 100

if risk_level == "Conservateur":
    target_return = 0.06
elif risk_level == "Agressif":
    target_return = 0.12

selected_tickers = st.sidebar.multiselect(
    "Sélectionnez les actifs", 
    all_tickers, 
    default=['AAPL', 'MSFT', 'MC.PA', 'ASML.AS', 'TTE.PA']
)

# ========================================
# FONCTION LOAD_DATA : FIXÉE (SANS group_by)
# ========================================
@st.cache_data(show_spinner=False)
def load_data(tickers):
    if not tickers:
        return pd.DataFrame(), []

    valid_returns = []
    failed_tickers = []

    for ticker in tickers:
        try:
            # Essai 5y d'abord
            data = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
            if data.empty:
                # Fallback 1mo
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

# ========================================
# BACKTEST SÉCURISÉ
# ========================================
def backtest(weights, returns):
    if returns.empty:
        return 0.0, 0.0, 0.0

    port_returns = np.dot(returns, weights)
    port_returns = pd.Series(port_returns, index=returns.index)

    if len(port_returns) == 0:
        return 0.0, 0.0, 0.0

    cum_ret = (1 + port_returns).cumprod()
    drawdown = (cum_ret / cum_ret.cummax() - 1).min()

    downside = port_returns[port_returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = (port_returns.mean() * 252 - 0.02) / downside_std

    total_ret = cum_ret.iloc[-1] - 1
    return total_ret, drawdown, sortino

# ========================================
# OPTIMISATION
# ========================================
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
        bounds = tuple((0, 1) for _ in range(n))
        init = np.ones(n) / n
        result = minimize(
            lambda x: np.dot(x.T, np.dot(cov, x)),
            init, method='SLSQP', bounds=bounds, constraints=constraints
        )
        return result.x if result.success else init

    weights = optimize_portfolio(mu, cov, target_return)
    port_ret, port_vol = portfolio_perf(weights, mu, cov)
    sharpe = (port_ret - 0.02) / port_vol if port_vol > 0 else 0
    total_ret, drawdown, sortino = backtest(weights, returns)

    allocation = pd.DataFrame({
        'Actif': selected_tickers,
        'Poids %': (weights * 100).round(2),
        'Montant €': (weights * budget).round(0)
    }).sort_values('Poids %', ascending=False)

    st.subheader("📈 Frontière Efficiente")
    def efficient_frontier(mu, cov, n_points=50):
        target_rets = np.linspace(mu.min(), mu.max(), n_points)
        vols = []
        for tr in target_rets:
            w = optimize_portfolio(mu, cov, tr)
            _, v = portfolio_perf(w, mu, cov)
            vols.append(v)
        return target_rets, np.array(vols)

    tr_range, vols = efficient_frontier(mu, cov)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vols, y=tr_range, mode='lines', name='Frontière', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[port_vol], y=[port_ret], mode='markers', name='Optimal', marker=dict(size=12, color='red')))
    fig.update_layout(title="Frontière Efficiente", xaxis_title="Volatilité", yaxis_title="Rendement", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💼 Allocation Optimale")
    st.dataframe(allocation.style.format({"Poids %": "{:.1f}%", "Montant €": "€{:,.0f}"}))

    st.subheader("📊 Performances")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rendement", f"{port_ret*100:.1f}%")
    col2.metric("Volatilité", f"{port_vol*100:.1f}%")
    col3.metric("Sharpe", f"{sharpe:.2f}")
    col4.metric("Sortino", f"{sortino:.2f}")

    st.info(f"Backtest : {total_ret*100:.1f}% | Drawdown : {drawdown*100:.1f}%")

else:
    st.info("Sélectionnez + Cliquez 'Charger'.")

st.caption("Markowitz. Pas de conseil financier.")
