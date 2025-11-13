import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Listes des actifs (tickers yfinance : .PA pour Paris, .AS pour Amsterdam, etc.)
euronext_50 = [
    'MC.PA', 'ASML.AS', 'TTE.PA', 'OR.PA', 'RMS.PA', 'AIR.PA', 'SU.PA', 'SAN.PA', 'BNP.PA', 'ADYEN.AS', # Top 10
    'ORAN.PA', 'SAF.PA', 'EL.PA', 'CAP.PA', 'ORA.PA', 'ENGI.PA', 'ACA.PA', 'BN.PA', 'EN.PA', 'HO.PA', # CAC/AEX
    'ALO.PA', 'PUB.PA', 'URW.AS', 'ABI.BR', 'GLE.PA', 'VIE.PA', 'KER.PA', 'STLA.MI', 'NEC.PA', 'RF.PA', # Divers
    'DG.PA', 'SW.PA', 'ENX.PA', 'BIO.PA', 'TEP.PA', 'SGO.PA', 'HOA.PA', 'ML.PA', 'RI.PA', 'CA.PA', # Mid-caps
    # Europe large (ajustés pour liquidité)
    'ULVR.L', 'NOVO-B.CO', 'NOK.HE', 'VOLV-B.ST', 'NESN.SW', 'ROG.SW', 'SAP.DE', 'ALV.DE', 'SIE.DE', 'BAS.DE'
] # 50 actifs Euronext-ish (top market cap/liquidité nov. 2025)
global_25 = [
    'MSFT', 'AAPL', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO', # US Tech
    'JPM', 'UNH', 'V', 'XOM', 'PG', 'JNJ', 'HD', 'MA', 'CVX', 'BAC', # US Divers
    # Asie (Taiwan Semi, Tencent, Alibaba, Tencent HK, TSMC)
    'TSM', 'TCEHY', 'BABA', '0700.HK', '2330.TW'
] # 25 top global hors Europe (market cap >$200 Bn, nov. 2025)
all_tickers = euronext_50 + global_25

st.title("🤖 Robot Advisor : Optimisation Markowitz pour 50k€")

# Sidebar pour inputs
budget = st.sidebar.number_input("Budget (€)", value=50000.0, min_value=1000.0)
risk_level = st.sidebar.selectbox(
    "Niveau de risque", ["Conservateur", "Modéré", "Agressif"])
target_return = st.sidebar.slider(
    "Rendement cible annualisé (%)", 5.0, 15.0, 8.0) / 100

# Ajustement automatique du target_return basé sur le risque (AVANT le rerun)
if risk_level == "Conservateur":
    target_return = 0.06
elif risk_level == "Agressif":
    target_return = 0.12

selected_tickers = st.sidebar.multiselect(
    # Limite pour perf
    "Sélectionnez actifs (défaut: tous)", all_tickers, default=all_tickers[:20])

# Téléchargement données (5 ans) - CORRIGÉ : Utilise 'Close' avec auto_adjust=True
@st.cache_data
def load_data(tickers):
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers, period="5y", auto_adjust=True, progress=False)
    returns = data['Close'].pct_change().dropna()
    return returns

if st.sidebar.button("Charger Données"):
    with st.spinner("Téléchargement..."):
        returns = load_data(selected_tickers)
    
    if returns.empty:
        st.error("Aucune donnée chargée. Vérifiez les tickers.")
        st.stop()
    
    st.success(f"Données chargées pour {len(selected_tickers)} actifs.")
    
    # Calculs Markowitz
    mu = returns.mean() * 252 # Rendements annualisés
    cov = returns.cov() * 252 # Covariance annualisée
    
    def portfolio_perf(weights, mu, cov):
        ret = np.dot(weights, mu)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        return ret, vol
    
    def neg_sharpe(weights, mu, cov, rf=0.02):
        ret, vol = portfolio_perf(weights, mu, cov)
        return - (ret - rf) / vol if vol > 0 else np.inf
    
    def optimize_portfolio(mu, cov, target_ret):
        n = len(mu)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                       {'type': 'eq', 'fun': lambda x: np.dot(x, mu) - target_ret})
        bounds = tuple((0, 1) for _ in range(n))
        result = minimize(lambda x: np.dot(x.T, np.dot(cov, x)), np.ones(n)/n, method='SLSQP',
                          bounds=bounds, constraints=constraints)
        return result.x if result.success else np.ones(n)/n
    
    # Optimisation
    weights = optimize_portfolio(mu, cov, target_return)
    port_ret, port_vol = portfolio_perf(weights, mu, cov)
    sharpe = (port_ret - 0.02) / port_vol if port_vol > 0 else 0
    
    # Mesures de performance (sur backtest simple) - Cache déplacé ici pour éviter bugs
    def backtest(weights, returns):
        port_returns = np.dot(returns, weights)
        cum_ret = (1 + port_returns).cumprod()
        drawdown = (cum_ret / cum_ret.cummax() - 1).min()
        downside = port_returns[port_returns < 0].std() * np.sqrt(252)
        sortino = (port_returns.mean() * 252 - 0.02) / downside if downside > 0 else 0
        return cum_ret.iloc[-1] - 1, drawdown, sortino
    
    total_ret, drawdown, sortino = backtest(weights, returns)
    
    # Allocation en €
    allocation = pd.DataFrame(
        {'Actif': selected_tickers, 'Poids %': weights * 100, 'Montant €': weights * budget})
    allocation = allocation.sort_values('Poids %', ascending=False)
    
    # Visualisations
    st.subheader("Frontière Efficiente (Markowitz)")
    def efficient_frontier(mu, cov, n_points=50):
        target_returns = np.linspace(mu.min(), mu.max(), n_points)
        vols = []
        for tr in target_returns:
            w = optimize_portfolio(mu, cov, tr)
            _, v = portfolio_perf(w, mu, cov)
            vols.append(v)
        return target_returns, np.array(vols)
    
    tr_range, vols = efficient_frontier(mu, cov)
    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(x=vols, y=tr_range,
                     mode='lines', name='Frontière Efficiente'))
    fig_ef.add_trace(go.Scatter(x=[port_vol], y=[port_ret], mode='markers', name='Portefeuille Optimal', marker=dict(size=10, color='red')))
    fig_ef.update_layout(title='Efficient Frontier',
                         xaxis_title='Volatilité (%)', yaxis_title='Rendement (%)')
    st.plotly_chart(fig_ef)
    
    st.subheader("Allocation Optimale")
    st.dataframe(allocation)
    
    st.subheader("Mesures de Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rendement Attendu", f"{port_ret*100:.1f}%")
    col2.metric("Volatilité", f"{port_vol*100:.1f}%")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col4.metric("Sortino Ratio", f"{sortino:.2f}")
    st.info(
        f"Drawdown max historique: {drawdown*100:.1f}% | Rendement backtest 5 ans: {total_ret*100:.1f}%")
else:
    st.info("Cliquez sur 'Charger Données' pour démarrer l'optimisation.")

st.caption("Modèle basé sur Markowitz. Avertissement: Pas de conseil financier ; performances passées ≠ futures. Consultez un pro.")
