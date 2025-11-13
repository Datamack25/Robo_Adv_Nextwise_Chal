import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go

# ========================================
# LISTE DES ACTIFS (75 au total)
# ========================================
euronext_50 = [
    'MC.PA', 'ASML.AS', 'TTE.PA', 'OR.PA', 'RMS.PA', 'AIR.PA', 'SU.PA', 'SAN.PA', 'BNP.PA', 'ADYEN.AS',
    'ORAN.PA', 'SAF.PA', 'EL.PA', 'CAP.PA', 'ORA.PA', 'ENGI.PA', 'ACA.PA', 'BN.PA', 'EN.PA', 'HO.PA',
    'ALO.PA', 'PUB.PA', 'URW.AS', 'ABI.BR', 'GLE.PA', 'VIE.PA', 'KER.PA', 'STLA.MI', 'NEC.PA', 'RF.PA',
    'DG.PA', 'SW.PA', 'ENX.PA', 'BIO.PA', 'TEP.PA', 'SGO.PA', 'HOA.PA', 'ML.PA', 'RI.PA', 'CA.PA',
    'ULVR.L', 'NOVO-B.CO', 'NOK.HE', 'VOLV-B.ST', 'NESN.SW', 'ROG.SW', 'SAP.DE', 'ALV.DE', 'SIE.DE', 'BAS.DE'
]

global_25 = [
    'MSFT', 'AAPL', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO',
    'JPM', 'UNH', 'V', 'XOM', 'PG', 'JNJ', 'HD', 'MA', 'CVX', 'BAC',
    'TSM', 'TCEHY', 'BABA', '0700.HK', '2330.TW'
]

all_tickers = euronext_50 + global_25

# ========================================
# INTERFACE
# ========================================
st.title("Robot Advisor : Optimisation Markowitz pour 50k€")

# Sidebar
budget = st.sidebar.number_input("Budget (€)", value=50000.0, min_value=1000.0, step=1000.0)
risk_level = st.sidebar.selectbox("Niveau de risque", ["Conservateur", "Modéré", "Agressif"])
target_return = st.sidebar.slider("Rendement cible (%)", 5.0, 15.0, 8.0) / 100

# Ajustement automatique du risque
if risk_level == "Conservateur":
    target_return = 0.06
elif risk_level == "Agressif":
    target_return = 0.12

# Sélection des actifs
selected_tickers = st.sidebar.multiselect(
    "Sélectionnez les actifs", 
    all_tickers, 
    default=['AAPL', 'MSFT', 'MC.PA', 'ASML.AS', 'TTE.PA']  # 5 solides par défaut
)

# ========================================
# FONCTION : CHARGEMENT SÉCURISÉ DES DONNÉES
# ========================================
@st.cache_data(show_spinner=False)
def load_data(tickers):
    if not tickers:
        return pd.DataFrame(), []

    valid_returns = []
    failed_tickers = []
    chunk_size = 10

    with st.spinner(f"Téléchargement de {len(tickers)} tickers..."):
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            try:
                data = yf.download(
                    chunk, period="5y", auto_adjust=True, progress=False, 
                    group_by='ticker' if len(chunk) > 1 else None
                )
                
                close_data = data['Close'] if len(chunk) > 1 else data['Close']
                
                for ticker in chunk:
                    if ticker in close_data.columns:
                        series = close_data[ticker].dropna()
                        if len(series) > 500:  # ~2 ans de données
                            returns = series.pct_change().dropna()
                            if len(returns) > 100:
                                valid_returns.append(returns.rename(ticker))
                            else:
                                failed_tickers.append(f"{ticker} (peu de données)")
                        else:
                            failed_tickers.append(f"{ticker} (pas assez de prix)")
                    else:
                        failed_tickers.append(f"{ticker} (introuvable)")
            except Exception as e:
                failed_tickers.extend([f"{t} (erreur)" for t in chunk])

    if not valid_returns:
        return pd.DataFrame(), failed_tickers

    returns_df = pd.concat(valid_returns, axis=1).dropna()
    return returns_df, failed_tickers

# ========================================
# FONCTION : BACKTEST SÉCURISÉ
# ========================================
def backtest(weights, returns):
    if returns.empty or len(returns) == 0:
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
# BOUTON : CHARGER ET ANALYSER
# ========================================
if st.sidebar.button("Charger Données & Optimiser"):
    if len(selected_tickers) == 0:
        st.error("Veuillez sélectionner au moins un actif.")
        st.stop()

    returns, failed_tickers = load_data(selected_tickers)

    if returns.empty:
        st.error("**Aucune donnée valide trouvée.**")
        if failed_tickers:
            st.warning("**Tickers échoués :** " + ", ".join(failed_tickers[:10]) + 
                      ("..." if len(failed_tickers) > 10 else ""))
        st.stop()

    if failed_tickers:
        st.warning(f"**{len(failed_tickers)} tickers ignorés** : {', '.join(failed_tickers[:5])}{'...' if len(failed_tickers)>5 else ''}")

    st.success(f"**{len(returns.columns)} actifs chargés avec succès !**")
    selected_tickers = list(returns.columns)  # Mise à jour

    # === Calculs Markowitz ===
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

    # === Allocation ===
    allocation = pd.DataFrame({
        'Actif': selected_tickers,
        'Poids %': weights * 100,
        'Montant €': weights * budget
    }).round(2)
    allocation = allocation.sort_values('Poids %', ascending=False)

    # === Frontière Efficiente ===
    st.subheader("Frontière Efficiente")
    def efficient_frontier(mu, cov, n_points=50):
        target_rets = np.linspace(max(mu.min(), 0.03), mu.max(), n_points)
        vols = []
        for tr in target_rets:
            w = optimize_portfolio(mu, cov, tr)
            _, v = portfolio_perf(w, mu, cov)
            vols.append(v)
        return target_rets, np.array(vols)

    tr_range, vols = efficient_frontier(mu, cov)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vols, y=tr_range, mode='lines', name='Frontière Efficiente', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[port_vol], y=[port_ret], mode='markers',
                             name='Portefeuille Optimal', marker=dict(size=12, color='red')))
    fig.update_layout(
        title="Frontière Efficiente (Markowitz)",
        xaxis_title="Volatilité annualisée",
        yaxis_title="Rendement attendu",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # === Résultats ===
    st.subheader("Allocation Optimale")
    st.dataframe(
        allocation.style.format({"Poids %": "{:.2f}%", "Montant €": "€{:.0f}"}),
        use_container_width=True
    )

    st.subheader("Performance du Portefeuille")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rendement", f"{port_ret*100:.1f}%")
    col2.metric("Volatilité", f"{port_vol*100:.1f}%")
    col3.metric("Sharpe", f"{sharpe:.2f}")
    col4.metric("Sortino", f"{sortino:.2f}")

    st.info(f"**Backtest 5 ans** : Rendement = {total_ret*100:.1f}% | Drawdown max = {drawdown*100:.1f}%")

else:
    st.info("Sélectionnez des actifs → Cliquez sur **'Charger Données & Optimiser'** pour lancer.")

st.caption("Modèle Markowitz. Performances passées ≠ futures. **Pas un conseil financier.**")
