import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import csv
import yfinance as yf

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Stacked Dashboard", page_icon= "🥞:", layout="wide")

# --- CONSTANTS ---
PRIMARY_GREEN = '#2ECC71'    # Growth / Profit
BENCHMARK_ORANGE = '#E67E22' # Benchmark Line
DANGER_RED = '#E74C3C'       # Loss
NEUTRAL_GREY = '#7F8C8D'     # Invested Line
CATEGORY_COLORS = ['#F1C40F', '#95A5A6', '#3498DB', '#2ECC71', '#E67E22', '#9B59B6']

# Context for benchmarks (Added Currency Info)
BENCHMARK_CONTEXT = {
    "S&P 500": {"ticker": "^GSPC", "currency": "USD", "desc": "US Large Cap (USD)"},
    "Nasdaq-100": {"ticker": "^IXIC", "currency": "USD", "desc": "Tech Heavy (USD)"},
    "TSX Composite (Canada)": {"ticker": "^GSPTSE", "currency": "CAD", "desc": "Canadian Market (CAD)"}
}

# Hardcoded Historical Data (Fallback)
REAL_SP500_DATA = [
    3714.24, 3811.15, 3972.89, 4181.17, 4204.11, 4297.50, 4395.26, 4522.68, 4307.54, 4605.38, 
    4567.00, 4766.18, 4515.55, 4373.94, 4530.41, 4131.93, 4132.15, 3785.38, 4130.29, 3955.00, 
    3585.62, 3871.98, 4080.11, 3839.50, 4076.60, 3970.15, 4109.31, 4169.48, 4179.83, 4450.38, 
    4588.96, 4507.66, 4288.05, 4193.80, 4567.80, 4769.83, 4845.65, 5096.27, 5254.35, 5035.69, 
    5277.51, 5460.48, 5522.30, 5648.40, 5762.48, 5705.45, 6032.38, 5881.63, 6040.53, 5954.50, 
    5611.85, 5569.06, 5911.69, 6204.95, 6339.39, 6460.26, 6688.46, 6840.20, 6849.09, 6845.50, 
    6858.47
]
FALLBACK_DATES = pd.date_range(start='2021-01-01', periods=len(REAL_SP500_DATA), freq='MS')
FALLBACK_BENCH_MAP = dict(zip(FALLBACK_DATES, REAL_SP500_DATA))

# --- 1. DATA PROCESSING FUNCTIONS ---

@st.cache_data
def get_exchange_rate():
    """Fetches USD to CAD rate (CAD=X). Returns 1.40 as fallback."""
    try:
        # Get CAD=X (amount of CAD for 1 USD)
        ticker = yf.Ticker("CAD=X")
        hist = ticker.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
        return 1.40
    except:
        return 1.40

@st.cache_data
def get_benchmark_data(ticker_symbol, start_date):
    """Fetches real historical monthly closing prices from Yahoo Finance."""
    try:
        data = yf.download(ticker_symbol, start=start_date, interval="1mo")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data[['Close']].dropna()
        data.index = data.index.to_period('M').to_timestamp()
        return data['Close'].to_dict()
    except Exception:
        return FALLBACK_BENCH_MAP

@st.cache_data
def parse_data(uploaded_files):
    all_data = []

    def clean_money(s):
        if not isinstance(s, str): return 0.0
        s = s.replace('$', '').replace(',', '').replace('"', '').replace(' ', '')
        if '%' in s or s == '-' or s == '': return 0.0
        try: return float(s)
        except ValueError: return 0.0

    def classify_account(fund_name, bank_name):
        text = (str(fund_name) + " " + str(bank_name)).lower()
        if any(k in text for k in ['first home', 'fhsa']): return 'FHSA'
        if any(k in text for k in ['education', 'resp']): return 'RESP'
        if any(k in text for k in ['tax free', 'tfsa', 'tax-free']): return 'TFSA'
        if any(k in text for k in ['rrsp', 'retirement', 'rsp', 'lira', 'locked-in']): return 'RRSP'
        if any(k in text for k in ['unregistered', 'margin', 'cash', 'individual', 'joint', 'taxable']): return 'Non-Registered'
        return 'Non-Registered'

    for uploaded_file in uploaded_files:
        try:
            stringio = uploaded_file.getvalue().decode("utf-8", errors='ignore')
            reader = csv.reader(stringio.splitlines())
            rows = list(reader)
            current_date = None
            for row in rows:
                if not row: continue
                if len(row) > 0 and re.match(r'^\d{1,2}/\d{1,2}/\d{4}', row[0]):
                    try: current_date = pd.to_datetime(row[0])
                    except: pass
                
                market_value = 0.0
                book_cost = 0.0
                valid_row = False
                if current_date:
                    if len(row) >= 7: 
                        market_value = clean_money(row[6])
                        book_cost = clean_money(row[3])
                        valid_row = True
                    elif len(row) == 5:
                        market_value = clean_money(row[4])
                        book_cost = clean_money(row[3])
                        valid_row = True

                if valid_row and market_value > 0:
                    bank_name = row[1] if len(row) > 1 else ""
                    fund_name = row[2] if len(row) > 2 else ""
                    if fund_name != "" and "Total" not in fund_name:
                        acct_type = classify_account(fund_name, bank_name)
                        all_data.append({'Date': current_date, 'Type': acct_type, 'Value': market_value, 'BookCost': book_cost})
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")

    return pd.DataFrame(all_data).sort_values('Date') if all_data else pd.DataFrame()

def generate_example_data():
    dates = pd.date_range(start='2021-01-01', periods=len(REAL_SP500_DATA), freq='MS')
    all_records = []
    assets = [
        {"Name": "TFSA (S&P 500)", "Monthly": 400, "Prices": REAL_SP500_DATA},
        {"Name": "RRSP (Bonds)",   "Monthly": 300, "Prices": None, "Base": 100, "Growth": 0.003},
        {"Name": "RESP (Education)","Monthly": 200, "Prices": None, "Base": 100, "Growth": 0.005},
        {"Name": "Non-Reg (Tech)", "Monthly": 100, "Prices": None, "Base": 50,  "Growth": 0.012, "Vol": 0.08}
    ]
    units = {a["Name"]: 0.0 for a in assets}
    prices = {a["Name"]: a.get("Base", 100) for a in assets}

    for i, date in enumerate(dates):
        for a in assets:
            name = a["Name"]
            if a["Prices"]:
                price = a["Prices"][i]
            else:
                price = prices[name] * (1 + (np.random.normal(a["Growth"], a.get("Vol", 0))))
                prices[name] = price
            
            units[name] += a["Monthly"] / price
            all_records.append({
                'Date': date, 
                'Type': name, 
                'Value': units[name] * price, 
                'BookCost': (i + 1) * a["Monthly"]
            })
            
    return pd.DataFrame(all_records)

# --- 2. MONTE CARLO SIMULATOR ---

def run_monte_carlo(current_val, monthly_add, years, mean_ret_pct, vol_pct, num_sims=500):
    """Generates future portfolio paths based on geometric brownian motion."""
    months = int(years * 12)
    mu = mean_ret_pct / 12  # Monthly return
    sigma = vol_pct / np.sqrt(12) # Monthly volatility
    
    # Random shocks: matrix of shape [months, num_sims]
    shocks = np.random.normal(mu - 0.5 * sigma**2, sigma, (months, num_sims))
    
    # Initialize paths
    paths = np.zeros((months + 1, num_sims))
    paths[0] = current_val
    
    for t in range(1, months + 1):
        # Growth step
        growth = np.exp(shocks[t-1])
        paths[t] = paths[t-1] * growth
        # Contribution step
        paths[t] += monthly_add
        
    return paths

# --- 3. VISUALIZATION ENGINE ---

def render_dashboard(df, bench_key, bench_info, target_currency, usd_cad_rate):
    if df.empty:
        st.warning("No data available to render.")
        return

    # --- CURRENCY NORMALIZATION LOGIC ---
    # 1. Determine User Conversion Factors (Assuming User Data is CAD)
    user_fx_multiplier = 1.0
    if target_currency == "USD":
        user_fx_multiplier = 1.0 / usd_cad_rate

    # 2. Benchmark FX Logic
    bench_currency = bench_info['currency']
    bench_fx_multiplier = 1.0
    
    if bench_currency == "USD" and target_currency == "CAD":
        bench_fx_multiplier = usd_cad_rate
    elif bench_currency == "CAD" and target_currency == "USD":
        bench_fx_multiplier = 1.0 / usd_cad_rate

    # 3. Apply Conversion to Data
    df_conv = df.copy()
    df_conv['Value'] = df_conv['Value'] * user_fx_multiplier
    df_conv['BookCost'] = df_conv['BookCost'] * user_fx_multiplier

    # --- AGGREGATION ---
    df_total = df_conv.groupby('Date')[['Value', 'BookCost']].sum().reset_index()
    unique_types = sorted(df_conv['Type'].unique())
    color_map = {t: CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i, t in enumerate(unique_types)}
    
    # --- BENCHMARK CALCULATION ---
    bench_data_raw = get_benchmark_data(bench_info['ticker'], df_total['Date'].min())
    
    bench_units = 0
    bench_values = []
    # Get first available price or default
    first_date_match = next(iter(bench_data_raw))
    first_price_raw = bench_data_raw.get(first_date_match, 100.0)
    
    df_total['Incr_Invest'] = df_total['BookCost'].diff().fillna(df_total['BookCost'].iloc[0])
    
    for _, row in df_total.iterrows():
        match_date = row['Date'].replace(day=1) # normalize to start of month
        # Get raw price and apply FX
        raw_price = bench_data_raw.get(match_date, first_price_raw)
        adj_price = raw_price * bench_fx_multiplier
        
        bench_units += row['Incr_Invest'] / adj_price
        bench_values.append(bench_units * adj_price)
    
    df_total['BenchValue'] = bench_values
    latest = df_total.iloc[-1]
    
    # --- KPI Calculations ---
    u_roi = (latest['Value'] / latest['BookCost'] - 1) * 100 if latest['BookCost'] > 0 else 0
    b_roi = (latest['BenchValue'] / latest['BookCost'] - 1) * 100 if latest['BookCost'] > 0 else 0

    # --- TABS LAYOUT ---
    tab1, tab2 = st.tabs(["Portfolio Performance", "Future Simulator (FIRE)"])

    # --- TAB 1: HISTORY ---
    with tab1:
        # 1. Metric Row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Net Worth ({target_currency})", f"${latest['Value']:,.2f}")
        c2.metric("Total Invested", f"${latest['BookCost']:,.2f}")
        c3.metric("My ROI", f"{u_roi:.2f}%")
        c4.metric(f"{bench_key} ROI", f"{b_roi:.2f}%", delta=f"{(u_roi - b_roi):.2f}% vs Market")

        st.markdown("---")

        # 2. Chart: Net Worth vs Invested
        st.subheader("Net Worth vs. Invested (Capital Growth)")
        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(
            x=df_total['Date'], y=df_total['BookCost'], 
            mode='lines', name='Invested', 
            line=dict(color=NEUTRAL_GREY, dash='dash'), 
            hovertemplate='Invested: $%{y:,.2f}'
        ))
        fig_net.add_trace(go.Scatter(
            x=df_total['Date'], y=df_total['Value'], 
            mode='lines', name='Net Worth', 
            fill='tonexty', line=dict(color=PRIMARY_GREEN), 
            hovertemplate='Value: $%{y:,.2f}'
        ))
        fig_net.update_layout(
            hovermode="x unified", template="plotly_white", height=400, 
            yaxis_tickprefix="$", yaxis_tickformat=",.2f"
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # 3. Chart: Benchmark Comparison
        st.subheader(f"Portfolio vs. {bench_key}")
        st.info(f"Comparing your Active Strategy vs. Passive {bench_key} Index Fund.")

        fig_bench = go.Figure()
        fig_bench.add_trace(go.Scatter(
            x=df_total['Date'], y=df_total['BenchValue'], 
            name=f'{bench_key} ({target_currency})', 
            line=dict(color=BENCHMARK_ORANGE, width=2), 
            hovertemplate=f'{bench_key}: '+'$%{y:,.2f}'
        ))
        fig_bench.add_trace(go.Scatter(
            x=df_total['Date'], y=df_total['Value'], 
            name='My Portfolio', 
            line=dict(color=PRIMARY_GREEN, width=3), 
            hovertemplate='My Portfolio: $%{y:,.2f}'
        ))
        fig_bench.update_layout(
            hovermode="x unified", template="plotly_white", height=400, 
            yaxis_tickprefix="$", yaxis_tickformat=",.2f"
        )
        st.plotly_chart(fig_bench, use_container_width=True)

        # 4. Charts: Composition & Returns
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Portfolio Composition")
            fig_comp = px.area(
                df_conv.groupby(['Date', 'Type'])['Value'].sum().reset_index(), 
                x="Date", y="Value", color="Type", 
                color_discrete_map=color_map
            )
            fig_comp.update_layout(
                hovermode="x unified", template="plotly_white", 
                yaxis_tickprefix="$", yaxis_tickformat=",.2f"
            )
            fig_comp.update_traces(hovertemplate='$%{y:,.2f}')
            st.plotly_chart(fig_comp, use_container_width=True)

        with col_b:
            st.subheader("Monthly Market Gain ($)")
            df_total['TotalDiff'] = df_total['Value'].diff().fillna(0)
            df_total['Contribution'] = df_total['BookCost'].diff().fillna(0)
            df_total['MarketGain'] = df_total['TotalDiff'] - df_total['Contribution']
            df_total['BarColor'] = df_total['MarketGain'].apply(lambda x: PRIMARY_GREEN if x >= 0 else DANGER_RED)
            
            fig_bar = go.Figure(go.Bar(
                x=df_total['Date'], y=df_total['MarketGain'], 
                marker_color=df_total['BarColor'], 
                hovertemplate='Gain: $%{y:,.2f}<extra></extra>'
            ))
            fig_bar.update_layout(
                template="plotly_white", 
                yaxis_tickprefix="$", yaxis_tickformat=",.2f"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # 5. Chart: Allocation Donut
        st.subheader("Current Allocation")
        fig_pie = px.pie(
            df[df['Date'] == latest['Date']].groupby('Type')['Value'].sum().reset_index(),
            values='Value', names='Type', hole=0.5,
            color='Type', color_discrete_map=color_map
        )

        fig_pie.update_traces(
            textfont_color='white', textinfo='percent+label',
            insidetextorientation='horizontal',
            hovertemplate='%{label}<br>$%{value:,.2f}<br>%{percent}'
        )

        fig_pie.update_layout(
            annotations=[dict(text=f"${latest['Value']:,.2f}", x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)

        with st.expander("View Raw Data"):
            st.dataframe(df_conv)

    # --- TAB 2: SIMULATOR ---
    with tab2:
        st.subheader("Monte Carlo Wealth Projection")
        st.info("This simulation runs 500 possible market scenarios to estimate your future net worth.")
        
        # Controls
        col_input, col_chart = st.columns([1, 3])
        
        with col_input:
            st.markdown("### Parameters")
            sim_years = st.slider("Years to Grow", 1, 60, 25)
            sim_contrib = st.number_input(f"Monthly Contribution ({target_currency})", value=2000, step=100)
            sim_return = st.slider("Exp. Annual Return (%)", 0.0, 50.0, 7.0) / 100
            sim_vol = st.slider("Volatility (%)", 5.0, 30.0, 15.0) / 100
            
            current_nw = latest['Value']
            st.divider()
            st.metric("Starting Capital", f"${current_nw:,.2f}")
        
        # Run Simulation
        paths = run_monte_carlo(current_nw, sim_contrib, sim_years, sim_return, sim_vol)
        
        # Calculate Percentiles
        p10 = np.percentile(paths, 10, axis=1)
        p50 = np.percentile(paths, 50, axis=1)
        p90 = np.percentile(paths, 90, axis=1)
        
        future_dates = pd.date_range(start=latest['Date'], periods=len(p50), freq='ME')
        
        # Plotting
        with col_chart:
            fig_mc = go.Figure()
            # 90th Percentile (Optimistic)
            fig_mc.add_trace(go.Scatter(
                x=future_dates, y=p90, mode='lines', line=dict(color=BENCHMARK_ORANGE,width=3), name='Upper Bound (90%)', hovertemplate="$%{y:,.0f}"
            ))
            # 10th Percentile (Pessimistic) - Fill area
            fig_mc.add_trace(go.Scatter(
                x=future_dates, y=p10, mode='lines', line=dict(color= DANGER_RED, width=3), fill='tonexty', fillcolor='rgba(46, 204, 113, 0.2)', name='Lower Bound (10%)', hovertemplate="$%{y:,.0f}"
            ))
            # Median
            fig_mc.add_trace(go.Scatter(
                x=future_dates, y=p50, mode='lines', line=dict(color=PRIMARY_GREEN, width=3), name='Median Outcome' , hovertemplate="$%{y:,.0f}"
            ))
            
            final_median = p50[-1]
            fig_mc.update_layout(
                title=f"Projected Median (In {sim_years} Years): ${final_median:,.2f}", 
                template="plotly_white", hovermode="x unified", yaxis_tickprefix="$"
            )
            st.plotly_chart(fig_mc, use_container_width=True)

# --- 4. MAIN CONTROLLER ---

# Initialize Session State
if 'demo_active' not in st.session_state:
    st.session_state.demo_active = False
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

st.title("🥞Stacked: Investment Dashboard")

# Initialize FX
usd_cad_rate = get_exchange_rate()

with st.sidebar:
    st.header("1. Data Import")
    uploaded_files = st.file_uploader("Upload CSV Files", type="csv", accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")
    
    col_demo, col_clear = st.columns(2)
    with col_demo:
        if st.button("Load Demo"):
            st.session_state.demo_active = True
            st.session_state.uploader_key += 1 # Increment key to reset uploader
            st.rerun()
    with col_clear:
        if st.button("Clear Data"):
            st.session_state.demo_active = False
            st.session_state.uploader_key += 1 # Increment key to reset uploader
            st.rerun()

    if uploaded_files:
        st.session_state.demo_active = False

    with st.expander("Template CSV"):
        template = "Date,Bank,Fund Name,Book Cost,Market Value\n01/31/2023,Questrade,TFSA S&P 500,$1000.00,$1050.00\n01/31/2023,Wealthsimple,RESP Education,$500.00,$505.00"
        st.download_button("Download Template", template, "stacked_template.csv", "text/csv")
        
    st.markdown("---")

    # 2. Settings
    st.header("2. Settings")
    
    # Currency Toggle
    target_currency = st.radio("Display Currency", ["CAD", "USD"], horizontal=True)
    st.caption(f"Live Rate: 1 USD = {usd_cad_rate:.2f} CAD")
    
    # Benchmark Selector
    bench_choice = st.selectbox("Compare Against:", list(BENCHMARK_CONTEXT.keys()))
    st.caption(BENCHMARK_CONTEXT[bench_choice]['desc'])


# Logic to Switch Data Source
if st.session_state.demo_active:
    df_to_show = generate_example_data()
    render_dashboard(df_to_show, bench_choice, BENCHMARK_CONTEXT[bench_choice], target_currency, usd_cad_rate)
elif uploaded_files:
    df_to_show = parse_data(uploaded_files)
    render_dashboard(df_to_show, bench_choice, BENCHMARK_CONTEXT[bench_choice], target_currency, usd_cad_rate)
else:
    st.info("Upload your CSV files to begin, or click 'Load Demo' in the sidebar.")