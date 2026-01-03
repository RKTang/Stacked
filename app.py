import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
import csv
import yfinance as yf

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Stacked Dashboard", layout="wide")

# --- CONSTANTS ---
PRIMARY_GREEN = '#2ECC71'    # Growth / Profit
BENCHMARK_ORANGE = '#E67E22' # Benchmark Line
DANGER_RED = '#E74C3C'       # Loss
NEUTRAL_GREY = '#7F8C8D'     # Invested Line
CATEGORY_COLORS = ['#F1C40F', '#95A5A6', '#3498DB', '#2ECC71', '#E67E22', '#9B59B6']

# Context for benchmarks
BENCHMARK_CONTEXT = {
    "S&P 500": "Tracks the 500 largest companies in the US. The standard benchmark for overall market performance.",
    "Nasdaq-100": "Tracks the 100 largest non-financial companies on Nasdaq. Heavily weighted towards Technology and Growth.",
    "TSX Composite (Canada)": "Tracks the largest companies on the Toronto Stock Exchange. Represents the broad Canadian market (Banks, Energy, Mining)."
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

# --- 2. VISUALIZATION ENGINE ---

def render_dashboard(df, bench_ticker, bench_name):
    if df.empty:
        st.warning("No data available to render.")
        return

    # A. Aggregate Data
    df_total = df.groupby('Date')[['Value', 'BookCost']].sum().reset_index()
    unique_types = sorted(df['Type'].unique())
    color_map = {t: CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i, t in enumerate(unique_types)}
    
    # B. Benchmark Calculation
    bench_data = get_benchmark_data(bench_ticker, df_total['Date'].min())
    df_total['Incr_Invest'] = df_total['BookCost'].diff().fillna(df_total['BookCost'].iloc[0])
    
    bench_units = 0
    bench_values = []
    # Fallback price if data missing
    first_price = next(iter(bench_data.values())) if bench_data else 100.0
    
    for _, row in df_total.iterrows():
        match_date = row['Date'].replace(day=1)
        price = bench_data.get(match_date, first_price)
        bench_units += row['Incr_Invest'] / price
        bench_values.append(bench_units * price)
    
    df_total['BenchValue'] = bench_values
    latest = df_total.iloc[-1]
    
    # C. KPI Calculations
    u_roi = (latest['Value'] / latest['BookCost'] - 1) * 100 if latest['BookCost'] > 0 else 0
    b_roi = (latest['BenchValue'] / latest['BookCost'] - 1) * 100 if latest['BookCost'] > 0 else 0

    # --- RENDER UI ---
    
    # 1. Metric Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Worth", f"${latest['Value']:,.2f}")
    c2.metric("Total Invested", f"${latest['BookCost']:,.2f}")
    c3.metric("My ROI", f"{u_roi:.2f}%")
    c4.metric(f"{bench_name} ROI", f"{b_roi:.2f}%", delta=f"{(u_roi - b_roi):.2f}% vs Market")

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

    # 3. Chart: Benchmark Comparison (WITH CONTEXT)
    st.subheader(f"Portfolio vs. {bench_name} (Value Comparison)")
    
    # --- ADDED CONTEXT SECTION ---
    st.info(f"""
    **Strategy Comparison:** Active (Your Portfolio) vs. Passive ({bench_name})
    
    This simulation answers the question: *'What if I had invested every dollar into the {bench_name} instead?'*
    It helps you calculate the **opportunity cost** of your specific stock picks and timing.
    """)
    # -----------------------------

    fig_bench = go.Figure()
    fig_bench.add_trace(go.Scatter(
        x=df_total['Date'], y=df_total['BenchValue'], 
        name=bench_name, 
        line=dict(color=BENCHMARK_ORANGE, width=2), 
        hovertemplate=f'{bench_name}: '+'$%{y:,.2f}'
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
            df.groupby(['Date', 'Type'])['Value'].sum().reset_index(), 
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
        st.dataframe(df)

# --- 3. MAIN CONTROLLER ---

# Initialize Session State
if 'demo_active' not in st.session_state:
    st.session_state.demo_active = False
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

st.title("Stacked: Investment Dashboard")

with st.sidebar:
    # 1. Data Import (TOP PRIORITY)
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

    # 2. Settings (SECONDARY)
    st.header("2. Benchmark Settings")
    bench_choice = st.selectbox("Compare Against:", ["S&P 500", "Nasdaq-100", "TSX Composite (Canada)"])
    st.caption(BENCHMARK_CONTEXT[bench_choice])
    bench_map = {"S&P 500": "^GSPC", "Nasdaq-100": "^IXIC", "TSX Composite (Canada)": "^GSPTSE"}


# Logic to Switch Data Source
if st.session_state.demo_active:
    df_to_show = generate_example_data()
    render_dashboard(df_to_show, bench_map[bench_choice], bench_choice)
elif uploaded_files:
    df_to_show = parse_data(uploaded_files)
    render_dashboard(df_to_show, bench_map[bench_choice], bench_choice)
else:
    st.info("Upload your CSV files to begin, or click 'Load Demo' in the sidebar.")