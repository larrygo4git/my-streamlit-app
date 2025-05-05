import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Set page configuration
st.set_page_config(
    page_title="Nike vs Adidas Performance Dashboard",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply dark theme CSS for better visibility
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        color: white;
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        gap: 1px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #404040;
        color: #FF4B4B;
        border-bottom: 2px solid #FF4B4B;
    }
    .metric-card {
        background-color: #262730;
        color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    .data-frame {
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }
    p {
        color: #E0E0E0;
    }
    .stMarkdown {
        color: white;
    }
    div[data-testid="stMetricValue"] {
        color: white;
    }
    .css-50ug3q {
        font-size: 16px;
        color: #31333F;
    }
    .stDataFrame {
        color: white;
    }
    /* Make sure dataframe is visible with light text on dark background */
    .dataframe {
        color: white;
    }
    /* Make select boxes more visible */
    .stSelectbox label {
        color: white !important;
    }
    /* Make multiselect more visible */
    .stMultiSelect label {
        color: white !important;
    }
    /* Make date input more visible */
    .stDateInput label {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to generate stock data
def generate_stock_data():
    # Define date range (2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Generate data with some correlation
    np.random.seed(42)
    
    # Base values
    nike_price = 100
    adidas_price = 80
    
    nike_prices = [nike_price]
    adidas_prices = [adidas_price]
    
    # Generate price movements with correlation
    for i in range(1, len(dates)):
        nike_change = np.random.normal(0.0003, 0.015)
        nike_price = nike_price * (1 + nike_change)
        nike_prices.append(nike_price)
        
        # Adidas price movements (correlated with Nike)
        correlation = 0.7
        adidas_change = correlation * nike_change + (1 - correlation) * np.random.normal(0.0002, 0.014)
        adidas_price = adidas_price * (1 + adidas_change)
        adidas_prices.append(adidas_price)
    
    # Create dataframes
    nike_df = pd.DataFrame({
        'Date': dates,
        'Close': nike_prices,
        'Volume': np.random.randint(1000000, 5000000, len(dates)),
        'Company': 'Nike'
    })
    
    adidas_df = pd.DataFrame({
        'Date': dates,
        'Close': adidas_prices,
        'Volume': np.random.randint(800000, 4000000, len(dates)),
        'Company': 'Adidas'
    })
    
    # Combine into single dataframe
    stock_df = pd.concat([nike_df, adidas_df])
    
    return stock_df

# Generate financial metrics data
def generate_financial_metrics():
    metrics = {
        'Metric': ['Revenue (B)', 'Gross Margin (%)', 'Operating Margin (%)', 
                   'Net Income (B)', 'ROE (%)', 'Market Cap (B)'],
        'Nike': [44.5, 45.5, 15.2, 5.9, 42.7, 167.2],
        'Adidas': [21.9, 50.3, 8.5, 1.8, 15.6, 34.5],
        'Industry Avg': [15.6, 43.7, 12.1, 2.2, 23.4, 45.6]
    }
    
    return pd.DataFrame(metrics)

# Generate market share data
def generate_market_share():
    companies = ['Nike', 'Adidas', 'Puma', 'Under Armour', 'Others']
    share = [27.4, 16.8, 5.9, 3.8, 46.1]
    
    return pd.DataFrame({'Company': companies, 'Share': share})

# Generate regional sales data
def generate_regional_sales():
    regions = ['North America', 'Europe', 'Asia', 'Rest of World']
    
    nike_data = {
        'Region': regions,
        'Revenue': [17.2, 12.6, 8.3, 6.4],
        'Company': 'Nike'
    }
    
    adidas_data = {
        'Region': regions,
        'Revenue': [5.9, 8.3, 5.1, 2.6],
        'Company': 'Adidas'
    }
    
    # Combine data
    regional_df = pd.DataFrame(nike_data)
    regional_df = pd.concat([regional_df, pd.DataFrame(adidas_data)])
    
    return regional_df

# Generate all data
stock_data = generate_stock_data()
financial_metrics = generate_financial_metrics()
market_share = generate_market_share()
regional_sales = generate_regional_sales()

# Dashboard title
st.title("Nike vs Adidas: Athletic Wear Industry Analysis")

# Introduction
st.markdown("""
This dashboard provides comparative analysis of Nike and Adidas performance across multiple metrics 
in the athletic wear industry. Use the tabs below to explore different aspects of their business performance.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📈 Stock Performance", "📊 Financial Analysis", "🌍 Market Analysis"])

# Tab 1: Stock Performance
with tab1:
    st.header("Stock Performance")
    
    # Date filter
    date_min = stock_data['Date'].min()
    date_max = stock_data['Date'].max()
    
    date_range = st.date_input(
        "Select Date Range",
        value=(date_min.date(), date_max.date()),
        min_value=date_min.date(),
        max_value=date_max.date()
    )
    
    # Filter data based on date selection
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_stock = stock_data[
            (stock_data['Date'] >= pd.Timestamp(start_date)) & 
            (stock_data['Date'] <= pd.Timestamp(end_date))
        ]
    else:
        filtered_stock = stock_data
    
    # Pivot data for plotting
    pivot_df = filtered_stock.pivot(index='Date', columns='Company', values='Close')
    
    # Calculate returns
    nike_return = ((pivot_df['Nike'].iloc[-1] / pivot_df['Nike'].iloc[0]) - 1) * 100
    adidas_return = ((pivot_df['Adidas'].iloc[-1] / pivot_df['Adidas'].iloc[0]) - 1) * 100
    
    # Display KPIs
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Nike Return", f"{nike_return:.2f}%", f"{nike_return - adidas_return:.2f}% vs Adidas")
    
    with col2:
        st.metric("Adidas Return", f"{adidas_return:.2f}%", f"{adidas_return - nike_return:.2f}% vs Nike")
    
    # Stock price chart
    st.subheader("Stock Price Comparison")
    
    # Select additional options
    show_ma = st.checkbox("Show Moving Average", value=True)
    
    if show_ma:
        ma_period = st.select_slider("Moving Average Period", options=[5, 10, 20, 50, 100, 200], value=50)
        # Calculate moving averages
        pivot_df['Nike_MA'] = pivot_df['Nike'].rolling(window=ma_period).mean()
        pivot_df['Adidas_MA'] = pivot_df['Adidas'].rolling(window=ma_period).mean()
    
    # Create the plot
    fig = go.Figure()
    
    # Add Nike line
    fig.add_trace(
        go.Scatter(
            x=pivot_df.index,
            y=pivot_df['Nike'],
            name="Nike",
            line=dict(color='#FF4B4B', width=2),
        )
    )
    
    # Add Adidas line
    fig.add_trace(
        go.Scatter(
            x=pivot_df.index,
            y=pivot_df['Adidas'],
            name="Adidas",
            line=dict(color='#0066FF', width=2),
        )
    )
    
    # Add moving averages if selected
    if show_ma:
        fig.add_trace(
            go.Scatter(
                x=pivot_df.index,
                y=pivot_df['Nike_MA'],
                name=f"Nike {ma_period}-day MA",
                line=dict(color='#FF4B4B', width=1, dash='dash'),
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=pivot_df.index,
                y=pivot_df['Adidas_MA'],
                name=f"Adidas {ma_period}-day MA",
                line=dict(color='#0066FF', width=1, dash='dash'),
            )
        )
    
    # Update layout with dark theme
    fig.update_layout(
        height=500,
        title=f"Nike vs Adidas Stock Price",
        xaxis_title="Date",
        yaxis_title="Stock Price ($)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        template="plotly_dark",  # Use dark template
        plot_bgcolor='rgba(17, 17, 17, 0.8)',
        paper_bgcolor='rgba(17, 17, 17, 0.8)',
        font=dict(color='white'),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Price Correlation Analysis")
    
    # Calculate daily returns
    returns_df = pivot_df.pct_change().dropna()
    correlation = returns_df['Nike'].corr(returns_df['Adidas'])
    
    # Create scatter plot
    fig = px.scatter(
        x=returns_df['Nike'],
        y=returns_df['Adidas'],
        trendline="ols",
        labels={"x": "Nike Daily Returns", "y": "Adidas Daily Returns"},
        title="Nike vs Adidas Returns Correlation"
    )
    
    # Update with dark theme
    fig.update_layout(
        height=400,
        xaxis=dict(
            title="Nike Daily Returns",
            tickformat=".1%",
        ),
        yaxis=dict(
            title="Adidas Daily Returns",
            tickformat=".1%",
        ),
        template="plotly_dark",  # Use dark template
        plot_bgcolor='rgba(17, 17, 17, 0.8)',
        paper_bgcolor='rgba(17, 17, 17, 0.8)',
        font=dict(color='white'),
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Correlation Coefficient", f"{correlation:.4f}")
        
        if correlation >= 0.7:
            st.markdown("**Strong positive correlation** indicates that Nike and Adidas stocks tend to move together.")
        elif correlation >= 0.4:
            st.markdown("**Moderate positive correlation** suggests some common factors drive both stocks.")
        else:
            st.markdown("**Weak correlation** indicates relatively independent price movements.")
        st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Financial Analysis
with tab2:
    st.header("Financial Performance")
    
    # Financial metrics table
    st.subheader("Key Financial Metrics")
    st.dataframe(financial_metrics, use_container_width=True)
    
    # Financial metrics visualization
    st.subheader("Financial Metrics Comparison")
    
    # Select metrics to visualize
    selected_metrics = st.multiselect(
        "Select Metrics to Compare",
        options=financial_metrics['Metric'].tolist(),
        default=['Gross Margin (%)', 'Operating Margin (%)', 'ROE (%)']
    )
    
    if selected_metrics:
        # Filter the dataframe for selected metrics
        filtered_metrics = financial_metrics[financial_metrics['Metric'].isin(selected_metrics)]
        
        # Create comparison chart
        fig = go.Figure()
        
        # Define width for grouped bars
        bar_width = 0.25
        
        # Define positions for each group
        positions = np.arange(len(filtered_metrics))
        
        # Add Nike bars
        fig.add_trace(go.Bar(
            x=positions - bar_width,
            y=filtered_metrics['Nike'],
            name='Nike',
            marker_color='#FF4B4B',
            width=bar_width,
            text=filtered_metrics['Nike'],
            textposition='outside',
        ))
        
        # Add Adidas bars
        fig.add_trace(go.Bar(
            x=positions,
            y=filtered_metrics['Adidas'],
            name='Adidas',
            marker_color='#0066FF',
            width=bar_width,
            text=filtered_metrics['Adidas'],
            textposition='outside',
        ))
        
        # Add Industry Average bars
        fig.add_trace(go.Bar(
            x=positions + bar_width,
            y=filtered_metrics['Industry Avg'],
            name='Industry Average',
            marker_color='#888888',
            width=bar_width,
            text=filtered_metrics['Industry Avg'],
            textposition='outside',
        ))
        
        # Update layout with dark theme
        fig.update_layout(
            xaxis=dict(
                title="Metrics",
                tickvals=positions,
                ticktext=filtered_metrics['Metric'],
                tickangle=-45,
            ),
            yaxis=dict(title="Value"),
            barmode='group',
            title="Financial Metrics Comparison",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            height=500,
            template="plotly_dark",  # Use dark template
            plot_bgcolor='rgba(17, 17, 17, 0.8)',
            paper_bgcolor='rgba(17, 17, 17, 0.8)',
            font=dict(color='white'),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Market share analysis
    st.subheader("Athletic Wear Market Share")
    
    # Create pie chart
    fig = px.pie(
        market_share,
        values='Share',
        names='Company',
        title='Global Athletic Wear Market Share (%)',
        color='Company',
        color_discrete_map={
            'Nike': '#FF4B4B',
            'Adidas': '#0066FF',
            'Puma': '#7CFC00',
            'Under Armour': '#000000',
            'Others': '#D3D3D3'
        },
        hole=0.4,
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+percent+value',
    )
    
    # Update layout with dark theme
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(17, 17, 17, 0.8)',
        paper_bgcolor='rgba(17, 17, 17, 0.8)',
        font=dict(color='white'),
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Market Analysis
with tab3:
    st.header("Market Analysis")
    
    # Regional sales analysis
    st.subheader("Regional Sales Analysis")
    
    # Create grouped bar chart
    fig = px.bar(
        regional_sales,
        x='Region',
        y='Revenue',
        color='Company',
        barmode='group',
        title='Regional Revenue Distribution (Billions $)',
        color_discrete_map={'Nike': '#FF4B4B', 'Adidas': '#0066FF'},
        text='Revenue',
    )
    
    fig.update_traces(
        texttemplate='$%{text}B',
        textposition='outside'
    )
    
    # Update layout with dark theme
    fig.update_layout(
        xaxis_title="Region",
        yaxis_title="Revenue (Billions $)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=500,
        template="plotly_dark",  # Use dark template
        plot_bgcolor='rgba(17, 17, 17, 0.8)',
        paper_bgcolor='rgba(17, 17, 17, 0.8)',
        font=dict(color='white'),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Product category analysis
    st.subheader("Product Category Analysis")
    
    # Create product category data
    categories = ['Footwear', 'Apparel', 'Accessories', 'Equipment']
    
    nike_revenue = [24.8, 15.3, 3.2, 1.2]
    adidas_revenue = [10.8, 8.4, 1.9, 0.8]
    
    category_df = pd.DataFrame({
        'Category': categories + categories,
        'Company': ['Nike']*4 + ['Adidas']*4,
        'Revenue': nike_revenue + adidas_revenue
    })
    
    # Create grouped bar chart
    fig = px.bar(
        category_df,
        x='Category',
        y='Revenue',
        color='Company',
        barmode='group',
        title='Product Category Revenue (Billions $)',
        color_discrete_map={'Nike': '#FF4B4B', 'Adidas': '#0066FF'},
        text='Revenue',
    )
    
    fig.update_traces(
        texttemplate='$%{text}B',
        textposition='outside'
    )
    
    # Update layout with dark theme
    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Revenue (Billions $)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=500,
        template="plotly_dark",  # Use dark template
        plot_bgcolor='rgba(17, 17, 17, 0.8)',
        paper_bgcolor='rgba(17, 17, 17, 0.8)',
        font=dict(color='white'),
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #E0E0E0;">
    <p>Nike vs Adidas Performance Dashboard | Created with Streamlit</p>
</div>
""", unsafe_allow_html=True)
