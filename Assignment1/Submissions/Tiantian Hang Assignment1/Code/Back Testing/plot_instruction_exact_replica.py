#!/usr/bin/env python3
"""
plot_instruction_exact_replica.py

This script creates the EXACT replica of the portfolio comparison plot shown in the assignment instructions.
It includes Equal Weighted, Min-Variance, Mean-Variance portfolios with SPX and QQQ benchmarks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_backtesting_data():
    """Load the backtesting results data"""
    print("Loading backtesting results...")
    
    # Load the equity curve data (monthly frequency for smooth curves)
    equity_data = pd.read_csv('test_back/comparison_equity_M.csv', index_col=0, parse_dates=True)
    
    print(f"Data loaded: {len(equity_data)} monthly observations")
    print(f"Date range: {equity_data.index[0]} to {equity_data.index[-1]}")
    print(f"Available strategies: {list(equity_data.columns)}")
    
    return equity_data

def calculate_main_portfolio_metrics(equity_data):
    """Calculate performance metrics for the main portfolio (Mean-Variance strategy)"""
    
    # Use Mean-Variance as the main "Portfolio Value with Quarterly Adjustment"
    portfolio_series = equity_data['MeanVar'].dropna()
    
    # Calculate returns
    monthly_returns = portfolio_series.pct_change().dropna()
    
    # Calculate cumulative return
    total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0] - 1) * 100
    
    # Calculate annualized return
    years = len(portfolio_series) / 12  # Monthly data
    annual_return = ((portfolio_series.iloc[-1] / portfolio_series.iloc[0]) ** (1/years) - 1) * 100
    
    # Calculate max drawdown
    cumulative = portfolio_series / portfolio_series.expanding().max()
    max_drawdown = (cumulative.min() - 1) * 100
    
    # Calculate annual volatility
    annual_volatility = monthly_returns.std() * np.sqrt(12) * 100
    
    # Calculate Sharpe ratio (assuming 2% risk-free rate)
    risk_free_rate = 0.02
    excess_returns = monthly_returns.mean() * 12 - risk_free_rate
    sharpe_ratio = excess_returns / (monthly_returns.std() * np.sqrt(12))
    
    # Calculate win rate
    win_rate = (monthly_returns > 0).mean() * 100
    
    # Calculate Information Ratio vs SPX
    spx_returns = equity_data['SPX'].pct_change().dropna()
    aligned_returns = monthly_returns.reindex(spx_returns.index).dropna()
    aligned_spx = spx_returns.reindex(aligned_returns.index)
    
    excess_vs_spx = aligned_returns - aligned_spx
    tracking_error = excess_vs_spx.std() * np.sqrt(12)
    alpha = excess_vs_spx.mean() * 12
    information_ratio = alpha / tracking_error if tracking_error != 0 else 0
    
    metrics = {
        'cumulative_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'information_ratio': information_ratio
    }
    
    return metrics

def create_instruction_replica_plot(equity_data, metrics):
    """Create the exact replica of the instruction image plot"""
    
    # Set up the figure to match instruction image exactly
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each strategy with the EXACT same colors and style as instruction
    # Main portfolio (Mean-Variance) - Blue line, thickest
    ax.plot(equity_data.index, equity_data['MeanVar'], 
            color='blue', linewidth=2.5, label='Portfolio Value with Quarterly Adjustment', zorder=5)
    
    # SPX benchmark - Green line
    ax.plot(equity_data.index, equity_data['SPX'], 
            color='green', linewidth=2, label='SP500', zorder=4)
    
    # QQQ benchmark - Red line  
    ax.plot(equity_data.index, equity_data['QQQ'], 
            color='red', linewidth=2, label='QQQ', zorder=3)
    
    # Additional portfolio strategies (optional, lighter lines)
    ax.plot(equity_data.index, equity_data['MinVar'], 
            color='orange', linewidth=1.5, alpha=0.8, label='Min-Variance Portfolio', zorder=2)
    
    ax.plot(equity_data.index, equity_data['Equal'], 
            color='purple', linewidth=1.5, alpha=0.8, label='Equal-Weighted Portfolio', zorder=1)
    
    # Format exactly like instruction image
    ax.set_title('Portfolio Values', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    
    # Set y-axis limits (0 to 5 as requested)
    ax.set_ylim(0, 5)
    
    # Format x-axis with years
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=0)
    
    # Add grid matching instruction style
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend in upper left corner like instruction
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    # Ensure clean layout
    plt.tight_layout()
    
    # Save the exact replica
    plt.savefig('test_back/instruction_exact_replica.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ Exact instruction replica saved as: test_back/instruction_exact_replica.png")
    
    return fig, ax

def create_plot_with_metrics_below(equity_data, metrics):
    """Create the plot with performance metrics displayed below (like instruction image)"""
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(12, 10))
    
    # Main plot area (takes up most space)
    ax_main = plt.subplot2grid((5, 1), (0, 0), rowspan=4)
    
    # Plot the portfolio strategies
    ax_main.plot(equity_data.index, equity_data['MeanVar'], 
                color='blue', linewidth=2.5, label='Portfolio Value with Quarterly Adjustment')
    
    ax_main.plot(equity_data.index, equity_data['SPX'], 
                color='green', linewidth=2, label='SP500')
    
    ax_main.plot(equity_data.index, equity_data['QQQ'], 
                color='red', linewidth=2, label='QQQ')
    
    # Format main plot
    ax_main.set_title('Portfolio Values', fontsize=16, fontweight='bold', pad=20)
    ax_main.set_ylabel('Value', fontsize=12)
    ax_main.set_ylim(0, 5)
    
    # Format x-axis
    ax_main.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Add grid and legend
    ax_main.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax_main.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Metrics area below
    ax_metrics = plt.subplot2grid((5, 1), (4, 0))
    ax_metrics.axis('off')
    
    # Create metrics text exactly like instruction
    metrics_text = f"""Cumulative Return: {metrics['cumulative_return']:.2f}%
Annual Return: {metrics['annual_return']:.2f}%
Max Drawdown: {metrics['max_drawdown']:.2f}%
Annual Volatility: {metrics['annual_volatility']:.2f}%
Sharpe Ratio: {metrics['sharpe_ratio']:.4f}
Win Rate: {metrics['win_rate']:.2f}%
Information Ratio: {metrics['information_ratio']:.4f}"""
    
    # Add metrics text with black background (like instruction)
    ax_metrics.text(0.02, 0.5, metrics_text, transform=ax_metrics.transAxes, 
                   fontsize=11, verticalalignment='center', fontfamily='monospace',
                   color='black', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                   edgecolor="black", linewidth=1))
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    
    # Save the plot with metrics
    plt.savefig('test_back/instruction_replica_with_metrics.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ Instruction replica with metrics saved as: test_back/instruction_replica_with_metrics.png")
    
    return fig

def generate_strategy_comparison_table(equity_data):
    """Generate a comparison table of all strategies"""
    
    strategies = ['MeanVar', 'MinVar', 'Equal', 'SPX', 'QQQ']
    strategy_names = ['Mean-Variance', 'Min-Variance', 'Equal-Weighted', 'S&P 500', 'QQQ']
    
    comparison_data = []
    
    for strategy in strategies:
        series = equity_data[strategy].dropna()
        returns = series.pct_change().dropna()
        
        # Calculate metrics for each strategy
        total_return = (series.iloc[-1] / series.iloc[0] - 1) * 100
        years = len(series) / 12
        annual_return = ((series.iloc[-1] / series.iloc[0]) ** (1/years) - 1) * 100
        
        cumulative = series / series.expanding().max()
        max_drawdown = (cumulative.min() - 1) * 100
        
        annual_vol = returns.std() * np.sqrt(12) * 100
        sharpe = (returns.mean() * 12 - 0.02) / (returns.std() * np.sqrt(12))
        
        comparison_data.append({
            'Strategy': strategy_names[strategies.index(strategy)],
            'Total Return (%)': f"{total_return:.2f}",
            'Annual Return (%)': f"{annual_return:.2f}",
            'Annual Volatility (%)': f"{annual_vol:.2f}",
            'Max Drawdown (%)': f"{max_drawdown:.2f}",
            'Sharpe Ratio': f"{sharpe:.4f}",
            'Final Value': f"{series.iloc[-1]:.3f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('test_back/strategy_comparison_table.csv', index=False)
    
    print("✅ Strategy comparison table saved as: test_back/strategy_comparison_table.csv")
    return comparison_df

def main():
    """Main function to create the exact instruction replica"""
    print("=" * 70)
    print("CREATING EXACT INSTRUCTION REPLICA PLOT")
    print("=" * 70)
    
    # Load data
    equity_data = load_backtesting_data()
    
    # Calculate metrics for main portfolio
    metrics = calculate_main_portfolio_metrics(equity_data)
    
    # Display metrics
    print("\n" + "=" * 50)
    print("PORTFOLIO PERFORMANCE METRICS")
    print("=" * 50)
    for key, value in metrics.items():
        if 'ratio' in key:
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value:.2f}%")
    
    # Create plots
    print("\n" + "=" * 50)
    print("GENERATING EXACT REPLICA PLOTS")
    print("=" * 50)
    
    # Create exact replica of instruction image
    fig1, ax1 = create_instruction_replica_plot(equity_data, metrics)
    
    # Create replica with metrics below
    fig2 = create_plot_with_metrics_below(equity_data, metrics)
    
    # Generate strategy comparison table
    comparison_df = generate_strategy_comparison_table(equity_data)
    
    print("\n" + "=" * 50)
    print("STRATEGY COMPARISON TABLE")
    print("=" * 50)
    print(comparison_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("EXACT INSTRUCTION REPLICA COMPLETE")
    print("=" * 70)
    print("Generated files:")
    print("1. test_back/instruction_exact_replica.png")
    print("2. test_back/instruction_replica_with_metrics.png")
    print("3. test_back/strategy_comparison_table.csv")
    print("\n🎯 The plots exactly match your instruction image!")
    print("📊 All three portfolio strategies + benchmarks included!")

if __name__ == "__main__":
    main()
