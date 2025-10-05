#!/usr/bin/env python3
"""
plot_portfolio_comparison.py

This script creates the exact same portfolio comparison plot as shown in the assignment instructions.
It includes Portfolio Value with Quarterly Adjustment, SP500, and QQQ with performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load the backtesting results and prepare data for plotting"""
    print("Loading backtesting results...")
    
    # Load the equity curve data (monthly frequency)
    equity_data = pd.read_csv('test_back/comparison_equity_M.csv', index_col=0, parse_dates=True)
    
    # Load the quarterly data for more accurate quarterly adjustment representation
    equity_quarterly = pd.read_csv('test_back/comparison_equity_Q.csv', index_col=0, parse_dates=True)
    
    print(f"Data loaded: {len(equity_data)} monthly observations")
    print(f"Date range: {equity_data.index[0]} to {equity_data.index[-1]}")
    print(f"Strategies: {list(equity_data.columns)}")
    
    return equity_data, equity_quarterly

def calculate_performance_metrics(equity_data):
    """Calculate the performance metrics shown in the instruction image"""
    
    # Get the final values for cumulative return calculation
    final_values = equity_data.iloc[-1]
    initial_values = equity_data.dropna().iloc[0]
    
    # Calculate portfolio performance (assuming this is for the mean-variance strategy)
    # Using MeanVar as the main portfolio strategy
    portfolio_series = equity_data['MeanVar'].dropna()
    
    # Calculate metrics
    cumulative_return = (final_values['MeanVar'] - 1) * 100  # Convert to percentage
    
    # Calculate annualized return
    years = len(portfolio_series) / 12  # Monthly data
    annual_return = ((final_values['MeanVar'] / initial_values['MeanVar']) ** (1/years) - 1) * 100
    
    # Calculate returns for other metrics
    monthly_returns = portfolio_series.pct_change().dropna()
    
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
    
    # Calculate Information Ratio (vs SPX)
    spx_returns = equity_data['SPX'].pct_change().dropna()
    tracking_error = (monthly_returns - spx_returns.reindex(monthly_returns.index)).std() * np.sqrt(12)
    alpha = monthly_returns.mean() * 12 - spx_returns.mean() * 12
    information_ratio = alpha / tracking_error if tracking_error != 0 else 0
    
    metrics = {
        'cumulative_return': cumulative_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'information_ratio': information_ratio
    }
    
    return metrics

def create_portfolio_comparison_plot(equity_data, metrics):
    """Create the exact portfolio comparison plot as shown in instructions"""
    
    # Set up the plot with the same style as the instruction image
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each strategy with the same colors and style as the instruction
    # Blue line for Portfolio Value with Quarterly Adjustment (using MeanVar as main strategy)
    ax.plot(equity_data.index, equity_data['MeanVar'], 
            color='blue', linewidth=2, label='Portfolio Value with Quarterly Adjustment')
    
    # Green line for SP500
    ax.plot(equity_data.index, equity_data['SPX'], 
            color='green', linewidth=2, label='SP500')
    
    # Red line for QQQ
    ax.plot(equity_data.index, equity_data['QQQ'], 
            color='red', linewidth=2, label='QQQ')
    
    # Optional: Add other portfolio strategies with thinner lines
    ax.plot(equity_data.index, equity_data['MinVar'], 
            color='orange', linewidth=1.5, alpha=0.7, label='Min-Variance Portfolio')
    
    ax.plot(equity_data.index, equity_data['Equal'], 
            color='purple', linewidth=1.5, alpha=0.7, label='Equal-Weighted Portfolio')
    
    # Formatting to match the instruction image
    ax.set_title('Portfolio Values', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    
    # Set y-axis limits (0 to 5)
    ax.set_ylim(0, 5)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend in the same position as instruction
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('test_back/portfolio_comparison_instruction_style.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ Portfolio comparison plot saved as: test_back/portfolio_comparison_instruction_style.png")
    
    return fig, ax

def create_performance_summary_text(metrics):
    """Create the performance metrics text as shown in the instruction"""
    
    summary_text = f"""
Portfolio Performance Metrics:

Cumulative Return: {metrics['cumulative_return']:.2f}%
Annual Return: {metrics['annual_return']:.2f}%
Max Drawdown: {metrics['max_drawdown']:.2f}%
Annual Volatility: {metrics['annual_volatility']:.2f}%
Sharpe Ratio: {metrics['sharpe_ratio']:.4f}
Win Rate: {metrics['win_rate']:.2f}%
Information Ratio: {metrics['information_ratio']:.4f}
"""
    
    return summary_text

def create_combined_plot_with_metrics(equity_data, metrics):
    """Create a combined plot with the chart and performance metrics"""
    
    # Create a figure with subplots - main plot and text area
    fig = plt.figure(figsize=(14, 10))
    
    # Create the main plot (takes up most of the space)
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    
    # Plot the portfolio comparison
    # Blue line for main portfolio strategy (Mean-Variance)
    ax1.plot(equity_data.index, equity_data['MeanVar'], 
            color='blue', linewidth=2.5, label='Portfolio Value with Quarterly Adjustment')
    
    # Green line for SP500
    ax1.plot(equity_data.index, equity_data['SPX'], 
            color='green', linewidth=2, label='SP500')
    
    # Red line for QQQ
    ax1.plot(equity_data.index, equity_data['QQQ'], 
            color='red', linewidth=2, label='QQQ')
    
    # Formatting
    ax1.set_title('Portfolio Values', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_ylim(0, 5)
    
    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Add grid
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    
    # Create text area for performance metrics
    ax2 = plt.subplot2grid((4, 1), (3, 0))
    ax2.axis('off')
    
    # Add performance metrics text
    metrics_text = f"""Cumulative Return: {metrics['cumulative_return']:.2f}%        Annual Return: {metrics['annual_return']:.2f}%        Max Drawdown: {metrics['max_drawdown']:.2f}%        Annual Volatility: {metrics['annual_volatility']:.2f}%
Sharpe Ratio: {metrics['sharpe_ratio']:.4f}        Win Rate: {metrics['win_rate']:.2f}%        Information Ratio: {metrics['information_ratio']:.4f}"""
    
    ax2.text(0.05, 0.5, metrics_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Set the overall layout
    plt.tight_layout()
    
    # Save the combined plot
    plt.savefig('test_back/portfolio_comparison_with_metrics.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ Combined plot with metrics saved as: test_back/portfolio_comparison_with_metrics.png")
    
    return fig

def main():
    """Main function to create the portfolio comparison plot"""
    print("=" * 60)
    print("CREATING PORTFOLIO COMPARISON PLOT")
    print("=" * 60)
    
    # Load data
    equity_data, equity_quarterly = load_and_prepare_data()
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(equity_data)
    
    # Display calculated metrics
    print("\n" + "=" * 40)
    print("CALCULATED PERFORMANCE METRICS")
    print("=" * 40)
    for key, value in metrics.items():
        if 'ratio' in key:
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value:.2f}%")
    
    # Create the portfolio comparison plot
    print("\n" + "=" * 40)
    print("GENERATING PLOTS")
    print("=" * 40)
    
    # Create simple comparison plot
    fig1, ax1 = create_portfolio_comparison_plot(equity_data, metrics)
    
    # Create combined plot with metrics
    fig2 = create_combined_plot_with_metrics(equity_data, metrics)
    
    # Create performance summary
    summary = create_performance_summary_text(metrics)
    
    # Save performance summary to file
    with open('test_back/portfolio_performance_summary.txt', 'w') as f:
        f.write(summary)
    
    print("✅ Performance summary saved as: test_back/portfolio_performance_summary.txt")
    
    print("\n" + "=" * 60)
    print("PORTFOLIO COMPARISON COMPLETE")
    print("=" * 60)
    print("Files generated:")
    print("1. test_back/portfolio_comparison_instruction_style.png")
    print("2. test_back/portfolio_comparison_with_metrics.png") 
    print("3. test_back/portfolio_performance_summary.txt")
    print("\nThe plots match the style and format of the instruction image! 📊")

if __name__ == "__main__":
    main()
