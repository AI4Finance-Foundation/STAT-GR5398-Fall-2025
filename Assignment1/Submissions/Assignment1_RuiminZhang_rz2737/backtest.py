import warnings
warnings.filterwarnings("ignore")

import math
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime
import yfinance as yf

daily_filename = "./sp500_tickers_daily_price.csv" ### Replace with your own daily stock information file name
selected_stock_filename = "./stock_selected.csv" ### Replace with your own selected stocks file name
weights_filename = "./EqualWeights.csv" ### Replace with your own stock weights file name
enddate = "2025-09-09" ### Replace with today's date

fee_rate = 1e-3

class Stock:
    def __init__(self, name) -> None:
        self.name: str = name
        self.market_value: float = 0
        self.price: float = 0
        self.quantity: int = 0

    def calc_market_value(self, price: float) -> float:
        if price is not None and math.isnan(price) is False:
            self.price = price
        self.market_value = self.price * self.quantity
        return self.market_value

    def purchase(self, price: float, quantity: int) -> float:
        if quantity == 0:
            return 0
        self.price = price
        self.quantity += quantity
        self.calc_market_value(price)
        return price * quantity
    
    def sell(self, price: float, quantity: int) -> float:
        if price == 0 and quantity == self.quantity:
            self.quantity -= quantity
            return self.price * quantity
        
        if quantity == self.quantity:
            self.price = 0
        else:
            self.price = price

        self.quantity -= quantity
        self.calc_market_value(price)
        return price * quantity
    
    def info(self, price: float = None) -> None:
        print(f"Stock's name: {self.name}")
        print(f"Stock's quantity: {self.quantity}")
        print(f"Stock's market value: {self.calc_market_value(price)}")

class Portfolio:
    def __init__(self, capital) -> None:
        self.capital: float = capital
        self.total_market_value: float = 0
        self.fee: float = 0
        self.stocklist: list[Stock] = []

    def calc_total_market_value(self) -> float:
        total_market_value = 0
        for stock in self.stocklist:
            total_market_value += stock.market_value
        self.total_market_value = total_market_value + self.capital
        return self.total_market_value
    
    def purchase(self, stock: Stock, price: float, quantity: int) -> None:
        if price * quantity * (1 + fee_rate) > self.capital:
            return
        capital_change = stock.purchase(price, quantity)
        self.fee += capital_change * fee_rate
        self.capital -= capital_change * (1 + fee_rate)
        return 
    
    def sell(self, stock: Stock, price: float, quantity: int) -> None:
        capital_change = stock.sell(price, quantity)
        self.fee += capital_change * fee_rate
        self.capital += capital_change * (1 - fee_rate)
        return
    
    def quarterly_rebalancing(self, weightsdf: pd.DataFrame, dailyinfo: pd.DataFrame) -> None:
        ### First, use open price to calculate today's total market value
        for stock in self.stocklist:
            infodf = dailyinfo[dailyinfo.index == stock.name]
            if infodf.empty:
                openprice = stock.price
            else:
                openprice = infodf.iloc[0]["open"]
            stock.market_value = stock.calc_market_value(openprice)

        market_value = self.calc_total_market_value()
        current_names = weightsdf.index

        ### Check if holding stocks should be sold
        for stock in self.stocklist:
            name = stock.name
            if name not in dailyinfo.index:
                openprice = stock.price
            else:
                infodf = dailyinfo[dailyinfo.index == name]
                openprice = infodf.iloc[0]["open"]
            if math.isnan(openprice) or openprice < 1e-3:
                openprice = stock.price
            
            if name not in current_names:
                self.sell(stock, openprice, stock.quantity)
            else:
                ### Check the difference between holding quantity and needed quantity
                quantity = math.floor(weightsdf.loc[name].values[0] * market_value / openprice)
                diff = quantity - stock.quantity
                if diff > 0:
                    self.purchase(stock, openprice, diff)
                else:
                    self.sell(stock, openprice, -diff)

        portfolio_names = [stock.name for stock in self.stocklist]
        ### Check if hold all needed stocks
        for name in current_names:
            if name not in portfolio_names:
                infodf = dailyinfo[dailyinfo.index == name]
                if infodf.empty:
                    continue
                openprice = infodf.iloc[0]["open"]
                if math.isnan(openprice) or openprice < 1e-3:
                    continue
                quantity = math.floor(weightsdf.loc[name].values[0] * market_value / openprice)
                stock = Stock(name)
                self.purchase(stock, openprice, quantity)
                self.stocklist.append(stock)

        ### Update stock list to delete all useless stocks
        self.stocklist = [stock for stock in self.stocklist if stock.quantity > 0]

        ### Update market value using latest stock list
        for stock in self.stocklist:
            infodf = dailyinfo[dailyinfo.index == stock.name]
            if infodf.empty:
                closeprice = stock.price
            else:
                closeprice = infodf.iloc[0]["close"]
            stock.market_value = stock.calc_market_value(closeprice)
        return
    
    def daily_trading(self, date: datetime, weightsdf: pd.DataFrame, dailyinfo: pd.DataFrame) -> None:
        deleteidx = []
        for idx, stock in enumerate(self.stocklist):
            name = stock.name
            if name not in dailyinfo.index:
                openprice = stock.price
                closeprice = stock.price
            else:
                infodf = dailyinfo[dailyinfo.index == name]
                openprice = infodf["open"].iloc[-1]
                closeprice = infodf["close"].iloc[-1]
            
            if name not in weightsdf.index:
                # print(date, stock.name, stock.quantity)
                self.sell(stock, openprice, stock.quantity)
                deleteidx.append(idx)
                continue
            stock.market_value = stock.calc_market_value(closeprice)
            continue

        self.stocklist = [stock for i, stock in enumerate(self.stocklist) if i not in deleteidx]
        return

    def daily_update(self, date: datetime, weightsdf: pd.DataFrame, dailyinfo: pd.DataFrame = None, rebalancing: bool = False) -> None:
        if rebalancing:
            self.quarterly_rebalancing(weightsdf, dailyinfo)
        else:
            self.daily_trading(date, weightsdf, dailyinfo)
        return
    
def evaluate_strategy(portfolio_values, benchmark_values=None, risk_free_rate=0.0):
    portfolio_values = np.array(portfolio_values)
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    cumulative_return = portfolio_values[-1] / portfolio_values[0] - 1

    annual_return = (portfolio_values[-1] / portfolio_values[0]) ** (252 / len(portfolio_values)) - 1

    annual_volatility = returns.std() * np.sqrt(252)

    excess_return = returns.mean() - risk_free_rate / 252
    sharpe_ratio = excess_return / returns.std() * np.sqrt(252)

    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = drawdown.min()

    win_rate = (returns > 0).sum() / (len(returns) - (returns == 0).sum())

    if benchmark_values is not None:
        benchmark_values = np.array(benchmark_values)
        benchmark_returns = pd.Series(benchmark_values).pct_change().dropna()
        excess_returns = returns.values - benchmark_returns.values
        tracking_error = np.std(excess_returns)
        information_ratio = np.mean(excess_returns) / tracking_error * np.sqrt(252) if tracking_error != 0 else np.nan
    else:
        information_ratio = np.nan

    print("Evaluation metrics of your portfolio:")
    print(f"Cumulative Return: {cumulative_return:.2%}")
    print(f"Annual Return: {annual_return:.2%}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Annual Volatility: {annual_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Information Ratio: {information_ratio:.4f}")

    return {
        "Cumulative Return": cumulative_return,
        "Annual Return": annual_return,
        "Max Drawdown": max_drawdown,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Win Rate": win_rate,
        "Information Ratio": information_ratio
    }
    
if __name__ == "__main__":
    ### Load some useful data
    print("Loading portfolio weights...")
    weights = pd.read_csv(weights_filename)
    weights["trade_date"] = pd.to_datetime(weights["trade_date"])

    print("Loading stock daily price...")
    usefulcolumns = ["datadate", "prcod", "prccd", "ajexdi", "tic"]
    dailyinfo = pd.read_csv(daily_filename, usecols=usefulcolumns)
    dailyinfo["datadate"] = pd.to_datetime(dailyinfo["datadate"])
    dailyinfo["ajexdi"] = dailyinfo["ajexdi"].replace(np.nan, 1)

    dailyinfo["open"] = (dailyinfo["prcod"] / dailyinfo["ajexdi"]).astype(float)
    dailyinfo["close"] = (dailyinfo["prccd"] / dailyinfo["ajexdi"]).astype(float)

    dailyinfo.dropna(inplace=True)

    dailyinfo = dailyinfo[(dailyinfo["datadate"] >= weights["trade_date"].min()) & (dailyinfo["datadate"] <= weights["trade_date"].max())]
    dates = sorted(dailyinfo["datadate"].unique())

    print("Downloading SP500 & QQQ daily price...")
    SP500 = yf.download("^GSPC", start="1990-01-01", end=enddate, interval="1d")
    SP500['Close'] = SP500['Close'].ffill()
    mask = (SP500.index >= dates[0]) & (SP500.index <= dates[-1])
    SP500 = SP500.loc[mask, 'Close']["^GSPC"].tolist()

    QQQ = yf.download("QQQ", start="1990-01-01", end=enddate, interval="1d")
    QQQ['Close'] = QQQ['Close'].ffill()
    mask = (QQQ.index >= dates[0]) & (QQQ.index <= dates[-1])
    QQQ = QQQ.loc[mask, 'Close']["QQQ"].tolist()

    ### Strategy Backtest
    print("Backtesting using historical data...")
    stock_selected = pd.read_csv(selected_stock_filename)
    stock_selected["trade_date"] = pd.to_datetime(stock_selected["trade_date"])
    stock_selected["hold_date"] = stock_selected["trade_date"] - pd.DateOffset(months=3)
    tradedates = stock_selected["hold_date"].unique()

    capital = 1e7
    port = Portfolio(capital)
    port_mvs_quart = []

    for date in tqdm(dates, desc="Backtesting"):
        cur_weights = weights[weights["trade_date"] == date][["tic", "weights"]].set_index("tic")
        cur_info = dailyinfo[dailyinfo["datadate"] == date].set_index("tic")

        if date in tradedates:
            port.daily_update(date, cur_weights, cur_info, rebalancing = True)
        else:
            port.daily_update(date, cur_weights, cur_info, rebalancing = False)

        port_mvs_quart.append(port.calc_total_market_value())
    
    ### Results Visualization
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(dates, [data/port_mvs_quart[0] for data in port_mvs_quart], label='Portfolio Value with Quarterly Adjustment', color='blue', linewidth=2)
    ax.plot(dates, [data/SP500[0] for data in SP500], label='SP500', color='green', linewidth=2)
    ax.plot(dates, [data/QQQ[0] for data in QQQ], label='QQQ', color='red', linewidth=2)

    ax.set_title("Portfolio Values", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.legend()

    plt.tight_layout()
    plt.savefig("./Portfolio_Values.png")

    quart_metrics = evaluate_strategy(port_mvs_quart, benchmark_values=SP500)

    with open("./Result_Metrics.json", "w") as f:
        json.dump(quart_metrics, f)
