# GR5398 FinGPT Assignment1 Instruction

## 0. Targets

In assignment 1, we want you to:

+ Run **[FinRL-Trading](https://github.com/AI4Finance-Foundation/FinRL-Trading)** to get a basic understanding of what you will do in this semester.
+ Design a portfolio using the selected stocks, and learn some fundamental information of quantitative trading (especially stock selection part).
+ Summarize your result in a very brief research report. Submit your codes onto GitHub repo in a new folder called `Assignment1_Name_UNI`.

Assignment 1 Report Submission Due Day: Oct 5th, 2025.

## 1. Run FinRL-Trading

### 1.1 Using Machine Learning to select stocks

In this part, you should fully understand what we have done, and reproduce this research, especially the stock selection section for beginners.

#### 1.1.1 Data Processing

To achieve this goal, you should focus on folder *[data_processor](https://github.com/AI4Finance-Foundation/FinRL-Trading/tree/master/data_processor)* first to get the useful fundamental data for stocks. 

+ Note: Here we highly recommend you to register for a WRDS account which our university has provided for all master students for free. Please follow the instruction below to get a WRDS account:
  + https://guides.library.columbia.edu/wrds

**Step 1: Get some data**

Firstly, you should head to [S&P 500 Historical Components & Changes(07-12-2025).csv](https://github.com/fja05680/sp500) to get latest components stocks of S&P 500 index. You can use this file as input of `Step1_get_sp500_ticker.py` by:

```bash
python Step1_get_sp500_ticker.py --Stock_Index_His_file "S&P 500 Historical Components & Changes(xx-xx-xxxx).csv" --output_filename "sp500_tickers"
```

which will output a file named after your parameter.

Secondly, you should go to [WRDS-Fundamentals Quarterly](https://wrds-www.wharton.upenn.edu/pages/get-data/compustat-capital-iq-standard-poors/compustat/north-america-daily/fundamentals-quarterly/) to download fundamental data for S&P 500 component stocks from 1996-01-01 to the most current date. You will get a raw csv file contains all fundamental data for the tickers in sp500_tickers, about 200MB.

Finally, you should go to [WRDS-Security Daily](https://wrds-www.wharton.upenn.edu/pages/get-data/compustat-capital-iq-standard-poors/compustat/north-america-daily/security-daily/) to download daily data for S&P 500 component stocks from 1996-01-01 to the most current date. You will get a raw csv file contains all daily price for the tickers in sp500_tickers, about 1GB. 

+ Note: If you don't have enough storage or want this data file to be loaded faster while doing calculation, you can select these columns below only while querying the WRDS database.
  + **prccd (Price - Close - Daily)**, **prcod (Price - Open - Daily)**, **ajexdi (Adjustment Factor (Issue)-Cumulative by Ex-Date)**, **tic (Ticker)**

**Step 2: Preprocess fundamental data**

You should use two data files from step 1 as input to get final_ratios.csv and ratios by sector in folder `outputs` using command below:

```bash
python Step2_preprocess_fundmental_data.py --Stock_Index_fundation_file "sp500_tickers_fundamental_quarterly.csv" --Stock_Index_price_file "sp500_tickers_daily_price.csv"
```

In this code file, we implement:

+ Use Trade date instead of quarterly report date
+ Get next quarter's return
+ Calculate Financial Ratios: PE (Price–to-Earnings Ratio), PS (Price-to-Sales Ratio), PB (Price-to-Book Ratio), OPM (Operating Margin), NPM (Net Profit Margin), ROA (Return On Assets), ROE (Return on Equity), EPS (Earnings Per Share), BPS (Book Per Share), DPS (Dividend Per Share) ...
+ Split the financial ratios by the Global Industry Classification Standard (GICS) sectors (total 11 sectors): 10-Energy, 15-Materials, 20-Industrials, 25-Consumer Discretionary, 30-Consumer Staples, 35-Health Care, 40-Financials, 45-Information Technology, 50-Communication Services, 55-Utilities, 60-Real Estate

#### 1.1.2 Stock Selection

After this is done, you should get a folder with a `final_ratios.csv` and fundamental ratios by stocks' industry sector. Then, you should use Machine Learning to select stocks using those fundamental data to determine what stocks should we hold for every quarter. You can directly run `stock_selection.py` to use ML to select stocks by:

```bash
python stock_selection.py --data_path "your folder that include final_ratios.csv" --output_path "./result"
```

+ Reminder: After our experiments, we found that you should manually change the label for stocks from "gvkey" to "tic" in  `stock_selection.py` since there were some overlaps among different stocks that used the same gvkey during the same period.

In fact, the core part of stock selection using machine learning is the function `run_4model` defined in file `ml_model.py`. Here we use Random Forest, Gradient Boosting and XGBoost, while using the best model with smallest MSE among these 3 models, to do the stock selection. If you have any better ideas, you can modify this part or add some new algorithms.

### Optional: Using Deep Reinforcement Learning to select stocks

Apart from Machine Learning methods, you can also try Deep Reinforcement Learning to select stocks by running `fundamental_portfolio_drl.py`. We highly recommend you to use GPU(s) to implement this.

### 1.2 Outputs of stock selection

You should have a file called `stock_selected.csv` which contains the stocks you selected for every quarter. The content should be like this:

| tic  | predicted_return   | trade_date |
| ---- | ------------------ | ---------- |
| COP  | 0.0146875288337469 | 2001-03-01 |
| APC  | 0.0251015825683922 | 2001-06-01 |
| ...  | ...                | ...        |
| WY   | 0.0006114667630754 | 2025-09-01 |

## 2. Build a Portfolio

### 2.1 Build a Mean-Variance Portfolio

After you get the selected stocks for every quarter, you should design a way to allocate your capital and build a portfolio. The easiest way to build a portfolio is to give each component stock an equal weight. However, we have already tried for this, and it didn't work very well. Also, we tried the Minimum Variance Portfolio, which got the least volatility during a certain period. But this strategy performed badly as well (we also highly recommend you to try these by yourself and you can find some ways to optimize these two strategies).

Thus, your main task is to build a **Mean-Variance Portfolio** like the portfolio below **(you are NOT allowed to use short sales)**:

![image-20250918215104177](./assets/image-20250918215104177.png)

This portfolio has the largest sharpe ratio among all the possible combinations. You can have the best base model return using this portfolio.

+ Reminder: you can use `minimize` in `scipy` to calculate this MVP.

  ```python
  from scipy.optimize import minimize
  ```


### 2.2 Output of your portfolio

You should generate a csv file with information below:

| trade_date | tic  | weights              |
| ---------- | ---- | -------------------- |
| 2001-03-01 | APA  | 0.004807692307692308 |
| 2001-03-02 | APA  | 0.004807692307692308 |
| ...        | ...  | ...                  |
| 2001-05-31 | APA  | 0.004807692307692308 |
| ...        | ...  | ...                  |

**You should submit your csv file and your code (best in Python) that helps you design this portfolio.**

## 3. Portfolio Trading Backtest

We will provide you with backtest code so you can check your portfolio's performance on historical data. This code will generate a portfolio market value chart and the strategy's performance metrics. In the code, the whole portfolio will only trade on "trade_date" according to your portfolio. And during the holding period, there will be no trades. You can also try other strategies and trading signals if you are interested. Here is an example of the results:

![image-20250910123306370](./assets/image-20250910123306370.png)

```
Cumulative Return: 1326.15%
Annual Return: 13.55%
Max Drawdown: -28.78%
Annual Volatility: 16.92%
Sharpe Ratio: 0.8361
Win Rate: 54.58%
Information Ratio: 0.3078
```

**You should add your portfolio value chart and these performance measurement metrics into your report.**

## 4. Research Report for Assignment 1

Since you have to submit a research report as final evaluation of your performance in this course, here we highly recommend you to submit all your codes and files onto GitHub while putting all your charts and measurement results into your report.

Your report for assignment 1 should include:

+ Core code of your MVP calculation
+ Portfolio trading backtest results
+ **Optional**: DRL stock selection results, your own stock selecting/trading strategies and their results
