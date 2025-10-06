import pandas as pd
w = pd.read_csv("result/portfolio_weights.csv", parse_dates=["trade_date"])
# 1) 每天权重和必须=1
chk1 = w.groupby("trade_date")["weights"].sum().round(6).value_counts().head()
print("Sum(weights) by day (head):\n", chk1)

# 2) 不允许负权重
print("Any negative weights? ->", (w["weights"] < 0).any())

# 3) 是否有 NaN
print("Any NaNs? ->", w.isna().any().to_dict())

# 4) 天数覆盖
print("Dates covered:", w["trade_date"].min().date(), "->", w["trade_date"].max().date(), 
      "| #days:", w["trade_date"].nunique())
