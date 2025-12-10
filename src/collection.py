# collection.py
import os
import time
import pandas as pd
import yfinance as yf
import requests
# data folder in root dir
dataF = os.path.join(os.path.dirname(__file__), "..", "data")
# date range
startDate = "2015-01-01"
endDate = "2024-12-31"
# sp500 index symbol
sp500Ticker = "^GSPC"
# get tickers
def getTickers():
    # pull sp 
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # fake a browser so the site is okay with our request (ran into many issues w this. best solution via geeks4geeks)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    # read
    tables = pd.read_html(response.text)
    table = tables[0]
    # get tickers
    tickers = table["Symbol"].tolist()
    # format/clean ticker
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers

# get prices for tickers
def getPrices(tickers, startDate, endDate):
    data = yf.download(
        tickers,
        start=startDate,
        end=endDate,
        auto_adjust=True,
        progress=True,
    )
    # make sure close price
    return data["Close"]

# get np 500 price, this is benchmark data i believe
def getSPprice(symbol, startDate, endDate):
    # download daily close prices for the index
    data = yf.download(
        symbol,
        start=startDate,
        end=endDate,
        auto_adjust=True,
        progress=True,
    )
    return data["Close"]

# get metrics
def getMetrics(tickers):
    rows = []
    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            b = stock.balance_sheet
            # sector
            sector = stock.info.get("sector")
            # making sure we always have a pe
            # metric pe
            pe = stock.info.get("trailingPE")
            if pe is None: pe = stock.info.get("forwardPE")
            # metric: pb
            pb = stock.info.get("priceToBook")
            # metric: share price
            sharePrice = stock.info.get("regularMarketPrice")
            # metric:  debt
            debt = None
            if "Total Debt" in b.index: debt = b.loc["Total Debt"].iloc[0]
            # metric: total liability
            totalLiability = None
            if "Total Liab" in b.index: totalLiability = b.loc["Total Liab"].iloc[0]
            # metric: current liability
            currentLiability = None
            if "Total Current Liabilities" in b.index: currentLiability = b.loc["Total Current Liabilities"].iloc[0]
            # metric: intangible assets
            intangibleAssets = None
            if "Intangible Assets" in b.index: intangibleAssets = b.loc["Intangible Assets"].iloc[0]
            # metric: total assets
            totalAssets = None
            if "Total Assets" in b.index: totalAssets = b.loc["Total Assets"].iloc[0]
            # metric: current assets
            currentAssets = None
            if "Total Current Assets" in b.index: currentAssets = b.loc["Total Current Assets"].iloc[0]
            # making sure we always have a bval for equity
            # metric: equity
            equity = None
            if "Common Stock Equity" in b.index: equity = b.loc["Common Stock Equity"].iloc[0]
            elif "Stockholders Equity" in b.index: equity = b.loc["Stockholders Equity"].iloc[0]
            # metric: cash
            cash = None
            if "Cash And Cash Equivalents" in b.index: cash = b.loc["Cash And Cash Equivalents"].iloc[0]
            # making sure we always have a company name
            # metric: company name
            company = stock.info.get("shortName")
            if company is None: company = stock.info.get("longName") 
            rows.append(
                {
                    "ticker": ticker,
                    "company": company,
                    "sector": sector,
                    "pe": pe,
                    "pb": pb,
                    "sharePrice": sharePrice,
                    "debt": debt,
                    "totalLiability": totalLiability,
                    "currentLiability": currentLiability,
                    "intangibleAssets": intangibleAssets,
                    "totalAssets": totalAssets,
                    "currentAssets": currentAssets,
                    "equity": equity,
                    "cash": cash,
                }
            )
        except Exception as e:
            # print [specific ticker], prob: [the problem]
            print(f" {ticker} , prob: {e}")
        # so that i never get a super irritating 429 error msg again...
        time.sleep(0.1)
    metrics = pd.DataFrame(rows)
    return metrics
# main
def main():
    # make sure data folder ac
    os.makedirs(dataF, exist_ok=True)
    # get s&p 500 tickers
    tickers = getTickers()
    # make sure they are randos, if we dont want them to be random, we can keep printing the same 50 by deleting line 100 & 101
    import random
    random.shuffle(tickers)
    # 50 for test purposes, CAN CHANGE NUMBER LATER IF NEEDED
    tickers = tickers[:50]
    # tickers
    pathT = os.path.join(dataF, "sp500Tickers.csv")
    pd.Series(tickers, name="ticker").to_csv(pathT, index=False)
    print(f"tickers: {pathT}")
    # get ticker prices
    prices = getPrices(tickers, startDate, endDate)
    pricesPath = os.path.join(dataF, "sp500Prices.csv")
    prices.to_csv(pricesPath)
    print(f"ticker prices : {pricesPath}")
    # get snp 500 prices
    SPprice = getSPprice(sp500Ticker, startDate, endDate)
    SPpath = os.path.join(dataF, "sp500Index.csv")
    SPprice.to_csv(SPpath)
    print(f"S&P 500 Prices: {SPpath}")
    # get fundamentals and save
    metrics = getMetrics(tickers)
    pathM = os.path.join(dataF, "metrics.csv")
    metrics.to_csv(pathM, index=False)
    print(f"metrics: {pathM}")

if __name__ == "__main__":
    main()