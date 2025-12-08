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
            info = stock.info
            sector = info.get("sector")
            # making sure we always have a pe
            pe = info.get("trailingPE")
            if pe is None:
                pe = info.get("forwardPE")
            pb = info.get("priceToBook")
            de = info.get("debtToEquity")
            fcf = info.get("freeCashflow")
            rows.append(
                {
                    "ticker": ticker,
                    "sector": sector,
                    "pe": pe,
                    "pb": pb,
                    "de": de,
                    "fcf": fcf,
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