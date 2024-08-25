import yfinance as yf


class DataFetcher:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        return yf.download(self.tickers, start=self.start_date, end=self.end_date)[
            "Adj Close"
        ]
