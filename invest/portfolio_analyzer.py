from dataclasses import dataclass

import pandas as pd
from matplotlib import pyplot as plt

from invest.investment_schedule_strategy import (
    InvestmentFrequency,
    create_investment_schedule,
)
from invest.rebalance_strategy import RebalanceFrequency, determine_rebalance_periods
from invest.resample import resample_returns


@dataclass
class InvestStrategy:
    investment_frequency: InvestmentFrequency
    rebalance_frequency: RebalanceFrequency
    investment_period_amount: float


class PortfolioAnalyzer:
    def __init__(
        self,
        data: pd.DataFrame,
        weights: dict[str, float],
        invest_strategy: InvestStrategy,
    ):
        self.daily_returns = data.pct_change().dropna()
        self.weights = pd.Series(weights).reindex(self.daily_returns.columns).fillna(0)
        self.invest_strategy = invest_strategy

    def calculate_portfolio_returns(self):
        return self.daily_returns.dot(self.weights)

    def calculate_portfolio_value(
        self,
        portfolio_returns_resampled: pd.DataFrame,
        initial_weights: pd.Series,
        investment_schedule: pd.Series,
        rebalance_periods: pd.Index,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculate the portfolio value over time, considering returns, contributions, and rebalancing.
        """
        cumulative_value = []
        total_invested = []
        total_amount_invested = 0.0
        portfolio_value = (investment_schedule.iloc[0] * initial_weights).copy()

        for period in portfolio_returns_resampled.index:
            total_amount_invested += investment_schedule.loc[period]
            total_invested.append(total_amount_invested)

            # Apply returns to the portfolio value
            portfolio_value *= 1 + portfolio_returns_resampled.loc[period]

            # Add new investment
            portfolio_value += investment_schedule.loc[period] * initial_weights

            cumulative_portfolio_value = portfolio_value.sum()
            cumulative_value.append(cumulative_portfolio_value)

            # Rebalance portfolio if it's a rebalance period
            if period in rebalance_periods:
                portfolio_value = (
                    cumulative_portfolio_value * initial_weights
                )  # Rebalance to target weights

        return pd.Series(
            cumulative_value, index=portfolio_returns_resampled.index
        ), pd.Series(total_invested, index=portfolio_returns_resampled.index)

    def invest_periodically(self):
        self.portfolio_returns = self.calculate_portfolio_returns()
        portfolio_returns_resampled = resample_returns(
            self.portfolio_returns, self.invest_strategy.investment_frequency
        )
        rebalance_periods = determine_rebalance_periods(
            portfolio_returns_resampled.index, self.invest_strategy.rebalance_frequency
        )
        investment_schedule = create_investment_schedule(
            portfolio_returns_resampled.index,
            self.invest_strategy.investment_period_amount,
            self.invest_strategy.investment_frequency,
        )
        cumulative_value, total_invested = self.calculate_portfolio_value(
            portfolio_returns_resampled,
            self.weights,
            investment_schedule,
            rebalance_periods,
        )

        return cumulative_value, total_invested

    def plot_cumulative_value(self, cumulative_value_series, total_invested_series):
        plt.figure(figsize=(12, 6))
        cumulative_value_series.plot(label="Portfolio Value", color="blue")
        total_invested_series.plot(label="Total Amount Invested", color="orange")
        plt.title("Cumulative Portfolio Value and Total Investment Over Time")
        plt.ylabel("Value ($)")
        plt.xlabel("Date")
        plt.legend()
        plt.show()

    def plot_percentage_return(self, cumulative_value_series, total_invested_series):
        percentage_returns_series = (
            (cumulative_value_series - total_invested_series) / total_invested_series
        ) * 100
        plt.figure(figsize=(12, 6))
        percentage_returns_series.plot(label="Percentage Return", color="green")
        plt.title("Percentage Return on Portfolio Over Time")
        plt.ylabel("Return (%)")
        plt.xlabel("Date")
        plt.legend()
        plt.show()
