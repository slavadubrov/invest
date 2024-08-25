import pandas as pd

from invest.investment_schedule_strategy import create_investment_schedule
from invest.rebalance_strategy import determine_rebalance_periods
from invest.resample import resample_returns


def calculate_cumulative_value_with_contributions_and_rebalancing(
    portfolio_returns: pd.DataFrame,
    initial_weights: pd.Series,
    monthly_investment_amount: float,
    frequency: str,
    rebalance_frequency: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate the cumulative value of a portfolio with regular contributions and periodic rebalancing.
    """
    portfolio_returns_resampled = resample_returns(portfolio_returns, frequency)
    rebalance_periods = determine_rebalance_periods(
        portfolio_returns_resampled.index, rebalance_frequency
    )
    investment_schedule = create_investment_schedule(
        portfolio_returns_resampled.index, monthly_investment_amount, frequency
    )
    cumulative_value, total_invested = calculate_portfolio_value(
        portfolio_returns_resampled,
        initial_weights,
        investment_schedule,
        rebalance_periods,
    )

    return cumulative_value, total_invested


def calculate_portfolio_value(
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
