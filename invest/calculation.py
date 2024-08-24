import pandas as pd

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

    Parameters:
    -----------
    portfolio_returns : pd.DataFrame
        A DataFrame containing the portfolio returns indexed by time.
    initial_weights : pd.Series
        A Series representing the initial weights of the portfolio.
    monthly_investment_amount : float
        The amount invested monthly into the portfolio.
    frequency : str
        The frequency at which the portfolio returns are resampled.
        Can be 'weekly', 'monthly', or 'quarterly'.
    rebalance_frequency : Optional[str], default=None
        The frequency at which the portfolio is rebalanced.
        Can be 'quarterly', 'yearly', or None for no rebalancing.

    Returns:
    --------
    Tuple[pd.Series, pd.Series]
        A tuple containing two Series:
        - The cumulative value of the portfolio over time.
        - The total amount invested over time.
    """
    portfolio_returns_resampled = resample_returns(portfolio_returns, frequency)
    rebalance_periods = determine_rebalance_periods(
        portfolio_returns_resampled.index, rebalance_frequency
    )
    investment_schedule = create_investment_schedule(
        portfolio_returns_resampled.index, monthly_investment_amount, frequency
    )

    return calculate_portfolio_value(
        portfolio_returns_resampled,
        initial_weights,
        investment_schedule,
        rebalance_periods,
    )


def determine_rebalance_periods(
    periods: pd.Index, rebalance_frequency: str | None
) -> pd.Index:
    if rebalance_frequency:
        if rebalance_frequency == "quarterly":
            return pd.date_range(start=periods.min(), end=periods.max(), freq="Q")
        elif rebalance_frequency == "yearly":
            return pd.date_range(start=periods.min(), end=periods.max(), freq="YE")
        else:
            raise ValueError(
                "Invalid rebalance_frequency. Use 'quarterly' or 'yearly'."
            )
    return periods  # No rebalancing


def create_investment_schedule(
    periods: pd.Index, monthly_investment_amount: float, frequency: str
) -> pd.Series:
    investment_schedule = pd.Series(0, index=periods)
    if frequency == "weekly":
        investment_schedule.loc[periods] = monthly_investment_amount / 4
    elif frequency == "monthly":
        investment_schedule.loc[periods] = monthly_investment_amount
    elif frequency == "quarterly":
        investment_schedule.loc[periods] = monthly_investment_amount * 3
    else:
        raise ValueError("Invalid frequency. Use 'weekly', 'monthly', or 'quarterly'.")
    return investment_schedule


def calculate_portfolio_value(
    portfolio_returns_resampled: pd.DataFrame,
    initial_weights: pd.Series,
    investment_schedule: pd.Series,
    rebalance_periods: pd.Index,
) -> tuple[pd.Series, pd.Series]:
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
