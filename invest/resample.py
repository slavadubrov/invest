"""Module to resample portfolio returns based on different frequencies."""

import pandas as pd


def _get_closest_available_dates(
    dates: pd.DatetimeIndex, target_days: list[int]
) -> list[pd.Timestamp]:
    """
    Get the closest available dates in the same month for the specified target days.

    Args:
        dates (pd.DatetimeIndex): A datetime index of available dates in a specific month.
        target_days (list[int]): A list of target days to select the closest available dates for.

    Returns:
        list[pd.Timestamp]: A list of selected dates closest to the target days.
    """
    selected_dates = []

    for day in target_days:
        # Find the closest date after or on the target day
        available_dates_after = dates[dates.day >= day]
        if not available_dates_after.empty:
            selected_date = available_dates_after[0]
        else:
            # If no date is available after the target day, take the closest before
            available_dates_before = dates[dates.day < day]
            if available_dates_before.empty:
                continue  # Skip if no valid date is found
            selected_date = available_dates_before[-1]

        selected_dates.append(selected_date)
        # Remove this date and any prior dates from further consideration
        dates = dates[dates > selected_date]

    return selected_dates


def _custom_resample_on_days(
    portfolio_returns: pd.DataFrame, contribution_days: list[int]
) -> pd.Series:
    """
    Resample the portfolio returns on the closest available dates to specified contribution days.

    Args:
        portfolio_returns (pd.DataFrame): A DataFrame of portfolio returns with a datetime index.
        contribution_days (list[int]): A list of days in the month to select for resampling.

    Returns:
        pd.Series: A Series of resampled returns with the selected dates as the index.
    """
    unique_months = portfolio_returns.index.to_period("ME").unique()
    resampled_returns = []

    for month in unique_months:
        monthly_dates = portfolio_returns[
            portfolio_returns.index.to_period("ME") == month
        ].index
        selected_dates = _get_closest_available_dates(monthly_dates, contribution_days)

        for date in selected_dates:
            resampled_value = (1 + portfolio_returns.loc[date]).prod() - 1
            resampled_returns.append((date, resampled_value))

    return pd.Series(dict(resampled_returns))


def resample_returns(portfolio_returns: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Resample portfolio returns based on the specified frequency.

    Args:
        portfolio_returns (pd.DataFrame): A DataFrame of portfolio returns with a datetime index.
        frequency (str): The frequency to resample the returns. Options are 'weekly', 'monthly', or 'quarterly'.

    Returns:
        pd.DataFrame: A DataFrame of resampled returns.

    Raises:
        ValueError: If an invalid frequency is provided.
    """
    contribution_days = [2, 9, 16, 23]

    if frequency == "weekly":
        return _custom_resample_on_days(portfolio_returns, contribution_days)
    elif frequency == "monthly":
        return portfolio_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    elif frequency == "quarterly":
        return portfolio_returns.resample("Q").apply(lambda x: (1 + x).prod() - 1)
    else:
        raise ValueError("Invalid frequency. Use 'weekly', 'monthly', or 'quarterly'.")
