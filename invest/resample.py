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
    selected_dates: list[pd.Timestamp] = []

    for day in target_days:
        # Try to find the closest date after or on the target day
        available_dates_after = dates[
            (dates.day >= day) & (dates.month == dates[0].month)
        ]

        if not available_dates_after.empty:
            selected_date = available_dates_after[0]
        else:
            # If no date is available after the target day, take the closest before
            available_dates_before = dates[
                (dates.day < day) & (dates.month == dates[0].month)
            ]
            if not available_dates_before.empty:
                selected_date = available_dates_before[-1]
            else:
                continue  # Skip if no valid date is found

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
        portfolio_returns (DataFrame): A DataFrame of portfolio returns with a datetime index.
        contribution_days (list[int]): A list of days in the month to select for resampling.

    Returns:
        Series: A Series of resampled returns with the selected dates as the index.
    """
    # Get unique months in the data
    unique_months = portfolio_returns.index.to_period("M").unique()

    # Initialize a list to store the results
    resampled_returns: list[tuple] = []

    # Iterate through each month
    for month in unique_months:
        # Get the dates for the current month
        monthly_dates = portfolio_returns[
            portfolio_returns.index.to_period("M") == month
        ].index

        # Get the closest dates to the target contribution days
        selected_dates = _get_closest_available_dates(monthly_dates, contribution_days)

        # Calculate the returns for the selected dates
        for date in selected_dates:
            resampled_value = (1 + portfolio_returns.loc[date]).prod() - 1
            resampled_returns.append((date, resampled_value))

    # Convert the results into a pandas Series
    return pd.Series(dict(resampled_returns))


def resample_returns(portfolio_returns: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    Resample portfolio returns based on the specified frequency.

    Args:
        portfolio_returns (DataFrame): A DataFrame of portfolio returns with a datetime index.
        frequency (str): The frequency to resample the returns. Options are 'weekly', 'monthly', or 'quarterly'.

    Returns:
        DataFrame: A DataFrame of resampled returns.

    Raises:
        ValueError: If an invalid frequency is provided.
    """
    contribution_days = [2, 9, 16, 23]

    if frequency == "weekly":
        return _custom_resample_on_days(portfolio_returns, contribution_days)
    elif frequency == "monthly":
        return portfolio_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    elif frequency == "quarterly":
        return portfolio_returns.resample("Q").apply(lambda x: (1 + x).prod() - 1)
    else:
        raise ValueError("Invalid frequency. Use 'weekly', 'monthly', or 'quarterly'.")
