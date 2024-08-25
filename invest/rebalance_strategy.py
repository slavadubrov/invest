"""Module with RebalanceStrategy"""

from enum import Enum, auto

import pandas as pd


class RebalanceFrequency(Enum):
    QUARTERLY = auto()
    YEARLY = auto()
    NONE = auto()


def determine_rebalance_periods(
    periods: pd.Index, rebalance_frequency: RebalanceFrequency
) -> pd.Index:
    """
    Determine the periods at which the portfolio should be rebalanced.
    """
    rebalance_strategy = RebalanceStrategyFactory.create(rebalance_frequency)
    return rebalance_strategy.get_rebalance_periods(periods)


class RebalanceStrategy:
    """Abstract base class for rebalance strategies."""

    def get_rebalance_periods(self, periods: pd.Index) -> pd.Index:
        raise NotImplementedError


class QuarterlyRebalanceStrategy(RebalanceStrategy):
    """Concrete implementation for quarterly rebalancing."""

    def get_rebalance_periods(self, periods: pd.Index) -> pd.Index:
        return pd.date_range(start=periods.min(), end=periods.max(), freq="Q")


class YearlyRebalanceStrategy(RebalanceStrategy):
    """Concrete implementation for yearly rebalancing."""

    def get_rebalance_periods(self, periods: pd.Index) -> pd.Index:
        return pd.date_range(start=periods.min(), end=periods.max(), freq="YE")


class NoRebalanceStrategy(RebalanceStrategy):
    """Concrete implementation for no rebalancing."""

    def get_rebalance_periods(self, periods: pd.Index) -> pd.Index:
        return periods


class RebalanceStrategyFactory:
    """Factory for creating rebalance strategies."""

    @staticmethod
    def create(rebalance_frequency: RebalanceFrequency) -> RebalanceStrategy:
        if rebalance_frequency == RebalanceFrequency.QUARTERLY:
            return QuarterlyRebalanceStrategy()
        elif rebalance_frequency == RebalanceFrequency.YEARLY:
            return YearlyRebalanceStrategy()
        elif rebalance_frequency == RebalanceFrequency.NONE:
            return NoRebalanceStrategy()
        else:
            raise ValueError(
                "Invalid rebalance_frequency. Use RebalanceFrequency enum."
            )
