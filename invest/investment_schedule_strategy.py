"""Module with InvestmentScheduleStrategy"""

from enum import Enum, auto

import pandas as pd


class InvestmentFrequency(Enum):
    WEEKLY = auto()
    MONTHLY = auto()
    QUARTERLY = auto()


def create_investment_schedule(
    periods: pd.Index, monthly_investment_amount: float, frequency: InvestmentFrequency
) -> pd.Series:
    """
    Create an investment schedule based on the frequency.
    """
    investment_schedule_strategy = InvestmentScheduleFactory.create(frequency)
    return investment_schedule_strategy.create_schedule(
        periods, monthly_investment_amount
    )


class InvestmentScheduleStrategy:
    """Abstract base class for investment schedule strategies."""

    def create_schedule(
        self, periods: pd.Index, monthly_investment_amount: float
    ) -> pd.Series:
        raise NotImplementedError


class WeeklyInvestmentScheduleStrategy(InvestmentScheduleStrategy):
    """Concrete implementation for weekly investment schedules."""

    def create_schedule(
        self, periods: pd.Index, monthly_investment_amount: float
    ) -> pd.Series:
        investment_schedule = pd.Series(0, index=periods)
        investment_schedule.loc[periods] = monthly_investment_amount / 4
        return investment_schedule


class MonthlyInvestmentScheduleStrategy(InvestmentScheduleStrategy):
    """Concrete implementation for monthly investment schedules."""

    def create_schedule(
        self, periods: pd.Index, monthly_investment_amount: float
    ) -> pd.Series:
        investment_schedule = pd.Series(0, index=periods)
        investment_schedule.loc[periods] = monthly_investment_amount
        return investment_schedule


class QuarterlyInvestmentScheduleStrategy(InvestmentScheduleStrategy):
    """Concrete implementation for quarterly investment schedules."""

    def create_schedule(
        self, periods: pd.Index, monthly_investment_amount: float
    ) -> pd.Series:
        investment_schedule = pd.Series(0, index=periods)
        investment_schedule.loc[periods] = monthly_investment_amount * 3
        return investment_schedule


class InvestmentScheduleFactory:
    """Factory for creating investment schedule strategies."""

    @staticmethod
    def create(frequency: InvestmentFrequency) -> InvestmentScheduleStrategy:
        if frequency == InvestmentFrequency.WEEKLY:
            return WeeklyInvestmentScheduleStrategy()
        elif frequency == InvestmentFrequency.MONTHLY:
            return MonthlyInvestmentScheduleStrategy()
        elif frequency == InvestmentFrequency.QUARTERLY:
            return QuarterlyInvestmentScheduleStrategy()
        else:
            raise ValueError("Invalid frequency. Use InvestmentFrequency enum.")
