from matplotlib import pyplot as plt
from pypfopt import EfficientFrontier, expected_returns, plotting, risk_models


class PortfolioOptimizer:
    def __init__(self, data):
        self.mu = expected_returns.mean_historical_return(data)
        self.S = risk_models.sample_cov(data)

    def optimize_portfolio(
        self, weight_bounds: dict[str, tuple[float, float]] | None, plot=True
    ) -> dict[str, float]:
        if plot:
            self.plot_optimizer(weight_bounds=weight_bounds)

        weight_bounds = (
            [weight_bounds[ticker] for ticker in self.mu.index]
            if weight_bounds
            else (0, 1)
        )

        ef = EfficientFrontier(self.mu, self.S, weight_bounds=weight_bounds)
        _ = ef.max_sharpe()
        return ef.clean_weights()

    def plot_optimizer(self, weight_bounds: dict[str, tuple[float, float]] | None):

        weight_bounds = (
            [weight_bounds[ticker] for ticker in self.mu.index]
            if weight_bounds
            else (0, 1)
        )

        # Plotting the efficient frontier
        _, ax = plt.subplots()
        ef = EfficientFrontier(self.mu, self.S, weight_bounds=weight_bounds)
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True, show_tickers=True)

        # Plotting the maximum Sharpe ratio portfolio
        ret_tangent, std_tangent, _ = ef.portfolio_performance()
        ax.scatter(
            std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe"
        )
        plt.legend()
        plt.show()
