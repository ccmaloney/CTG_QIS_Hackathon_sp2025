from quant_strategy_framework import BaseStrategy
from typing import List, Optional


class MomentumStrategy(BaseStrategy):
    """
    A momentum strategy that goes long the top 20% of assets and short the bottom 20%
    based on recent price momentum, selecting only from the equity_tickers list.

    Parameters:
        equity_tickers: List of equity tickers to consider for the strategy
        momentum_period: The momentum period to use (e.g., 'mom_close_20', 'mom_close_60', 'mom_close_240')
        top_pct: Percentage of top performers to go long (default: 0.2 or 20%)
        bottom_pct: Percentage of bottom performers to go short (default: 0.2 or 20%)
    """

    def __init__(
        self,
        equity_tickers: Optional[List[str]] = None,
        momentum_period: str = "mom_close_240",
        top_pct: float = 0.2,
        bottom_pct: float = 0.2,
    ):
        """
        Initialize the momentum strategy.

        Args:
            equity_tickers: List of equity tickers to consider. If None, uses all tickers.
            momentum_period: Column name for the momentum indicator to use.
            top_pct: Percentage of top performers to go long.
            bottom_pct: Percentage of bottom performers to go short.
        """
        self.equity_tickers = equity_tickers
        self.momentum_period = momentum_period
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct

    def construct_insights(self, model_state, current_date):
        """
        Implement the momentum strategy logic.

        Args:
            model_state: DataFrame with data up to current_date
            current_date: The current trading day

        Returns:
            Dict containing portfolio weights and timestamp
        """
        # Get the latest data for each asset
        latest_data = model_state.groupby("ticker").last().reset_index()

        # Filter out assets with missing momentum data
        latest_data = latest_data.dropna(subset=[self.momentum_period])

        # Filter for only assets in the equity_tickers list if specified
        if self.equity_tickers is not None:
            latest_data = latest_data[latest_data["ticker"].isin(self.equity_tickers)]

        # If no valid data, return empty weights
        if len(latest_data) == 0:
            return {"timestamp": current_date, "weights": {}}

        # Initialize all weights to 0
        weights = {}
        for ticker in model_state["ticker"].unique():
            weights[ticker] = 0.0

        # Rank assets by momentum (higher is better)
        ranked_assets = latest_data.sort_values(self.momentum_period, ascending=False)

        total_assets = len(ranked_assets)

        # Calculate the number of assets in the top and bottom percentages
        top_count = max(1, int(total_assets * self.top_pct))
        bottom_count = max(1, int(total_assets * self.bottom_pct))

        if total_assets > 0:
            # Allocate equal positive weights to top assets (long positions)
            long_weight = 1.0 / (top_count + bottom_count)
            for i in range(top_count):
                if i < len(ranked_assets):
                    asset = ranked_assets.iloc[i]["ticker"]
                    weights[asset] = long_weight

            # Allocate equal negative weights to bottom assets (short positions)
            short_weight = -1.0 / (top_count + bottom_count)
            for i in range(1, bottom_count + 1):
                if i <= len(ranked_assets):
                    asset = ranked_assets.iloc[-i]["ticker"]
                    weights[asset] = short_weight

        return {"timestamp": current_date, "weights": weights}

    def construct_insights_multi(self, data, dates):
        """
        Optimized implementation for processing multiple dates in parallel.

        Args:
            data: Complete DataFrame containing all data.
            dates: List of dates to process.

        Returns:
            List of insight dictionaries, one for each date.
        """
        insights = []
        date_column = data.columns[0]  # Assuming first column is date column

        for date in dates:
            # Get data up to this date
            date_data = data[data[date_column] <= date]

            # Process this date
            insight = self.construct_insights(date_data, date)
            insights.append(insight)

        return insights
