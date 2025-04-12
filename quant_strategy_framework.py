"""
Quant Investment Strategy Framework

This module provides a simple framework for developing quantitative investment strategies.
It is designed for use in a hackathon environment where participants, including those with
limited technical experience, can focus on strategy development rather than data handling.

The framework handles data slicing, iteration through trading days, and provides a clean
interface for strategy implementation.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Union, List, Optional, Tuple
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """
    Abstract base class for all investment strategies.

    This class defines the interface that all concrete strategy implementations
    must adhere to. By using this abstract base class, we ensure that all
    strategies implement the required methods.
    """

    @abstractmethod
    def construct_insights(
        self, model_state: pd.DataFrame, current_date: pd.Timestamp
    ) -> Dict[str, Any]:
        """
        Construct portfolio insights based on the model state up to the current date.

        All strategy subclasses must implement this method to define their strategy logic.

        Args:
            model_state: Complete DataFrame containing data up to and including the current date.
                        This is the 'Model State' that includes all data available until current_date.
            current_date: The current trading day.

        Returns:
            A dictionary containing portfolio insights:
            {
                'timestamp': current_date,
                'weights': {
                    'TICKER1': weight1,
                    'TICKER2': weight2,
                    ...
                }
            }
        """
        pass


class QuantStrategyFramework:
    """
    A framework for developing and testing quantitative investment strategies.

    This framework provides an iterator-style interface that moves through trading days,
    slices the data up to the current day, and passes it to a strategy function.

    Attributes:
        data (pd.DataFrame): The complete dataset including OHLC(V) and additional features.
        date_column (str): The name of the date column in the dataset.
        current_date (pd.Timestamp): The current trading day in the iteration.
        start_date (pd.Timestamp): The starting date for the strategy execution.
        strategy (BaseStrategy): The strategy implementation to use.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        date_column: str = "Date",
        start_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
    ):
        """
        Initialize the QuantStrategyFramework with the dataset and strategy.

        Args:
            data: DataFrame containing OHLC(V) data and additional features (the Model State).
            strategy: An instance of a class implementing BaseStrategy.
            date_column: Name of the column containing date information.
            start_date: The date to start the strategy (defaults to the first date in the dataset).
        """
        # Ensure the date column is properly formatted
        if date_column in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
                data[date_column] = pd.to_datetime(data[date_column])
        else:
            raise ValueError(f"Date column '{date_column}' not found in the dataset")

        # Sort data by date
        self.data = data.sort_values(by=date_column).reset_index(drop=True)
        self.date_column = date_column

        # Store the strategy
        self.strategy = strategy

        # Extract unique dates from the dataset
        self.trading_dates = self.data[date_column].unique()

        # Set the start date
        if start_date is None:
            self.start_date = self.trading_dates[0]
        else:
            self.start_date = pd.Timestamp(start_date)
            if self.start_date < self.trading_dates[0]:
                self.start_date = self.trading_dates[0]
                print(
                    f"Warning: Provided start_date is before the first available date. "
                    f"Using {self.start_date} instead."
                )

        # Initialize the current date
        self.current_date = None
        self.current_date_index = -1

    def __iter__(self):
        """
        Initialize the iterator.

        Returns:
            The framework instance.
        """
        # Find the index for the start date
        for i, date in enumerate(self.trading_dates):
            if date >= self.start_date:
                self.current_date_index = (
                    i - 1
                )  # Set to before start so first next() gives start date
                break

        if self.current_date_index == -1 and len(self.trading_dates) > 0:
            self.current_date_index = -1  # Start from the beginning

        return self

    def __next__(self) -> Tuple[pd.Timestamp, pd.DataFrame]:
        """
        Advance to the next trading day.

        Returns:
            tuple: (current_date, sliced_data)

        Raises:
            StopIteration: When there are no more trading days.
        """
        self.current_date_index += 1

        if self.current_date_index >= len(self.trading_dates):
            raise StopIteration

        self.current_date = self.trading_dates[self.current_date_index]

        # Slice the data up to and including the current date
        sliced_data = self.data[self.data[self.date_column] <= self.current_date].copy()

        return self.current_date, sliced_data

    def run_strategy(self) -> pd.DataFrame:
        """
        Run the investment strategy from the start date to the end of available data.

        Returns:
            A DataFrame with dates as rows and tickers as columns, where each cell
            contains the portfolio weight for that ticker on that date.
        """
        all_insights = []

        for current_date, sliced_data in self:
            insight = self.strategy.construct_insights(sliced_data, current_date)

            # Ensure the insight has the correct format
            if (
                not isinstance(insight, dict)
                or "timestamp" not in insight
                or "weights" not in insight
            ):
                raise ValueError(
                    f"Strategy returned invalid insight format. Expected dict with 'timestamp' and 'weights' keys, "
                    f"got {insight}"
                )

            all_insights.append(insight)

        # Convert the list of insights to a DataFrame
        if not all_insights:
            return pd.DataFrame()  # Return empty DataFrame if no insights

        # Extract all unique tickers from all insights
        all_tickers = set()
        for insight in all_insights:
            all_tickers.update(insight["weights"].keys())

        # Create rows for the DataFrame
        rows = []
        for insight in all_insights:
            row = {"date": insight["timestamp"]}
            # Add weight for each ticker, defaulting to 0 if not allocated
            for ticker in all_tickers:
                row[ticker] = insight["weights"].get(ticker, 0.0)
            rows.append(row)

        # Create and return the DataFrame
        insights_df = pd.DataFrame(rows)

        # Sort by date
        insights_df = insights_df.sort_values("date")

        return insights_df


class EqualWeightStrategy(BaseStrategy):
    """
    A simple strategy that assigns equal weights to all assets.
    """

    def construct_insights(
        self, model_state: pd.DataFrame, current_date: pd.Timestamp
    ) -> Dict[str, Any]:
        """
        Construct equal weight portfolio.

        Args:
            model_state: Complete dataset up to current_date.
            current_date: Current trading day.

        Returns:
            Dictionary with equal weight allocation.
        """
        # Extract tickers from the model state
        tickers = []
        if "ticker" in model_state.columns:
            tickers = model_state["ticker"].unique().tolist()

        # Assign equal weights to all tickers
        weights = {}
        if tickers:
            equal_weight = 1.0 / len(tickers)
            for ticker in tickers:
                weights[ticker] = equal_weight

        return {"timestamp": current_date, "weights": weights}


# Example usage demonstration
def example_usage():
    """
    Demonstrate how to use the QuantStrategyFramework.
    """
    # Sample data creation (in a real scenario, this would be loaded from a file)
    dates = pd.date_range(start="2023-01-01", end="2023-01-10")
    assets = ["AAPL", "MSFT", "GOOGL"]

    data = []
    for date in dates:
        for asset in assets:
            # Generate some dummy OHLC data
            open_price = 100 + (hash(f"{asset}_{date}_open") % 20)
            high_price = open_price + (hash(f"{asset}_{date}_high") % 10)
            low_price = open_price - (hash(f"{asset}_{date}_low") % 10)
            close_price = open_price + (hash(f"{asset}_{date}_close") % 20) - 10
            volume = 1000 + (hash(f"{asset}_{date}_volume") % 1000)

            # Add some custom features
            momentum = (hash(f"{asset}_{date}_momentum") % 200) - 100
            volatility = (hash(f"{asset}_{date}_volatility") % 10) / 10

            data.append(
                {
                    "Date": date,
                    "ticker": asset,  # Using 'ticker' instead of 'Asset'
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                    "Volume": volume,
                    "Momentum": momentum,
                    "Volatility": volatility,
                }
            )

    # Create DataFrame
    sample_df = pd.DataFrame(data)

    # Define a custom strategy by implementing BaseStrategy
    class MomentumStrategy(BaseStrategy):
        def construct_insights(
            self, model_state: pd.DataFrame, current_date: pd.Timestamp
        ) -> Dict[str, Any]:
            """
            Implement a momentum-based strategy.

            Args:
                model_state: The data up to the current date.
                current_date: The current trading day.

            Returns:
                Dictionary with weights based on momentum.
            """
            # Simple momentum strategy:
            # 1. Get the latest data point for each asset
            latest_data = model_state.groupby("ticker").last().reset_index()

            # 2. Rank assets by momentum
            ranked_assets = latest_data.sort_values("Momentum", ascending=False)

            # 3. Allocate weights: 60% to top asset, 30% to second, 10% to third
            weights = {}
            for i, (_, row) in enumerate(ranked_assets.iterrows()):
                if i == 0:
                    weights[row["ticker"]] = 0.6
                elif i == 1:
                    weights[row["ticker"]] = 0.3
                elif i == 2:
                    weights[row["ticker"]] = 0.1
                else:
                    weights[row["ticker"]] = 0.0

            return {"timestamp": current_date, "weights": weights}

    # Create strategy instance
    momentum_strategy = MomentumStrategy()

    # Initialize the framework with the strategy
    framework = QuantStrategyFramework(
        data=sample_df,
        strategy=momentum_strategy,
        date_column="Date",
        start_date="2023-01-03",
    )

    # Run the strategy
    insights_df = framework.run_strategy()

    # Print insights
    print("Momentum Strategy Results:")
    print(insights_df.head())
    print("-" * 40)

    # Also demonstrate using the EqualWeightStrategy
    equal_weight_strategy = EqualWeightStrategy()
    equal_weight_framework = QuantStrategyFramework(
        data=sample_df,
        strategy=equal_weight_strategy,
        date_column="Date",
        start_date="2023-01-03",
    )

    equal_weight_insights_df = equal_weight_framework.run_strategy()

    print("\nEqual Weight Strategy Results:")
    print(equal_weight_insights_df.head(2))
    print("-" * 40)

    return insights_df


if __name__ == "__main__":
    # Run the example when executing the module directly
    example_usage()
