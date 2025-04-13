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
from typing import Dict, Any, Union, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import multiprocessing
from functools import partial


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

    def construct_insights_multi(
        self, data: pd.DataFrame, dates: List[pd.Timestamp]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple dates in parallel for improved performance.

        By default, this method calls construct_insights sequentially for each date.
        Override this method for custom parallel implementation.

        Args:
            data: Complete DataFrame containing all data needed for all dates.
            dates: List of dates to process.

        Returns:
            List of insight dictionaries, one for each date.
        """
        insights = []
        for date in dates:
            # Slice data up to this date
            model_state = data[data[data.columns[0]] <= date]
            insight = self.construct_insights(model_state, date)
            insights.append(insight)
        return insights


class QuantStrategyFramework:
    """
    A simplified framework for developing and testing quantitative investment strategies.

    This framework handles day-by-day iteration through trading days and provides
    a clean interface for strategy implementation with optimized dataframe operations.
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
            data: DataFrame containing OHLC(V) data and additional features.
            strategy: An instance of a class implementing BaseStrategy.
            date_column: Name of the column containing date information.
            start_date: The date to start the strategy (defaults to the first date in the dataset).
        """
        # Process input data
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found in the dataset")

        # Ensure the date column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column])

        # Store sorted data by date for efficient slicing
        self.data = data.sort_values(by=date_column).reset_index(drop=True)
        self.date_column = date_column
        self.strategy = strategy

        # Extract unique trading dates once and store them
        self.trading_dates = sorted(self.data[date_column].unique())

        # Set the start date
        if start_date is None:
            self.start_date = self.trading_dates[0]
        else:
            self.start_date = pd.Timestamp(start_date)
            # Adjust if start date is before first available date
            if self.start_date < self.trading_dates[0]:
                self.start_date = self.trading_dates[0]
                print(f"Warning: Using first available date {self.start_date} instead.")

    def _process_batch(self, batch_dates: List[pd.Timestamp]) -> List[Dict[str, Any]]:
        """
        Process a batch of dates and return insights.

        Args:
            batch_dates: List of dates to process.

        Returns:
            List of insight dictionaries.
        """
        return self.strategy.construct_insights_multi(self.data, batch_dates)

    def run_strategy(self) -> pd.DataFrame:
        """
        Run the investment strategy from the start date to the end of available data.

        This method processes each trading day sequentially, providing the strategy
        with data up to and including the current date.

        Returns:
            A DataFrame with dates as rows and tickers as columns, containing portfolio weights.
        """
        all_insights = []
        all_tickers = set()

        # Find start index for efficiency
        start_idx = 0
        for i, date in enumerate(self.trading_dates):
            if date >= self.start_date:
                start_idx = i
                break

        # Process each trading day
        for i in range(start_idx, len(self.trading_dates)):
            current_date = self.trading_dates[i]

            # Optimized slicing using boolean indexing
            sliced_data = self.data[self.data[self.date_column] <= current_date]

            # Get insights from strategy
            insight = self.strategy.construct_insights(sliced_data, current_date)

            # Validate insight format
            if (
                not isinstance(insight, dict)
                or "timestamp" not in insight
                or "weights" not in insight
            ):
                raise ValueError(
                    f"Strategy returned invalid insight format at {current_date}"
                )

            # Track all tickers for final dataframe construction
            all_tickers.update(insight["weights"].keys())
            all_insights.append(insight)

        # Convert insights to DataFrame efficiently
        if not all_insights:
            return pd.DataFrame()

        # Create DataFrame with consistent columns for all tickers
        rows = []
        for insight in all_insights:
            row = {"date": insight["timestamp"]}
            # Add weight for each ticker (default 0 if not allocated)
            for ticker in all_tickers:
                row[ticker] = insight["weights"].get(ticker, 0.0)
            rows.append(row)

        # Create and return sorted DataFrame
        return pd.DataFrame(rows).sort_values("date")

    def run_strategy_parallel(
        self, num_processes: int = None, batch_size: int = 10
    ) -> pd.DataFrame:
        """
        Run the investment strategy in parallel using multiprocessing.

        This method splits the trading dates into batches and processes them in parallel,
        which can significantly speed up execution for computationally intensive strategies.

        Args:
            num_processes: Number of processes to use. If None, uses CPU count.
            batch_size: Number of dates to process in each batch.

        Returns:
            A DataFrame with dates as rows and tickers as columns, containing portfolio weights.
        """
        # Determine number of processes to use
        if num_processes is None:
            num_processes = min(multiprocessing.cpu_count(), 4)  # Reasonable default

        # Find start index
        start_idx = 0
        for i, date in enumerate(self.trading_dates):
            if date >= self.start_date:
                start_idx = i
                break

        # Get trading dates to process
        dates_to_process = self.trading_dates[start_idx:]

        if not dates_to_process:
            return pd.DataFrame()

        # Create batches of dates (avoid too many small batches for efficiency)
        date_batches = []
        for i in range(0, len(dates_to_process), batch_size):
            date_batches.append(dates_to_process[i : i + batch_size])

        # Process batches in parallel
        all_insights = []
        all_tickers = set()

        # Use multiprocessing Pool to process batches
        with multiprocessing.Pool(processes=num_processes) as pool:
            batch_results = pool.map(self._process_batch, date_batches)

            # Flatten results and collect insights
            for batch_insights in batch_results:
                for insight in batch_insights:
                    # Validate insight format
                    if (
                        not isinstance(insight, dict)
                        or "timestamp" not in insight
                        or "weights" not in insight
                    ):
                        raise ValueError(f"Strategy returned invalid insight format")

                    # Track all tickers
                    all_tickers.update(insight["weights"].keys())
                    all_insights.append(insight)

        # Convert insights to DataFrame efficiently
        if not all_insights:
            return pd.DataFrame()

        # Create DataFrame with consistent columns for all tickers
        rows = []
        for insight in all_insights:
            row = {"date": insight["timestamp"]}
            # Add weight for each ticker (default 0 if not allocated)
            for ticker in all_tickers:
                row[ticker] = insight["weights"].get(ticker, 0.0)
            rows.append(row)

        # Create and return sorted DataFrame
        return pd.DataFrame(rows).sort_values("date")

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

    def construct_insights_multi(
        self, data: pd.DataFrame, dates: List[pd.Timestamp]
    ) -> List[Dict[str, Any]]:
        """
        Optimized implementation for processing multiple dates in parallel.

        This implementation is more efficient than the default method by
        avoiding redundant calculations across dates.

        Args:
            data: Complete DataFrame.
            dates: List of dates to process.

        Returns:
            List of insight dictionaries.
        """
        insights = []
        date_column = data.columns[0]  # Assuming first column is date column

        for date in dates:
            # For equal weight strategy, we just need the tickers at each date
            date_data = data[data[date_column] <= date]
            tickers = []
            if "ticker" in date_data.columns:
                tickers = date_data["ticker"].unique().tolist()

            # Calculate weights
            weights = {}
            if tickers:
                equal_weight = 1.0 / len(tickers)
                for ticker in tickers:
                    weights[ticker] = equal_weight

            insights.append({"timestamp": date, "weights": weights})

        return insights
