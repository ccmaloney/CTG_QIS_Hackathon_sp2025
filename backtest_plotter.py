import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Union, Optional


class BacktestPlotter:
    """
    A class for plotting backtest results using Plotly.

    This class provides methods to create visualizations of backtest performance
    metrics like cumulative returns and drawdowns.
    """

    def __init__(self):
        """Initialize the BacktestPlotter."""
        pass

    @staticmethod
    def plot_cumulative_returns(
        backtests: List[Dict[str, pd.DataFrame]],
        names: Optional[List[str]] = None,
        title: str = "Cumulative Log Returns",
    ) -> go.Figure:
        """
        Plot cumulative log returns for multiple backtests.

        Args:
            backtests: List of backtest DataFrames, each containing 'date' and 'cumulative_log_return' columns
            names: Optional list of names for each backtest (defaults to "Strategy 1", "Strategy 2", etc.)
            title: Title for the plot

        Returns:
            Plotly figure object
        """
        if names is None:
            names = [f"Strategy {i+1}" for i in range(len(backtests))]

        fig = go.Figure()

        for i, backtest in enumerate(backtests):
            # Convert date to datetime if it's a string
            dates = pd.to_datetime(backtest["date"])

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=backtest["cumulative_log_return"],
                    mode="lines",
                    name=names[i],
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative Log Return",
            legend_title="Strategies",
            hovermode="x unified",
        )

        return fig

    @staticmethod
    def plot_drawdowns(
        backtests: List[pd.DataFrame],
        names: Optional[List[str]] = None,
        title: str = "Drawdowns",
    ) -> go.Figure:
        """
        Plot drawdowns for multiple backtests.

        Args:
            backtests: List of backtest DataFrames, each containing 'date' and 'drawdown' columns
            names: Optional list of names for each backtest (defaults to "Strategy 1", "Strategy 2", etc.)
            title: Title for the plot

        Returns:
            Plotly figure object
        """
        if names is None:
            names = [f"Strategy {i+1}" for i in range(len(backtests))]

        fig = go.Figure()

        for i, backtest in enumerate(backtests):
            # Convert date to datetime if it's a string
            dates = pd.to_datetime(backtest["date"])

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=backtest["drawdown"],
                    mode="lines",
                    name=names[i],
                    fill="tozeroy",
                    line=dict(color=f"rgba({50*i}, 0, {255-50*i}, 0.8)"),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown",
            legend_title="Strategies",
            hovermode="x unified",
            yaxis=dict(
                tickformat=".1%",
                range=[
                    min([backtest["drawdown"].min() for backtest in backtests]) * 1.1,
                    0.01,
                ],
            ),
        )

        return fig

    @staticmethod
    def plot_combined_metrics(
        backtests: List[pd.DataFrame],
        names: Optional[List[str]] = None,
        title: str = "Backtest Performance Metrics",
    ) -> go.Figure:
        """
        Create a stacked plot with cumulative returns and drawdowns for multiple backtests.

        Args:
            backtests: List of backtest DataFrames, each containing 'date', 'cumulative_log_return',
                       and 'drawdown' columns
            names: Optional list of names for each backtest (defaults to "Strategy 1", "Strategy 2", etc.)
            title: Title for the plot

        Returns:
            Plotly figure object with stacked subplots
        """
        if names is None:
            names = [f"Strategy {i+1}" for i in range(len(backtests))]

        # Create subplot with 2 rows and 1 column
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Cumulative Log Returns", "Drawdowns"),
        )

        # Custom colors for traces
        colors = ["blue", "red", "green", "purple", "orange", "brown", "pink", "gray"]

        for i, backtest in enumerate(backtests):
            # Convert date to datetime if it's a string
            dates = pd.to_datetime(backtest["date"])
            color = colors[i % len(colors)]

            # Add cumulative return trace
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=backtest["cumulative_log_return"],
                    mode="lines",
                    name=f"{names[i]} - Return",
                    line=dict(color=color),
                ),
                row=1,
                col=1,
            )

            # Add drawdown trace
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=backtest["drawdown"],
                    mode="lines",
                    name=f"{names[i]} - Drawdown",
                    fill="tozeroy",
                    line=dict(color=color, dash="dot"),
                ),
                row=2,
                col=1,
            )

        # Update layout
        fig.update_layout(
            height=800,
            title=title,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        # Update y-axis labels
        fig.update_yaxes(title_text="Cumulative Log Return", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)

        # Update x-axis labels
        fig.update_xaxes(title_text="Date", row=2, col=1)

        return fig

    @staticmethod
    def plot_metrics_dashboard(
        backtests: List[pd.DataFrame],
        names: Optional[List[str]] = None,
        title: str = "Backtest Performance Dashboard",
    ) -> go.Figure:
        """
        Create a comprehensive dashboard of backtest metrics with separate plots.

        Args:
            backtests: List of backtest DataFrames, each containing 'date', 'cumulative_log_return',
                       and 'drawdown' columns
            names: Optional list of names for each backtest (defaults to "Strategy 1", "Strategy 2", etc.)
            title: Title for the dashboard

        Returns:
            Plotly figure object with multiple plots
        """
        # Create returns and drawdown plots
        returns_plot = BacktestPlotter.plot_cumulative_returns(backtests, names)
        drawdown_plot = BacktestPlotter.plot_drawdowns(backtests, names)

        # Get the combined plot
        combined_plot = BacktestPlotter.plot_combined_metrics(backtests, names, title)

        return combined_plot


# Example usage function
def example_usage():
    """
    Example of how to use the BacktestPlotter class.
    """
    # Create some sample backtest data
    import numpy as np

    # Sample data for strategy 1
    dates = pd.date_range(start="2022-01-01", end="2022-12-31")
    np.random.seed(42)

    # Strategy 1 data
    returns1 = np.random.normal(0.0005, 0.005, len(dates)).cumsum()
    drawdowns1 = np.zeros_like(returns1)
    peak = 0
    for i in range(len(returns1)):
        if returns1[i] > peak:
            peak = returns1[i]
        drawdowns1[i] = min(0, returns1[i] - peak)

    backtest1 = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "cumulative_log_return": returns1,
            "drawdown": drawdowns1,
        }
    )

    # Strategy 2 data
    returns2 = np.random.normal(0.0003, 0.004, len(dates)).cumsum()
    drawdowns2 = np.zeros_like(returns2)
    peak = 0
    for i in range(len(returns2)):
        if returns2[i] > peak:
            peak = returns2[i]
        drawdowns2[i] = min(0, returns2[i] - peak)

    backtest2 = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "cumulative_log_return": returns2,
            "drawdown": drawdowns2,
        }
    )

    # Create plotter and visualize
    plotter = BacktestPlotter()
    fig = plotter.plot_combined_metrics(
        [backtest1, backtest2],
        names=["Momentum Strategy", "Value Strategy"],
        title="Strategy Comparison",
    )

    # Show the plot (in notebook or export to file)
    fig.show()
    # Or save to file: fig.write_html("backtest_results.html")


if __name__ == "__main__":
    example_usage()
