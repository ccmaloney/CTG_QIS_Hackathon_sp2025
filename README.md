# Quant Investment Strategy Hackathon Framework

A simple, intuitive Python framework for developing and testing quantitative investment strategies during a hackathon.

## Overview

This framework is designed for participants with varying levels of technical expertise to easily develop and test investment strategies using OHLC(V) financial data. The framework handles data slicing, iteration through trading days, and provides a clean interface for strategy implementation.

## Features

- **User-friendly interface**: Easy to use for participants with limited technical experience
- **Iterator-style data slicing**: Automatically moves through trading days and slices data
- **Flexible strategy implementation**: Implement your strategy by overriding a single method
- **Comprehensive documentation**: Well-documented code with examples
- **Jupyter Notebook integration**: Ready to use in a Jupyter Notebook environment

## Getting Started

### Prerequisites

- Python 3.7+
- pandas
- numpy
- matplotlib (for visualization)
- jupyterlab/notebook (for running examples)

### Installation

Clone this repository:

```bash
git clone https://github.com/ccmaloney/CTG_QIS_Hackathon_sp2025.git
cd CTG_QIS_Hackathon_sp2025
```

#### Option 1: Using pip (Simpler)

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

#### Option 2: Using Poetry (Recommended for more advanced users)

If you prefer using Poetry for dependency management:

1. Install Poetry (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

4. To run Jupyter Notebook within the Poetry environment:
   ```bash
   poetry run jupyter notebook
   ```
   or
   ```bash
   poetry run jupyter lab
   ```

### Usage

1. Import the framework in your Jupyter Notebook:

```python
from quant_strategy_framework import QuantStrategyFramework
```

2. Create your strategy by subclassing `QuantStrategyFramework` and implementing the `construct_insights` method:

```python
class MyStrategy(QuantStrategyFramework):
    def construct_insights(self, sliced_data, current_date):
        # Implement your strategy logic here
        
        # Return a dictionary with portfolio weights
        return {
            'timestamp': current_date,
            'weights': {
                'ASSET1': 0.5,
                'ASSET2': 0.3,
                'ASSET3': 0.2
            }
        }
```

3. Initialize your strategy with your dataset:

```python
strategy = MyStrategy(
    data=your_dataset,
    date_column='Date',  # name of the date column
    asset_column='Asset',  # name of the asset identifier column
    start_date='2023-01-01'  # optional start date
)
```

4. Run your strategy to generate insights:

```python
insights = strategy.run_strategy()
```

5. Analyze your insights:

```python
for insight in insights[:5]:  # Look at first 5 days
    print(f"Date: {insight['timestamp'].strftime('%Y-%m-%d')}")
    print("Portfolio Weights:")
    for asset, weight in insight['weights'].items():
        print(f"  {asset}: {weight:.2f}")
    print("-" * 40)
```

## Example

Check out the included Jupyter Notebook `quant_strategy_example.ipynb` for a complete example with:
- Data preparation
- Strategy implementation
- Running and analyzing results
- Visualization

## Framework Structure

The framework operates through these main components:

1. **Initialization**: Load your dataset and configure parameters
2. **Iteration**: Move through trading days one at a time
3. **Data Slicing**: For each day, the framework slices data up to that day
4. **Strategy Execution**: Your strategy generates portfolio weights based on available data
5. **Insights Collection**: Portfolio weights are collected for each trading day

## Contributing

This framework is designed for a specific hackathon. If you have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
