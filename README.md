# Quant Investment Strategy Hackathon Framework

A simple, intuitive Python framework for developing and testing quantitative investment strategies during a hackathon.

## Overview

This framework is designed for participants with varying levels of technical expertise to easily develop and test investment strategies using OHLC(V) financial data. The framework handles data slicing, iteration through trading days, and provides a clean interface for strategy implementation.

## Features

- **User-friendly interface**: Easy to use for participants with limited technical experience
- **Iterator-style data slicing**: Automatically moves through trading days and slices data
- **Flexible strategy implementation**: Implement your strategy by implementing the `BaseStrategy` abstract class
- **Comprehensive documentation**: Well-documented code with examples
- **Jupyter Notebook integration**: Ready to use in a Jupyter Notebook environment
- **DataFrame output**: Returns portfolio weights in a structured DataFrame format for easy analysis

## Getting Started

### Prerequisites

- Python 3.8.1+
- pandas
- numpy
- matplotlib (for visualization)
- jupyterlab/notebook (for running examples)
- VSCode or PyCharm recommended

### Installation


To download and setup VSCode please follow the instructions here:

```bash
https://code.visualstudio.com/download
```

To download and setup PyCharm please follow the instructions here:

```bash
https://www.jetbrains.com/pycharm/download/?section=mac
```

NOTE: To register for an account, please use your academic license.

Clone this repository:

```bash
git clone https://github.com/ccmaloney/CTG_QIS_Hackathon_sp2025.git
cd CTG_QIS_Hackathon_sp2025
```

#### Option 1: Using pip (Simpler)

Install dependencies using pip:

Create a virtual environment

```bash
python3 -m venv venv
```

For Mac
```bash
source venv/bin/activate
```

For Windows
```
venv\Scripts\activate
```

Install the requirements

```bash
pip install --upgrade pip  # recommended
pip install -r requirements.txt
```

Install Jupyter Kernel

```bash
python -m ipykernel install --user --name=quant-strategy --display-name "Quant Strategy"
```

Launch Jupyter Notebook (from within the virtual environment)

```bash
python -m notebook
```

If you would like to exit the virtual env, then run

```bash
deactivate
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

5. Create a Jupyter kernel for your project (optional but recommended):
   ```bash
   poetry run python -m ipykernel install --user --name quant-strategy --display-name "Quant Strategy"
   ```

### Usage

1. Import the framework in your Jupyter Notebook:

```python
from quant_strategy_framework import QuantStrategyFramework, BaseStrategy
```

2. Create your strategy by implementing the `BaseStrategy` abstract class:

```python
class MyStrategy(BaseStrategy):
    def construct_insights(self, model_state, current_date):
        # Implement your strategy logic here
        
        # Return a dictionary with portfolio weights
        return {
            'timestamp': current_date,
            'weights': {
                'TICKER1': 0.5,
                'TICKER2': 0.3,
                'TICKER3': 0.2
            }
        }
```

3. Initialize the framework with your strategy and dataset:

```python
# Create strategy instance
my_strategy = MyStrategy()

# Initialize framework
framework = QuantStrategyFramework(
    data=your_dataset,
    strategy=my_strategy,
    date_column='Date',  # name of the date column
    start_date='2023-01-01'  # optional start date
)
```

4. Run the strategy to generate insights as a DataFrame:

```python
insights_df = framework.run_strategy()
```

5. Analyze your insights (now in DataFrame format):

```python
# First 5 rows of the insights DataFrame
print(insights_df.head())

# Plot the portfolio weights over time
import matplotlib.pyplot as plt

plot_df = insights_df.set_index('date')
plot_df.plot.area(stacked=True)
plt.title('Portfolio Weights Over Time')
plt.show()
```

## Example

Check out the included Jupyter Notebook `quant_strategy_example.ipynb` for a complete example with:
- Data preparation
- Strategy implementation
- Running and analyzing results
- Visualization

## Framework Structure

The framework operates through these main components:

1. **Strategy Definition**: Implement the `BaseStrategy` abstract class to define your investment logic
2. **Framework Initialization**: Load your dataset and create a framework instance with your strategy
3. **Iteration**: The framework moves through trading days one at a time
4. **Data Slicing**: For each day, the framework slices data up to that day and passes it to your strategy
5. **Strategy Execution**: Your strategy generates portfolio weights based on available data
6. **DataFrame Output**: Portfolio weights are compiled into a DataFrame with dates as rows and tickers as columns

## Contributing

This framework is designed for a specific hackathon. If you have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
