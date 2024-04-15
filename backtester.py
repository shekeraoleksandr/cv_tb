from tqdm import tqdm  # Import tqdm for the progress bar functionality
from trading_strategy import TradingStrategy


class Backtester:
    def __init__(self, initial_capital, transaction_cost_percent, model_path):
        self.initial_capital = initial_capital
        self.transaction_cost_percent = transaction_cost_percent  # Already a fraction like 0.15
        self.trading_model = TradingStrategy(model_path)
        self.position_open = False
        self.entry_price = None
        self.position_size = None

    def simulate_trading(self, data):
        capital = self.initial_capital
        position = 0
        portfolio_values = []

        # Use tqdm here to wrap the iteration
        for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Simulating Trades", unit="trade"):
            current_price = row['Close']

            if not self.position_open:
                decision = self.trading_model.decide_trade({
                    'Open': row['Open'],
                    'High': row['High'],
                    'Low': row['Low'],
                    'Close': current_price,
                    'Volume': row['Volume']
                })

                if decision == 'buy' and capital > 0:
                    trade_capital = capital * self.transaction_cost_percent
                    position = trade_capital / current_price  # Compute position size
                    self.entry_price = current_price
                    self.position_open = True
                    self.position_size = position

            else:  # Check conditions to close the position
                profit_target = self.entry_price * 1.05
                stop_loss_target = self.entry_price * 0.85

                if current_price >= profit_target or current_price <= stop_loss_target:
                    capital += self.position_size * current_price  # Sell all the position
                    self.position_open = False
                    self.position_size = 0

            current_value = capital + (self.position_size * current_price if self.position_open else 0)
            portfolio_values.append(round(current_value))

        return portfolio_values


