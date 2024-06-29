import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MarkovModel:
    def __init__(self, data):
        self.data = data
        self.transition_matrix = self.build_transition_matrix()

    def build_transition_matrix(self):
        transitions = {}
        states = self.data['Close'].pct_change().dropna().apply(lambda x: round(x, 2))

        for prev, curr in zip(states[:-1], states[1:]):
          if prev not in transitions:
               transitions[prev] = {}
          if curr not in transitions[prev]:
               transitions[prev][curr] = 0
          transitions[prev][curr] += 1

        for prev, currs in transitions.items():
          total = sum(currs.values())
          for key in currs:
               currs[key] /= total

        return transitions

    def next_state(self, current_state):
        if current_state in self.transition_matrix:
          next_states = self.transition_matrix[current_state]
          return np.random.choice(list(next_states.keys()), p=list(next_states.values()))
        return current_state

class TradingBot:
    def __init__(self, model, initial_cash=10000, tick_value=12.5):
        self.model = model
        self.cash = initial_cash
        self.tick_value = tick_value
        self.buy_points = []
        self.sell_points = []
        self.current_position = None 
        self.martingale_factor = 1
        self.holdings=0

    def simulate_trades(self):
        data = self.model.data

        for i in range(1, len(data)):
          current_date = data.index[i]
          current_price = data['Close'].iloc[i]
          prev_price = data['Close'].iloc[i - 1]
          state = round((current_price - prev_price) / prev_price, 2)
          predicted_state = self.model.next_state(state)
          predicted_price = current_price * (1 + predicted_state)

          # Close existing position if any
          if self.current_position == 'long':
               self.cash += self.tick_value * self.buy_points[-1][1] 
               self.sell_points.append((current_date, self.buy_points[-1][1]))
               self.buy_points.pop()
               self.current_position = None
          elif self.current_position == 'short':
               self.cash += self.holdings * current_price * self.tick_value
               self.holdings = 0
               self.sell_points.append((current_date, current_price))
               self.current_position = None

          # Predicted buy condition
          if predicted_price > current_price * 1.01:
               amount = self.cash // (current_price * self.tick_value) * self.martingale_factor
               self.holdings += amount
               self.cash -= amount * current_price * self.tick_value
               self.buy_points.append((current_date, current_price))
               self.current_position = 'long'
               self.martingale_factor = 1 

          # Predicted sell condition
          elif predicted_price < current_price * 0.99:
               # Sell with Martingale strategy
               amount = self.cash // (current_price * self.tick_value) * self.martingale_factor
               self.holdings += amount
               self.cash -= amount * current_price * self.tick_value
               self.buy_points.append((current_date, current_price))
               self.current_position = 'short'
               self.martingale_factor = 1 

        if self.current_position == 'long':
          self.cash += self.tick_value * self.buy_points[-1][1]  # Sell at last buy price
          self.sell_points.append((data.index[-1], self.buy_points[-1][1]))
        elif self.current_position == 'short':
          self.cash += self.holdings * data['Close'].iloc[-1] * self.tick_value  # Buy back at last close price
          self.sell_points.append((data.index[-1], data['Close'].iloc[-1]))

        print(f"Final cash: {self.cash:.2f}")

    def plot_trades(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.model.data['Close'], label='Close Price', color='blue')

        buy_dates, buy_prices = zip(*self.buy_points) if self.buy_points else ([], [])
        sell_dates, sell_prices = zip(*self.sell_points) if self.sell_points else ([], [])

        plt.scatter(buy_dates, buy_prices, marker='^', color='green', label='Buy')
        plt.scatter(sell_dates, sell_prices, marker='v', color='red', label='Sell')

        plt.title('Trading Bot Buy and Sell Points')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

def main():
    ticker = 'ES=F'
    data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
    model = MarkovModel(data)
    bot = TradingBot(model)
    bot.simulate_trades()
    bot.plot_trades()

if __name__ == "__main__":
    main()