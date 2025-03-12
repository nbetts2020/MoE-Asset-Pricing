import numpy as np

class OnlineMetrics:
    def __init__(self):
        self.count = 0
        self.sum_sq_error = 0.0  # For MSE
        self.sum_y = 0.0  # For R²
        self.sum_y_sq = 0.0  # For R²
        self.trend_correct = 0  # For trend accuracy
        self.excess_returns_sum = 0.0  # For Sharpe/Sortino
        self.excess_returns_sq_sum = 0.0  # For Sharpe
        self.downside_returns_sq_sum = 0.0  # For Sortino
        self.downside_count = 0
        self.strategy_returns_sum = 0.0  # For average return
        self.wins = 0  # For win rate
        self.gross_profits = 0.0  # For profit factor
        self.gross_losses = 0.0  # For profit factor

    def update(self, pred, actual, oldprice, riskfree):
        self.count += 1
        error = actual - pred
        self.sum_sq_error += error ** 2
        self.sum_y += actual
        self.sum_y_sq += actual ** 2

        true_trend = np.sign(actual - oldprice)
        pred_trend = np.sign(pred - oldprice)
        self.trend_correct += (true_trend == pred_trend)

        buy_signal = float(pred > oldprice)
        strategy_return = (actual - oldprice) / (oldprice + 1e-12) * buy_signal
        excess_return = strategy_return - riskfree
        self.excess_returns_sum += excess_return
        self.excess_returns_sq_sum += excess_return ** 2
        if excess_return < 0:
            self.downside_returns_sq_sum += excess_return ** 2
            self.downside_count += 1
        self.strategy_returns_sum += strategy_return
        if strategy_return > 0:
            self.wins += 1
            self.gross_profits += strategy_return
        elif strategy_return < 0:
            self.gross_losses -= strategy_return

    def compute(self):
        mse = self.sum_sq_error / self.count if self.count > 0 else 0.0
        mean_y = self.sum_y / self.count if self.count > 0 else 0.0
        ss_tot = self.sum_y_sq - 2 * mean_y * self.sum_y + self.count * mean_y ** 2
        ss_res = self.sum_sq_error
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        trend_acc = self.trend_correct / self.count if self.count > 0 else 0.0
        mean_excess = self.excess_returns_sum / self.count if self.count > 0 else 0.0
        variance_excess = (self.excess_returns_sq_sum / self.count - mean_excess ** 2) if self.count > 1 else 0.0
        sharpe = mean_excess / (variance_excess ** 0.5) if variance_excess > 1e-12 else 0.0
        downside_std = (self.downside_returns_sq_sum / self.downside_count) ** 0.5 if self.downside_count > 1 else 0.0
        sortino = mean_excess / downside_std if downside_std > 1e-12 else float('inf')
        avg_return = self.strategy_returns_sum / self.count if self.count > 0 else 0.0
        win_rate = (self.wins / self.count * 100) if self.count > 0 else 0.0
        profit_factor = self.gross_profits / self.gross_losses if self.gross_losses > 1e-12 else float('inf')

        return {
            "mse": mse,
            "r2": r2,
            "trend_acc": trend_acc,
            "sharpe": sharpe,
            "sortino": sortino,
            "avg_return": avg_return,
            "win_rate": win_rate,
            "profit_factor": profit_factor
        }
