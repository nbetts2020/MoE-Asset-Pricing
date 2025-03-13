import numpy as np

class OnlineMetrics:
    def __init__(self):
        self.thresholds = [0.0, 0.05, 0.10, 0.25, 0.50]  # Buy thresholds: 0%, 5%, 10%, 25%, 50%
        # Core metrics (unchanged)
        self.count = 0
        self.sum_sq_error = 0.0  # For MSE
        self.sum_y = 0.0  # For R²
        self.sum_y_sq = 0.0  # For R²
        self.trend_correct = 0  # For trend accuracy
        # Per-threshold metrics
        self.stats = {
            thresh: {
                "excess_returns_sum": 0.0,
                "excess_returns_sq_sum": 0.0,
                "downside_returns_sq_sum": 0.0,
                "downside_count": 0,
                "strategy_returns_sum": 0.0,
                "wins": 0,
                "gross_profits": 0.0,
                "gross_losses": 0.0
            } for thresh in self.thresholds
        }

    def update(self, pred, actual, oldprice, riskfree):
        self.count += 1
        error = actual - pred
        self.sum_sq_error += error ** 2
        self.sum_y += actual
        self.sum_y_sq += actual ** 2

        true_trend = np.sign(actual - oldprice)
        pred_trend = np.sign(pred - oldprice)
        self.trend_correct += (true_trend == pred_trend)

        for thresh in self.thresholds:
            threshold_price = oldprice * (1 + thresh)
            buy_signal = float(pred > threshold_price)  # 1 if pred exceeds threshold, 0 otherwise
            strategy_return = (actual - oldprice) / (oldprice + 1e-12) * buy_signal
            excess_return = strategy_return - riskfree

            stats = self.stats[thresh]
            stats["excess_returns_sum"] += excess_return
            stats["excess_returns_sq_sum"] += excess_return ** 2
            if excess_return < 0:
                stats["downside_returns_sq_sum"] += excess_return ** 2
                stats["downside_count"] += 1
            stats["strategy_returns_sum"] += strategy_return
            if strategy_return > 0:
                stats["wins"] += 1
                stats["gross_profits"] += strategy_return
            elif strategy_return < 0:
                stats["gross_losses"] -= strategy_return

    def compute(self):
        mse = self.sum_sq_error / self.count if self.count > 0 else 0.0
        mean_y = self.sum_y / self.count if self.count > 0 else 0.0
        ss_tot = self.sum_y_sq - 2 * mean_y * self.sum_y + self.count * mean_y ** 2
        ss_res = self.sum_sq_error
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        trend_acc = self.trend_correct / self.count if self.count > 0 else 0.0

        results = {"mse": mse, "r2": r2, "trend_acc": trend_acc}
        for thresh in self.thresholds:
            stats = self.stats[thresh]
            mean_excess = stats["excess_returns_sum"] / self.count if self.count > 0 else 0.0
            variance_excess = (stats["excess_returns_sq_sum"] / self.count - mean_excess ** 2) if self.count > 1 else 0.0
            sharpe = mean_excess / (variance_excess ** 0.5) if variance_excess > 1e-12 else 0.0
            downside_std = (stats["downside_returns_sq_sum"] / stats["downside_count"]) ** 0.5 if stats["downside_count"] > 1 else 0.0
            sortino = mean_excess / downside_std if downside_std > 1e-12 else float('inf')
            avg_return = stats["strategy_returns_sum"] / self.count if self.count > 0 else 0.0
            win_rate = (stats["wins"] / self.count * 100) if self.count > 0 else 0.0
            profit_factor = stats["gross_profits"] / stats["gross_losses"] if stats["gross_losses"] > 1e-12 else float('inf')

            results[f"sharpe_{thresh}"] = sharpe
            results[f"sortino_{thresh}"] = sortino
            results[f"avg_return_{thresh}"] = avg_return
            results[f"win_rate_{thresh}"] = win_rate
            results[f"profit_factor_{thresh}"] = profit_factor

        return results
