"""CSV + stdout logger for training runs."""

import csv
import os
import time
from collections import deque
from typing import Optional


class TrainingLogger:
    def __init__(self, log_dir: str = "logs", recent_window: int = 100) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "train.csv")
        self.recent_window = recent_window
        self._game_lengths: deque[int] = deque(maxlen=recent_window)
        self._game_outcomes: deque[int] = deque(maxlen=recent_window)  # 1/0/-1
        self._start_time = time.time()
        self._csv_file = open(self.log_path, "w", newline="")
        self._writer: Optional[csv.DictWriter] = None

    def record_game(self, length: int, outcome_winner: Optional[bool]) -> None:
        """outcome_winner: True=white, False=black, None=draw/ongoing."""
        self._game_lengths.append(length)
        if outcome_winner is None:
            code = 0
        elif outcome_winner:
            code = 1
        else:
            code = -1
        self._game_outcomes.append(code)

    def log_step(
        self, step: int, losses: dict, num_episodes: int
    ) -> None:
        elapsed = time.time() - self._start_time
        n_games = len(self._game_outcomes) or 1
        row = {
            "step": step,
            "elapsed_s": round(elapsed, 1),
            "num_episodes": num_episodes,
            "avg_game_len": (
                sum(self._game_lengths) / n_games if self._game_lengths else 0.0
            ),
            "white_win_rate": sum(1 for o in self._game_outcomes if o == 1) / n_games,
            "draw_rate": sum(1 for o in self._game_outcomes if o == 0) / n_games,
            "black_win_rate": sum(1 for o in self._game_outcomes if o == -1) / n_games,
        }
        for k, v in losses.items():
            row[f"loss_{k}"] = round(float(v), 5)
        if self._writer is None:
            self._writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._csv_file.flush()
        # Compact stdout summary
        loss_str = " ".join(f"{k}={row[f'loss_{k}']}" for k in losses)
        print(
            f"step={step:>6d} "
            f"games={num_episodes:>5d} "
            f"len={row['avg_game_len']:.1f} "
            f"W/D/B={row['white_win_rate']:.2f}/{row['draw_rate']:.2f}/{row['black_win_rate']:.2f} "
            f"t={int(elapsed):>5d}s "
            f"{loss_str}",
            flush=True,
        )

    def close(self) -> None:
        self._csv_file.close()
