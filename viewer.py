"""Real-time viewer for self-play games using the latest checkpoint.

Run as a separate process from training:

    python viewer.py --ckpt-dir checkpoints --port 5000

Then open http://localhost:5000 in a browser. The viewer plays self-vs-self
games using the most recent checkpoint and reloads between games to pick up
training progress. It runs on CPU by default so it doesn't contend with the
trainer for GPU memory.

Adjust per-move delay via the slider on the page, by POSTing to /set_delay,
or by appending ?delay=<seconds> to the URL.
"""

import argparse
import copy
import threading
import time
from collections import deque
from typing import Optional

import chess
import torch
from flask import Flask, Response, jsonify, request

from checkpoint import latest_checkpoint
from model import build_networks, create_mask
from observation import NUM_SNAPSHOTS, board_to_observation
from player import action_to_chess_move
from tree import get_target_policy


class ViewerState:
    """Shared state between the play-loop thread and HTTP handlers."""

    def __init__(self, delay: float = 1.0):
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.fen = chess.STARTING_FEN
        self.lastmove_uci: Optional[str] = None
        self.ply = 0
        self.result: Optional[str] = None
        self.ckpt_path: Optional[str] = None
        self.delay = delay
        self.version = 0

    def update(
        self,
        fen: str,
        lastmove_uci: Optional[str],
        ply: int,
        result: Optional[str],
        ckpt_path: Optional[str],
    ) -> None:
        with self.cv:
            self.fen = fen
            self.lastmove_uci = lastmove_uci
            self.ply = ply
            self.result = result
            self.ckpt_path = ckpt_path
            self.version += 1
            self.cv.notify_all()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "fen": self.fen,
                "lastmove": self.lastmove_uci,
                "ply": self.ply,
                "result": self.result,
                "ckpt": self.ckpt_path,
                "delay": self.delay,
                "version": self.version,
            }

    def set_delay(self, value: float) -> None:
        with self.lock:
            self.delay = max(0.0, value)

    def wait_for_change(self, last_version: int, timeout: float) -> int:
        with self.cv:
            if self.version == last_version:
                self.cv.wait(timeout=timeout)
            return self.version


def _load_weights(nets, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    for k in ("representation", "dynamics", "policy", "value"):
        getattr(nets, k).load_state_dict(ckpt[k])


def play_loop(state: ViewerState, ckpt_dir: str, device: torch.device) -> None:
    nets = build_networks(device=device)
    nets.eval()
    current_ckpt: Optional[str] = None

    while True:
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path is None:
            state.update(chess.STARTING_FEN, None, 0, "waiting for checkpoint", None)
            time.sleep(2.0)
            continue

        if ckpt_path != current_ckpt:
            try:
                _load_weights(nets, ckpt_path, device)
                current_ckpt = ckpt_path
                print(f"viewer: loaded {ckpt_path}", flush=True)
            except Exception as e:
                print(f"viewer: load failed for {ckpt_path}: {e}", flush=True)
                time.sleep(2.0)
                continue

        board = chess.Board()
        board_history: deque[chess.Board] = deque(maxlen=NUM_SNAPSHOTS)
        state.update(board.fen(), None, 0, None, current_ckpt)

        with torch.no_grad():
            while not board.is_game_over():
                real_board = copy.deepcopy(board)
                board_history.appendleft(real_board)
                if board.turn == chess.BLACK:
                    obs_board = copy.deepcopy(board).mirror()
                    obs_history = [copy.deepcopy(b).mirror() for b in board_history]
                else:
                    obs_board = real_board
                    obs_history = list(board_history)

                observation = board_to_observation(
                    obs_board, history=list(obs_history)
                ).to(device)
                latent = nets.representation(
                    observation.permute(2, 0, 1).unsqueeze(0)
                )
                policy_logits = nets.policy(
                    latent,
                    create_mask(obs_board, device=device).view(-1).unsqueeze(0),
                )
                target_action, _, _ = get_target_policy(
                    latent, policy_logits.squeeze(0), nets
                )
                move = action_to_chess_move(
                    target_action, board, board.turn == chess.BLACK
                )
                board.push(move)
                state.update(
                    board.fen(),
                    move.uci(),
                    len(board.move_stack),
                    None,
                    current_ckpt,
                )
                time.sleep(max(0.0, state.delay))

        outcome = board.outcome()
        if outcome is None:
            result_str = "game over"
        elif outcome.winner is None:
            result_str = f"draw ({outcome.termination.name})"
        else:
            side = "white" if outcome.winner else "black"
            result_str = f"{side} wins ({outcome.termination.name})"
        snap = state.snapshot()
        state.update(
            board.fen(),
            snap["lastmove"],
            len(board.move_stack),
            result_str,
            current_ckpt,
        )
        time.sleep(3.0)


HTML = """<!doctype html>
<html>
<head>
<title>chess viewer</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.base.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.brown.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.cburnett.css">
<style>
  body { font-family: monospace; margin: 20px; background: #1e1e1e; color: #ddd; }
  #board { width: 480px; height: 480px; }
  #info { margin-top: 12px; line-height: 1.6; }
  #info span { color: #8cf; }
  input { background: #2a2a2a; color: #ddd; border: 1px solid #444; padding: 2px 6px; }
  button { background: #333; color: #ddd; border: 1px solid #555; padding: 2px 10px; cursor: pointer; }
</style>
</head>
<body>
<div id="board"></div>
<div id="info">
  <div>ply: <span id="ply">0</span></div>
  <div>last move: <span id="lastmove">-</span></div>
  <div>result: <span id="result">-</span></div>
  <div>ckpt: <span id="ckpt">-</span></div>
  <div>
    delay (s):
    <input id="delay" type="number" step="0.1" min="0" value="1.0" />
    <button id="apply">apply</button>
  </div>
</div>
<script type="module">
  import { Chessground } from 'https://cdn.jsdelivr.net/npm/chessground@9/+esm';

  const cg = Chessground(document.getElementById('board'), {
    viewOnly: true,
    coordinates: true,
    animation: { enabled: true, duration: 200 },
    highlight: { lastMove: true, check: true },
  });

  function uciToSquares(uci) {
    if (!uci) return undefined;
    return [uci.slice(0, 2), uci.slice(2, 4)];
  }

  function refresh() {
    fetch('/state').then(r => r.json()).then(s => {
      const turn = s.fen.split(' ')[1] === 'w' ? 'white' : 'black';
      cg.set({
        fen: s.fen,
        turnColor: turn,
        lastMove: uciToSquares(s.lastmove),
      });
      document.getElementById('ply').textContent = s.ply;
      document.getElementById('lastmove').textContent = s.lastmove || '-';
      document.getElementById('result').textContent = s.result || '-';
      document.getElementById('ckpt').textContent = s.ckpt || '-';
      if (document.activeElement.id !== 'delay') {
        document.getElementById('delay').value = s.delay;
      }
    });
  }

  function setDelay() {
    const v = document.getElementById('delay').value;
    fetch('/set_delay?value=' + v, { method: 'POST' });
  }
  document.getElementById('apply').addEventListener('click', setDelay);

  const params = new URLSearchParams(window.location.search);
  if (params.has('delay')) {
    document.getElementById('delay').value = params.get('delay');
    setDelay();
  }
  const es = new EventSource('/events');
  es.onmessage = refresh;
  refresh();
</script>
</body>
</html>
"""


def make_app(state: ViewerState) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return HTML

    @app.route("/state")
    def get_state():
        return jsonify(state.snapshot())

    @app.route("/set_delay", methods=["POST", "GET"])
    def set_delay():
        try:
            value = float(request.args.get("value", "1.0"))
        except ValueError:
            return ("bad value", 400)
        state.set_delay(value)
        return ("", 204)

    @app.route("/events")
    def events():
        def stream():
            last_version = -1
            while True:
                new_version = state.wait_for_change(last_version, timeout=15.0)
                if new_version != last_version:
                    last_version = new_version
                    yield f"data: {new_version}\n\n"
                else:
                    yield ": keepalive\n\n"

        return Response(stream(), mimetype="text/event-stream")

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device for inference (default: cpu so it doesn't fight the trainer).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    state = ViewerState(delay=args.delay)
    threading.Thread(
        target=play_loop,
        args=(state, args.ckpt_dir, device),
        daemon=True,
    ).start()

    app = make_app(state)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
