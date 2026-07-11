"""Offline self-play simulation for the bot.

Plays the bot against itself with *simulated* wall-clock time — no display,
no mouse, no real-time sleeps — and writes PGNs with [%clk] tags that
cheat_detection can analyse. The simulated clock charges each move with the
same components a live Lichess game would: decided think time, engine compute
overrun, mouse-gesture execution, detection latency and the client's
fast-path waits (ponder-dic hits, premoves, time scrambles).

Modules:
- calibrate:      Phase-0 measurement of the real components (engine compute
                  time, vision/detection latency) on this machine.
- clock:          SimClock — per-side remaining time bookkeeping.
- latency_model:  samples per-move latency components (shared formulas from
                  common/move_timing.py + calibration JSON).
- client_model:   emulates clients/mp_original.py's decision branches
                  (ponder-dic fast paths, premove queueing, drag-vs-click).
- game_runner:    two Engine instances playing one game on a SimClock.
- pgn_writer:     PGN output with integer-second [%clk] comments (matching
                  Lichess export granularity).
- run:            CLI entry point (venv/bin/python -m simulation.run).
"""
