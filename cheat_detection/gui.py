"""Tkinter GUI for the human-likeness diagnostic and the self-play simulator.

Run:  venv/bin/python -m cheat_detection.gui

Diagnostic tab: enter the bot's Lichess account, rating band and time control,
point at a baseline (or a human corpus to build one), and hit Run. The heavy
analysis runs in a worker thread; progress streams to the log, and the result
renders as a z-score chart plus a feature table. A previously saved
report.json can be opened directly for instant viewing.

Simulation tab: configure a self-play matchup — each bot's name, rating,
difficulty, quickness and mouse speed, with fixed or alternating colours —
and run `simulation.run` as a subprocess; its progress streams into the tab's
log, and the finished PGN can be handed straight to the Diagnostic tab.
"""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .config import AnalysisConfig  # noqa: E402
from .orchestrate import DiagnosticSpec, run_diagnostic, save_result  # noqa: E402
from .report import FEATURE_META  # noqa: E402

_HERE = os.path.dirname(__file__)


class DiagnosticGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Bot human-likeness diagnostic")
        self.root.geometry("1150x780")
        self.q: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.last_result = None
        self.sim_proc: subprocess.Popen | None = None

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True)
        self.nb = nb
        self.tab_diag = ttk.Frame(nb)
        self.tab_sim = ttk.Frame(nb)
        nb.add(self.tab_diag, text="Diagnostic")
        nb.add(self.tab_sim, text="Simulation")

        self._build_inputs()
        self._build_examples()
        self._build_body()
        self._build_sim_tab()
        self.root.after(120, self._drain_queue)

    # ---------------------------------------------------------------- inputs
    def _build_inputs(self):
        f = ttk.LabelFrame(self.tab_diag, text="Configuration", padding=8)
        f.pack(fill="x", padx=8, pady=(8, 4))

        self.v_user = tk.StringVar(value="JXu2019")
        self.v_rmin = tk.StringVar(value="2300")
        self.v_rmax = tk.StringVar(value="2600")
        self.v_perf = tk.StringVar(value="bullet")
        self.v_tc = tk.StringVar(value="60+0")
        self.v_baseline = tk.StringVar(
            value=os.path.join(_HERE, "baselines", "bullet_1plus0_2300_2600.json"))
        self.v_corpus = tk.StringVar(
            value=os.path.join(_HERE, "corpora", "bullet_1plus0_2300_2600.pgn"))
        self.v_botpgn = tk.StringVar(value="")
        self.v_botmax = tk.StringVar(value="300")
        self.v_basemax = tk.StringVar(value="250")
        self.v_depth = tk.StringVar(value="10")
        self.v_multipv = tk.StringVar(value="5")
        self.v_workers = tk.StringVar(value="1")
        self.v_test = tk.StringVar(value="effect-size")
        self.v_alpha = tk.StringVar(value="0.05")
        self.v_oppmin = tk.StringVar(value="")
        self.v_oppmax = tk.StringVar(value="")
        self.v_diffmin = tk.StringVar(value="")
        self.v_diffmax = tk.StringVar(value="")

        def row(r, label, var, width=22, browse=None):
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", padx=4, pady=2)
            e = ttk.Entry(f, textvariable=var, width=width)
            e.grid(row=r, column=1, sticky="w", padx=4, pady=2)
            if browse:
                ttk.Button(f, text="...", width=3,
                           command=browse).grid(row=r, column=2, sticky="w")
            return e

        row(0, "Lichess account", self.v_user)
        ttk.Label(f, text="Rating band").grid(row=1, column=0, sticky="w", padx=4)
        rb = ttk.Frame(f)
        rb.grid(row=1, column=1, sticky="w")
        ttk.Entry(rb, textvariable=self.v_rmin, width=6).pack(side="left")
        ttk.Label(rb, text=" to ").pack(side="left")
        ttk.Entry(rb, textvariable=self.v_rmax, width=6).pack(side="left")
        ttk.Label(f, text="Time control").grid(row=2, column=0, sticky="w", padx=4)
        tcf = ttk.Frame(f)
        tcf.grid(row=2, column=1, sticky="w", padx=4)
        ttk.Combobox(tcf, textvariable=self.v_perf, width=12, state="readonly",
                     values=["bullet", "blitz", "rapid", "classical"]).pack(side="left")
        ttk.Label(tcf, text="  exact clock").pack(side="left")
        ttk.Entry(tcf, textvariable=self.v_tc, width=8).pack(side="left", padx=(4, 0))
        ttk.Label(tcf, text="(e.g. 60+0; blank = whole category)",
                  foreground="#888").pack(side="left", padx=(4, 0))

        row(3, "Baseline JSON", self.v_baseline, 48,
            browse=lambda: self._pick(self.v_baseline, [("JSON", "*.json")]))
        row(4, "Corpus PGN (to build baseline)", self.v_corpus, 48,
            browse=lambda: self._pick(self.v_corpus, [("PGN", "*.pgn")]))
        row(5, "Bot PGN (blank = fetch)", self.v_botpgn, 48,
            browse=lambda: self._pick(self.v_botpgn, [("PGN", "*.pgn")]))

        adv = ttk.Frame(f)
        adv.grid(row=6, column=0, columnspan=3, sticky="w", pady=(4, 0))
        for lbl, var, w in [("Bot games", self.v_botmax, 6),
                            ("Baseline games", self.v_basemax, 6),
                            ("Depth", self.v_depth, 5),
                            ("MultiPV", self.v_multipv, 5),
                            # parallel engine processes; ~cores/2 is a good max
                            ("Workers", self.v_workers, 5)]:
            ttk.Label(adv, text=lbl).pack(side="left", padx=(8, 2))
            ttk.Entry(adv, textvariable=var, width=w).pack(side="left")
        # effect-size: flag |z| >= 2 vs human spread (sample-size independent).
        # Welch t-test: flag p < alpha; sensitivity grows with more games.
        ttk.Label(adv, text="Test").pack(side="left", padx=(8, 2))
        ttk.Combobox(adv, textvariable=self.v_test, width=11, state="readonly",
                     values=["effect-size", "Welch t-test"]).pack(side="left")
        ttk.Label(adv, text="α").pack(side="left", padx=(8, 2))
        ttk.Entry(adv, textvariable=self.v_alpha, width=5).pack(side="left")

        # Optional pair-level unit filters; applied to BOTH the human baseline
        # (when it's built this run) and the bot's games, so the populations
        # stay comparable. Blank = off.
        flt = ttk.Frame(f)
        flt.grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(flt, text="Filters (optional):  opponent Elo").pack(side="left")
        for var in (self.v_oppmin, self.v_oppmax):
            ttk.Entry(flt, textvariable=var, width=6).pack(side="left", padx=2)
        ttk.Label(flt, text="Elo diff (self−opp)").pack(side="left", padx=(10, 0))
        for var in (self.v_diffmin, self.v_diffmax):
            ttk.Entry(flt, textvariable=var, width=6).pack(side="left", padx=2)
        ttk.Label(flt, text="e.g. diff max −200 = outrated by ≥200",
                  foreground="#888").pack(side="left", padx=(8, 0))

        btns = ttk.Frame(f)
        btns.grid(row=8, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.run_btn = ttk.Button(btns, text="Run diagnostic", command=self._on_run)
        self.run_btn.pack(side="left", padx=4)
        ttk.Button(btns, text="Open report JSON...",
                   command=self._open_report).pack(side="left", padx=4)
        ttk.Button(btns, text="Save report...",
                   command=self._save_report).pack(side="left", padx=4)
        self.phase_lbl = ttk.Label(btns, text="idle", foreground="#888")
        self.phase_lbl.pack(side="left", padx=12)

    def _pick(self, var, types):
        path = filedialog.askopenfilename(filetypes=types + [("All", "*.*")])
        if path:
            var.set(path)

    # -------------------------------------------------------------- examples
    def _build_examples(self):
        f = ttk.LabelFrame(
            self.tab_diag,
            text="Data source examples — read-only, copy/paste (shows the granularity needed)",
            padding=8)
        f.pack(fill="x", padx=8, pady=(0, 4))
        f.columnconfigure(1, weight=1)

        # Bot games: Lichess user API, built live from the fields above.
        ttk.Label(f, text="Bot games (Lichess API):").grid(row=0, column=0, sticky="w", padx=4)
        self.v_apilink = tk.StringVar()
        ttk.Entry(f, textvariable=self.v_apilink, state="readonly").grid(
            row=0, column=1, sticky="we", padx=4)
        ttk.Button(f, text="Copy", width=6,
                   command=lambda: self._copy(self.v_apilink.get())).grid(row=0, column=2, padx=2)

        # Human corpus: raw database dump (filtered later by fetch_corpus).
        ttk.Label(f, text="Human corpus (DB dump):").grid(row=1, column=0, sticky="w", padx=4)
        self.v_dumplink = tk.StringVar(
            value="https://database.lichess.org/standard/"
                  "lichess_db_standard_rated_2026-05.pgn.zst")
        ttk.Entry(f, textvariable=self.v_dumplink, state="readonly").grid(
            row=1, column=1, sticky="we", padx=4)
        ttk.Button(f, text="Copy", width=6,
                   command=lambda: self._copy(self.v_dumplink.get())).grid(row=1, column=2, padx=2)

        ttk.Label(
            f, foreground="#888",
            text="API link returns full PGNs with clocks for one account+speed. "
                 "The dump is every rated game — fetch_corpus filters it to your "
                 "rating band + time control before it becomes a baseline.",
        ).grid(row=2, column=1, sticky="w", padx=4, pady=(2, 0))

        for v in (self.v_user, self.v_perf, self.v_botmax):
            v.trace_add("write", lambda *_: self._refresh_links())
        self._refresh_links()

    def _refresh_links(self):
        user = self.v_user.get().strip() or "<username>"
        self.v_apilink.set(
            f"https://lichess.org/api/games/user/{user}"
            f"?rated=true&perfType={self.v_perf.get()}&clocks=true"
            f"&max={self.v_botmax.get()}")

    def _copy(self, text: str):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    # ------------------------------------------------------------------ body
    def _build_body(self):
        body = ttk.Frame(self.tab_diag)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        self.verdict = ttk.Label(left, text="", font=("TkDefaultFont", 11, "bold"))
        self.verdict.pack(anchor="w")
        chart_bar = ttk.Frame(left)
        chart_bar.pack(anchor="w", pady=(2, 0))
        ttk.Button(chart_bar, text="⟵ Overview",
                   command=self._show_overview).pack(side="left")
        self.chart_hint = ttk.Label(
            chart_bar, foreground="#888",
            text="  click a feature in the table to see its distribution")
        self.chart_hint.pack(side="left")
        self.fig = Figure(figsize=(5.5, 5.2), dpi=96)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_empty()

        right = ttk.Frame(body)
        right.pack(side="right", fill="both", expand=True, padx=(8, 0))
        cols = ("feature", "human", "bot", "z", "p", "vratio", "vp")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=16)
        for c, txt, w in [("feature", "Feature", 200), ("human", "Human m±sd", 115),
                          ("bot", "Bot", 75), ("z", "z", 55), ("p", "p (Welch)", 75),
                          ("vratio", "σ ratio", 60), ("vp", "var p", 65)]:
            self.tree.heading(c, text=txt)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_feature_select)

        logf = ttk.LabelFrame(self.tab_diag, text="Log", padding=4)
        logf.pack(fill="x", padx=8, pady=(0, 8))
        self.log = tk.Text(logf, height=8, wrap="word")
        self.log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        sb.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=sb.set)

    def _draw_empty(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, "Run a diagnostic to see divergences",
                     ha="center", va="center", color="#999")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

    # ---------------------------------------------------------- simulation tab
    def _build_sim_tab(self):
        from common.constants import DIFFICULTY, MOUSE_QUICKNESS, QUICKNESS
        t = self.tab_sim

        gen = ttk.LabelFrame(t, text="Match", padding=8)
        gen.pack(fill="x", padx=8, pady=(8, 4))
        self.s_games = tk.StringVar(value="20")
        self.s_tc = tk.StringVar(value="60+0")
        self.s_seed = tk.StringVar(value="0")
        self.s_workers = tk.StringVar(value="5")
        self.s_sides = tk.StringVar(value="alternate")
        self.s_out = tk.StringVar(value="")
        row = ttk.Frame(gen)
        row.pack(anchor="w")
        for lbl, var, w in [("Games", self.s_games, 6), ("TC", self.s_tc, 7),
                            ("Seed", self.s_seed, 7), ("Workers", self.s_workers, 5)]:
            ttk.Label(row, text=lbl).pack(side="left", padx=(8, 2))
            ttk.Entry(row, textvariable=var, width=w).pack(side="left")
        ttk.Label(row, text="Sides").pack(side="left", padx=(8, 2))
        ttk.Combobox(row, textvariable=self.s_sides, width=10, state="readonly",
                     values=["fixed", "alternate"]).pack(side="left")
        row2 = ttk.Frame(gen)
        row2.pack(anchor="w", fill="x", pady=(4, 0))
        ttk.Label(row2, text="Output PGN (blank = auto)").pack(side="left", padx=(8, 2))
        ttk.Entry(row2, textvariable=self.s_out, width=52).pack(side="left")
        ttk.Label(row2, foreground="#888",
                  text="  workers each own an engine pair — keep ≲ cores/3"
                  ).pack(side="left")

        # Per-bot personas. Blank fields fall back to common.constants defaults.
        bots = ttk.Frame(t)
        bots.pack(fill="x", padx=8, pady=4)
        self.s_bot = {}
        defaults = {"name": ("SimBotWhite", "SimBotBlack"), "rating": ("2450", "2450"),
                    "difficulty": ("", ""), "quickness": ("", ""), "mouse": ("", "")}
        hints = {"difficulty": f"blank = {DIFFICULTY}",
                 "quickness": f"blank = {QUICKNESS}",
                 "mouse": f"blank = {MOUSE_QUICKNESS}"}
        for i, tag in enumerate(("a", "b")):
            lf = ttk.LabelFrame(
                bots, padding=8,
                text=f"Bot {tag.upper()}"
                     + (" (white when sides=fixed)" if tag == "a" else ""))
            lf.pack(side="left", fill="both", expand=True, padx=(0 if i == 0 else 8, 0))
            self.s_bot[tag] = {}
            for r, field in enumerate(("name", "rating", "difficulty",
                                       "quickness", "mouse")):
                var = tk.StringVar(value=defaults[field][i])
                self.s_bot[tag][field] = var
                ttk.Label(lf, text=field.capitalize()).grid(
                    row=r, column=0, sticky="w", padx=2, pady=1)
                ttk.Entry(lf, textvariable=var, width=14).grid(
                    row=r, column=1, sticky="w", padx=2, pady=1)
                if field in hints:
                    ttk.Label(lf, text=hints[field], foreground="#888").grid(
                        row=r, column=2, sticky="w", padx=4)

        btns = ttk.Frame(t)
        btns.pack(anchor="w", padx=8, pady=4)
        self.sim_run_btn = ttk.Button(btns, text="Run simulation",
                                      command=self._on_sim_run)
        self.sim_run_btn.pack(side="left", padx=4)
        self.sim_stop_btn = ttk.Button(btns, text="Stop", state="disabled",
                                       command=self._on_sim_stop)
        self.sim_stop_btn.pack(side="left", padx=4)
        self.sim_analyze_btn = ttk.Button(btns, text="Analyze in Diagnostic tab →",
                                          state="disabled",
                                          command=self._sim_to_diagnostic)
        self.sim_analyze_btn.pack(side="left", padx=12)
        self.sim_status = ttk.Label(btns, text="idle", foreground="#888")
        self.sim_status.pack(side="left", padx=12)

        logf = ttk.LabelFrame(t, text="Simulation log", padding=4)
        logf.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.sim_log = tk.Text(logf, height=18, wrap="word")
        self.sim_log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(logf, command=self.sim_log.yview)
        sb.pack(side="right", fill="y")
        self.sim_log.configure(yscrollcommand=sb.set)

    def _sim_cmd(self) -> tuple[list[str], str]:
        """Build the simulation.run argv from the fields; returns (argv, out_path)."""
        tc = self.s_tc.get().strip()
        seed = int(self.s_seed.get())
        out = self.s_out.get().strip() or os.path.join(
            os.path.dirname(_HERE), "simulation", "games",
            f"gui_{tc.replace('+', 'plus')}_seed{seed}.pgn")
        argv = [sys.executable, "-u", "-m", "simulation.run", "--plain",
                "--games", str(int(self.s_games.get())),
                "--tc", tc, "--seed", str(seed),
                "--workers", str(int(self.s_workers.get())),
                "--sides", self.s_sides.get(), "--out", out]
        for tag in ("a", "b"):
            for field, flag in [("name", "name"), ("rating", "rating"),
                                ("difficulty", "difficulty"),
                                ("quickness", "quickness"), ("mouse", "mouse")]:
                v = self.s_bot[tag][field].get().strip()
                if v:
                    argv += [f"--{tag}-{flag}", v]
        return argv, out

    def _on_sim_run(self):
        if self.sim_proc and self.sim_proc.poll() is None:
            messagebox.showinfo("Busy", "A simulation is already running.")
            return
        try:
            argv, out = self._sim_cmd()
        except ValueError as e:
            messagebox.showerror("Bad input", str(e))
            return
        self.sim_out_path = out
        self.sim_log.delete("1.0", "end")
        self.q.put(("simlog", "$ " + " ".join(argv[3:])))  # skip python -u -m
        self.sim_run_btn.configure(state="disabled")
        self.sim_stop_btn.configure(state="normal")
        self.sim_analyze_btn.configure(state="disabled")
        self.sim_status.configure(text="running...")
        # Subprocess (not a thread): engines are heavy and crash-isolated this
        # way, and --plain progress streams line-by-line on stderr.
        self.sim_proc = subprocess.Popen(
            argv, cwd=os.path.dirname(_HERE),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True)

        def pump(proc):
            for line in proc.stdout:
                self.q.put(("simlog", line.rstrip("\n")))
            self.q.put(("simdone", proc.wait()))

        threading.Thread(target=pump, args=(self.sim_proc,), daemon=True).start()

    def _on_sim_stop(self):
        if self.sim_proc and self.sim_proc.poll() is None:
            # Kill the whole group: run.py spawns worker processes + Stockfish.
            os.killpg(os.getpgid(self.sim_proc.pid), signal.SIGTERM)
            self.q.put(("simlog", "[stopped by user]"))

    def _sim_to_diagnostic(self):
        self.v_botpgn.set(self.sim_out_path)
        self.v_user.set(self.s_bot["a"]["name"].get().strip() or "SimBotWhite")
        self.nb.select(self.tab_diag)
        self._log("Loaded simulation PGN into Bot PGN; account set to bot A "
                  "(rerun with bot B's name to profile the other side).")

    # ------------------------------------------------------------- run/thread
    def _log(self, msg): self.q.put(("log", msg))

    def _cfg(self) -> AnalysisConfig:
        cfg = AnalysisConfig()
        cfg.depth = int(self.v_depth.get())
        cfg.multipv = int(self.v_multipv.get())
        cfg.workers = int(self.v_workers.get() or 1)
        cfg.test_mode = "welch" if self.v_test.get() == "Welch t-test" else "effect_size"
        cfg.flag_pvalue = float(self.v_alpha.get())
        return cfg

    def _on_run(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "A diagnostic is already running.")
            return
        def band(vmin, vmax):
            """(min, max) from two entry fields; blank pair = None, blank end = open."""
            a, b = vmin.get().strip().replace("−", "-"), vmax.get().strip().replace("−", "-")
            if not a and not b:
                return None
            return (int(a) if a else -9999, int(b) if b else 9999)

        try:
            spec = DiagnosticSpec(
                username=self.v_user.get().strip(),
                rating_band=(int(self.v_rmin.get()), int(self.v_rmax.get())),
                perf=self.v_perf.get(),
                time_control=self.v_tc.get().strip() or None,
                bot_pgn=self.v_botpgn.get().strip() or None,
                baseline_path=self.v_baseline.get().strip(),
                corpus_pgn=self.v_corpus.get().strip() or None,
                bot_max_games=int(self.v_botmax.get()),
                baseline_max_games=int(self.v_basemax.get()) if self.v_basemax.get() else None,
                fetch_max_games=int(self.v_botmax.get()),
                opponent_band=band(self.v_oppmin, self.v_oppmax),
                diff_range=band(self.v_diffmin, self.v_diffmax),
            )
            if spec.time_control:
                from .fetch_lichess import parse_time_control
                parse_time_control(spec.time_control)  # fail fast on bad input
        except ValueError as e:
            messagebox.showerror("Bad input", str(e))
            return
        cfg = self._cfg()
        workdir = os.path.join(_HERE, "runs")
        self.run_btn.configure(state="disabled")
        self.log.delete("1.0", "end")

        def work():
            try:
                result = run_diagnostic(
                    cfg, spec, workdir,
                    on_log=lambda m: self.q.put(("log", m)),
                    on_progress=lambda n: self.q.put(("progress", n)),
                    on_phase=lambda p: self.q.put(("phase", p)),
                )
                self.q.put(("done", result))
            except Exception as e:  # surface any failure to the UI
                self.q.put(("error", f"{type(e).__name__}: {e}"))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _drain_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.log.insert("end", payload + "\n")
                    self.log.see("end")
                elif kind == "progress":
                    self.phase_lbl.configure(text=f"analysing game {payload}...")
                elif kind == "phase":
                    self.phase_lbl.configure(text=f"phase: {payload}")
                elif kind == "error":
                    self.run_btn.configure(state="normal")
                    self.phase_lbl.configure(text="error")
                    messagebox.showerror("Diagnostic failed", payload)
                elif kind == "done":
                    self.run_btn.configure(state="normal")
                    self.phase_lbl.configure(text="done")
                    self.last_result = payload
                    self._render(payload.report)
                elif kind == "simlog":
                    self.sim_log.insert("end", payload + "\n")
                    self.sim_log.see("end")
                    if payload.startswith("[progress]"):
                        self.sim_status.configure(text=payload[len("[progress] "):])
                elif kind == "simdone":
                    self.sim_run_btn.configure(state="normal")
                    self.sim_stop_btn.configure(state="disabled")
                    ok = payload == 0
                    self.sim_status.configure(
                        text="done" if ok else f"exited with code {payload}")
                    if ok:
                        self.sim_analyze_btn.configure(state="normal")
        except queue.Empty:
            pass
        self.root.after(120, self._drain_queue)

    # -------------------------------------------------------------- rendering
    def _render(self, report: dict):
        self.current_report = report
        feats = report["features"]
        scored = [f for f in feats if f.get("zscore") is not None]
        mean_abs_z = sum(abs(f["zscore"]) for f in scored) / len(scored) if scored else 0.0
        verdict = ("within normal human variation" if mean_abs_z < 1
                   else "noticeably distinguishable" if mean_abs_z < 2
                   else "strongly distinguishable")
        n_flag = sum(1 for f in feats if f.get("flagged"))
        welch = report.get("test_mode") == "welch"
        if welch:
            alpha = report.get("flag_pvalue", 0.05)
            flag_desc = f"{n_flag} feature(s) significant at p<{alpha:g} (Welch)"
        else:
            flag_desc = f"{n_flag} feature(s) beyond 2σ"
        n_vflag = sum(1 for f in feats if f.get("var_flagged"))
        self.verdict.configure(
            text=f"Overall divergence: mean |z| = {mean_abs_z:.2f}  ({verdict}); "
                 f"{flag_desc}; {n_vflag} with un-human game-to-game variance")

        self._draw_overview()

        # table (row iid = feature key, so selection can drill down)
        self.tree.delete(*self.tree.get_children())
        if welch:
            order = sorted(feats, key=lambda f: f.get("p_value")
                           if f.get("p_value") is not None else 2.0)
        else:
            order = sorted(feats, key=lambda f: -(abs(f["zscore"]) if f.get("zscore") is not None else -1))
        for f in order:
            label = FEATURE_META.get(f["key"], (f["key"],))[0]
            hm, hs = f.get("human_mean"), f.get("human_std")
            human = f"{hm:.3f}±{hs:.3f}" if hm is not None and hs is not None else "n/a"
            bot = "n/a" if f.get("bot_value") is None else f"{f['bot_value']:.3f}"
            z = "n/a" if f.get("zscore") is None else f"{f['zscore']:.2f}"
            p = "n/a" if f.get("p_value") is None else f"{f['p_value']:.4f}"
            vr = "n/a" if f.get("var_ratio") is None else f"{f['var_ratio']:.2f}"
            vp = "n/a" if f.get("var_pvalue") is None else f"{f['var_pvalue']:.4f}"
            tag = ("flag" if f.get("flagged")
                   else "varflag" if f.get("var_flagged") else "")
            self.tree.insert("", "end", iid=f["key"],
                             values=(label, human, bot, z, p, vr, vp), tags=(tag,))
        self.tree.tag_configure("flag", background="#ffe0e0")
        self.tree.tag_configure("varflag", background="#fff2d8")

    def _draw_overview(self):
        """Horizontal z-score bars for all features, sorted by |z|."""
        report = getattr(self, "current_report", None)
        if not report:
            self._draw_empty()
            return
        scored = [f for f in report["features"] if f.get("zscore") is not None]
        rows = sorted(scored, key=lambda f: abs(f["zscore"]))
        labels = [FEATURE_META.get(f["key"], (f["key"],))[0] for f in rows]
        zs = [f["zscore"] for f in rows]
        # Red = flagged by the active test; orange = large-ish effect anyway.
        colors = ["#d62728" if f.get("flagged")
                  else "#ff7f0e" if abs(f["zscore"]) >= 1 else "#4c78a8"
                  for f in rows]
        self.ax.clear()
        self.ax.barh(range(len(zs)), zs, color=colors)
        self.ax.set_yticks(range(len(zs)))
        self.ax.set_yticklabels(labels, fontsize=7)
        self.ax.axvline(0, color="#333", lw=0.8)
        for x in (-2, 2):
            self.ax.axvline(x, color="#d62728", lw=0.7, ls="--", alpha=0.6)
        self.ax.set_xlabel("z-score vs human baseline  (0 = perfectly human)")
        self.ax.set_title("Divergence per feature")
        self.fig.tight_layout()
        self.canvas.draw()

    def _show_overview(self):
        for sel in self.tree.selection():
            self.tree.selection_remove(sel)
        self._draw_overview()

    def _on_feature_select(self, _event=None):
        sel = self.tree.selection()
        if sel:
            self._draw_drilldown(sel[0])

    def _draw_drilldown(self, key: str):
        """Distribution view for one feature: human histogram vs bot per-game values."""
        report = getattr(self, "current_report", None)
        if not report:
            return
        feat = next((f for f in report["features"] if f["key"] == key), None)
        if feat is None:
            return
        label, meaning = FEATURE_META.get(key, (key, ""))
        hvals = (report.get("human_values") or {}).get(key) or []
        bvals = (report.get("bot_values") or {}).get(key) or []
        hmean, hstd = feat.get("human_mean"), feat.get("human_std")
        bval, z = feat.get("bot_value"), feat.get("zscore")
        pval = feat.get("p_value")

        self.ax.clear()
        if hvals:
            self.ax.hist(hvals, bins=30, density=True, color="#4c78a8", alpha=0.55,
                         label=f"human per-game ({len(hvals)})")
        elif hmean is not None and hstd:
            # Old baseline without raw values: show the fitted normal instead.
            import math
            xs = [hmean + hstd * (i / 25 - 4) for i in range(201)]
            ys = [math.exp(-((x - hmean) ** 2) / (2 * hstd ** 2))
                  / (hstd * math.sqrt(2 * math.pi)) for x in xs]
            self.ax.plot(xs, ys, color="#4c78a8",
                         label="human (normal fit; rebuild baseline for histogram)")
        if bvals:
            self.ax.hist(bvals, bins=30, density=True, color="#ff7f0e", alpha=0.55,
                         label=f"bot per-game ({len(bvals)})")
        if hmean is not None:
            self.ax.axvline(hmean, color="#4c78a8", lw=1.4, ls="--", label="human mean")
            if hstd:
                for k in (-2, 2):
                    self.ax.axvline(hmean + k * hstd, color="#d62728", lw=0.8,
                                    ls=":", alpha=0.7)
        if bval is not None:
            self.ax.axvline(bval, color="#ff7f0e", lw=1.8, label="bot mean")
        ztxt = "n/a" if z is None else f"{z:+.2f}"
        ptxt = "" if pval is None else f", p = {pval:.4f}"
        self.ax.set_title(f"{label}   (z = {ztxt}{ptxt})", fontsize=10)
        self.ax.set_xlabel(meaning, fontsize=8)
        self.ax.set_ylabel("density")
        self.ax.legend(fontsize=7)
        self.fig.tight_layout()
        self.canvas.draw()

    # ---------------------------------------------------------------- io
    def _open_report(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        import json
        try:
            with open(path, encoding="utf-8") as fh:
                report = json.load(fh)
            self._render(report)
            self._log(f"Loaded {os.path.basename(path)}")
        except (OSError, ValueError, KeyError) as e:
            messagebox.showerror("Cannot open", str(e))

    def _save_report(self):
        if not self.last_result:
            messagebox.showinfo("Nothing to save", "Run a diagnostic first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".md",
                                            filetypes=[("Markdown", "*.md")])
        if not path:
            return
        save_result(self.last_result, path, os.path.splitext(path)[0] + ".json")
        self._log(f"Saved report to {path}")


def main():
    root = tk.Tk()
    DiagnosticGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
