"""Tkinter GUI for the human-likeness diagnostic.

Run:  venv/bin/python -m cheat_detection.gui

Enter the bot's Lichess account, rating band and time control, point at a
baseline (or a human corpus to build one), and hit Run. The heavy analysis
runs in a worker thread; progress streams to the log, and the result renders
as a z-score chart plus a feature table. A previously saved report.json can be
opened directly for instant viewing.
"""

from __future__ import annotations

import os
import queue
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

        self._build_inputs()
        self._build_examples()
        self._build_body()
        self.root.after(120, self._drain_queue)

    # ---------------------------------------------------------------- inputs
    def _build_inputs(self):
        f = ttk.LabelFrame(self.root, text="Configuration", padding=8)
        f.pack(fill="x", padx=8, pady=(8, 4))

        self.v_user = tk.StringVar(value="JXu2019")
        self.v_rmin = tk.StringVar(value="2300")
        self.v_rmax = tk.StringVar(value="2600")
        self.v_perf = tk.StringVar(value="bullet")
        self.v_baseline = tk.StringVar(
            value=os.path.join(_HERE, "baselines", "bullet_1plus0_2300_2600.json"))
        self.v_corpus = tk.StringVar(
            value=os.path.join(_HERE, "corpora", "bullet_1plus0_2300_2600.pgn"))
        self.v_botpgn = tk.StringVar(value="")
        self.v_botmax = tk.StringVar(value="300")
        self.v_basemax = tk.StringVar(value="250")
        self.v_depth = tk.StringVar(value="18")
        self.v_multipv = tk.StringVar(value="5")

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
        ttk.Combobox(f, textvariable=self.v_perf, width=12, state="readonly",
                     values=["bullet", "blitz", "rapid", "classical"]).grid(
            row=2, column=1, sticky="w", padx=4)

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
                            ("MultiPV", self.v_multipv, 5)]:
            ttk.Label(adv, text=lbl).pack(side="left", padx=(8, 2))
            ttk.Entry(adv, textvariable=var, width=w).pack(side="left")

        btns = ttk.Frame(f)
        btns.grid(row=7, column=0, columnspan=3, sticky="w", pady=(6, 0))
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
            self.root,
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
        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        self.verdict = ttk.Label(left, text="", font=("TkDefaultFont", 11, "bold"))
        self.verdict.pack(anchor="w")
        self.fig = Figure(figsize=(5.5, 5.2), dpi=96)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_empty()

        right = ttk.Frame(body)
        right.pack(side="right", fill="both", expand=True, padx=(8, 0))
        cols = ("feature", "human", "bot", "z")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=16)
        for c, txt, w in [("feature", "Feature", 210), ("human", "Human m±sd", 120),
                          ("bot", "Bot", 80), ("z", "z", 60)]:
            self.tree.heading(c, text=txt)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True)

        logf = ttk.LabelFrame(self.root, text="Log", padding=4)
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

    # ------------------------------------------------------------- run/thread
    def _log(self, msg): self.q.put(("log", msg))

    def _cfg(self) -> AnalysisConfig:
        cfg = AnalysisConfig()
        cfg.depth = int(self.v_depth.get())
        cfg.multipv = int(self.v_multipv.get())
        return cfg

    def _on_run(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "A diagnostic is already running.")
            return
        try:
            spec = DiagnosticSpec(
                username=self.v_user.get().strip(),
                rating_band=(int(self.v_rmin.get()), int(self.v_rmax.get())),
                perf=self.v_perf.get(),
                bot_pgn=self.v_botpgn.get().strip() or None,
                baseline_path=self.v_baseline.get().strip(),
                corpus_pgn=self.v_corpus.get().strip() or None,
                bot_max_games=int(self.v_botmax.get()),
                baseline_max_games=int(self.v_basemax.get()) if self.v_basemax.get() else None,
                fetch_max_games=int(self.v_botmax.get()),
            )
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
        except queue.Empty:
            pass
        self.root.after(120, self._drain_queue)

    # -------------------------------------------------------------- rendering
    def _render(self, report: dict):
        feats = report["features"]
        scored = [f for f in feats if f.get("zscore") is not None]
        mean_abs_z = sum(abs(f["zscore"]) for f in scored) / len(scored) if scored else 0.0
        verdict = ("within normal human variation" if mean_abs_z < 1
                   else "noticeably distinguishable" if mean_abs_z < 2
                   else "strongly distinguishable")
        n_flag = sum(1 for f in feats if f.get("flagged"))
        self.verdict.configure(
            text=f"Overall divergence: mean |z| = {mean_abs_z:.2f}  ({verdict}); "
                 f"{n_flag} feature(s) beyond 2σ")

        # chart: horizontal bars of z, sorted by |z|
        rows = sorted(scored, key=lambda f: abs(f["zscore"]))
        labels = [FEATURE_META.get(f["key"], (f["key"],))[0] for f in rows]
        zs = [f["zscore"] for f in rows]
        colors = ["#d62728" if abs(z) >= 2 else "#ff7f0e" if abs(z) >= 1 else "#4c78a8"
                  for z in zs]
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

        # table
        self.tree.delete(*self.tree.get_children())
        order = sorted(feats, key=lambda f: -(abs(f["zscore"]) if f.get("zscore") is not None else -1))
        for f in order:
            label = FEATURE_META.get(f["key"], (f["key"],))[0]
            hm, hs = f.get("human_mean"), f.get("human_std")
            human = f"{hm:.3f}±{hs:.3f}" if hm is not None and hs is not None else "n/a"
            bot = "n/a" if f.get("bot_value") is None else f"{f['bot_value']:.3f}"
            z = "n/a" if f.get("zscore") is None else f"{f['zscore']:.2f}"
            tag = "flag" if f.get("flagged") else ""
            self.tree.insert("", "end", values=(label, human, bot, z), tags=(tag,))
        self.tree.tag_configure("flag", background="#ffe0e0")

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
