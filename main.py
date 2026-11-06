"""tabs - 音声/YouTubeからギターtab譜を生成する Windows GUI アプリ"""
from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from tabgen.pipeline import generate
from tabgen.tab import STANDARD_TUNING, TabOptions

TUNINGS = {
    "レギュラー (E A D G B e)": list(STANDARD_TUNING),
    "ドロップD (D A D G B e)": [64, 59, 55, 50, 45, 38],
    "半音下 (Eb): Eb Ab Db Gb Bb eb": [63, 58, 54, 49, 44, 39],
    "Open G (D G D G B d)": [62, 55, 50, 43, 47, 38],
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎸 tabs - ギターtab譜生成")
        self.geometry("900x700")
        self.minsize(700, 500)
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="音声ファイル または YouTube URL:").pack(anchor="w")
        row = ttk.Frame(top)
        row.pack(fill="x")
        self.path_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="参照…", command=self._browse).pack(side="left", padx=4)

        opts = ttk.LabelFrame(self, text="オプション")
        opts.pack(fill="x", **pad)

        ttk.Label(opts, text="チューニング").grid(row=0, column=0, sticky="w", padx=4)
        self.tuning_var = tk.StringVar(value=list(TUNINGS)[0])
        ttk.Combobox(opts, textvariable=self.tuning_var, values=list(TUNINGS),
                     state="readonly", width=34).grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(opts, text="テンポ (BPM)").grid(row=0, column=2, sticky="w", padx=4)
        self.bpm_var = tk.IntVar(value=120)
        ttk.Spinbox(opts, from_=40, to=240, textvariable=self.bpm_var, width=6).grid(row=0, column=3)

        ttk.Label(opts, text="最小ノート信頼度").grid(row=1, column=0, sticky="w", padx=4)
        self.conf_var = tk.DoubleVar(value=0.3)
        ttk.Scale(opts, from_=0.05, to=0.8, variable=self.conf_var,
                 orient="horizontal", length=150).grid(row=1, column=1, sticky="w")
        self.capo_var = tk.IntVar(value=0)
        ttk.Label(opts, text="カポ(フレット)").grid(row=1, column=2, sticky="w")
        ttk.Spinbox(opts, from_=0, to=7, textvariable=self.capo_var, width=4).grid(row=1, column=3)

        self.sep_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="ギターパートを分離する (Demucs / 要 torch・時間大)",
                        variable=self.sep_var).grid(row=2, column=0, columnspan=4, sticky="w", padx=4)

        self.run_btn = ttk.Button(self, text="タブ譜を生成", command=self._run)
        self.run_btn.pack(pady=6)

        self.progress = ttk.Label(self, text="")
        self.progress.pack(anchor="w", padx=10)

        out = ttk.LabelFrame(self, text="タブ譜")
        out.pack(fill="both", expand=True, **pad)
        self.text = tk.Text(out, font=("Consolas", 10), wrap="none")
        hx = ttk.Scrollbar(out, orient="horizontal", command=self.text.xview)
        vy = ttk.Scrollbar(out, orient="vertical", command=self.text.yview)
        self.text.configure(xscrollcommand=hx.set, yscrollcommand=vy.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        vy.grid(row=0, column=1, sticky="ns")
        hx.grid(row=1, column=0, sticky="ew")
        out.rowconfigure(0, weight=1)
        out.columnconfigure(0, weight=1)

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", **pad)
        ttk.Button(bottom, text="タブを保存 (.txt)", command=self._save_tab).pack(side="left")
        self.midi_path = ""
        self.tab_text = ""

    def _browse(self):
        f = filedialog.askopenfilename(filetypes=[
            ("音声ファイル", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac *.webm"),
            ("すべてのファイル", "*.*"),
        ])
        if f:
            self.path_var.set(f)

    def _run(self):
        src = self.path_var.get().strip()
        if not src:
            messagebox.showwarning("tabs", "音声ファイルまたはYouTube URLを入力してください")
            return
        self.run_btn.config(state="disabled")
        self.progress.config(text="処理中…")
        threading.Thread(target=self._worker, args=(src,), daemon=True).start()

    def _worker(self, src):
        try:
            opts = TabOptions(
                tuning=TUNINGS[self.tuning_var.get()],
                bpm=float(self.bpm_var.get()),
                min_confidence=float(self.conf_var.get()),
                capo=int(self.capo_var.get()),
            )
            tab_text, midi_path, _ = generate(
                src, opts, use_separation=self.sep_var.get(),
                progress=lambda m: self.after(0, lambda: self.progress.config(text=m)),
            )
            self.after(0, lambda: self._done(tab_text, midi_path))
        except Exception as e:
            msg = str(e)
            self.after(0, lambda: self._fail(msg))

    def _done(self, tab_text, midi_path):
        self.tab_text = tab_text
        self.midi_path = midi_path
        self.text.delete("1.0", "end")
        self.text.insert("1.0", tab_text)
        self.run_btn.config(state="normal")

    def _fail(self, msg):
        self.run_btn.config(state="normal")
        self.progress.config(text="エラー")
        messagebox.showerror("tabs", msg)

    def _save_tab(self):
        if not self.tab_text:
            messagebox.showinfo("tabs", "まずタブ譜を生成してください")
            return
        f = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("テキスト", "*.txt")],
            initialfile="tab.txt")
        if f:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(self.tab_text)
            self.progress.config(text=f"保存しました: {f}")


if __name__ == "__main__":
    App().mainloop()
