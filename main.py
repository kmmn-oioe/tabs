"""tabs - 音声/YouTubeからギターtab譜を生成する Windows GUI アプリ"""
from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import filedialog, messagebox, ttk

from tabgen.pipeline import generate
from tabgen.player import AudioPlayer
from tabgen.playhead import PlaybackTiming, TabPlayhead
from tabgen.tab import STANDARD_TUNING, TabOptions

TUNINGS = {
    "レギュラー (E A D G B e)": list(STANDARD_TUNING),
    "ドロップD (D A D G B e)": [64, 59, 55, 50, 45, 38],
    "半音下 (Eb): Eb Ab Db Gb Bb eb": [63, 58, 54, 49, 44, 39],
    "Open G (D G D G B d)": [62, 55, 50, 43, 47, 38],
}


DIFFICULTY_LABELS = {"初級（最大3音）": "easy", "標準（最大4音）": "standard", "上級（最大6音）": "advanced"}
RHYTHM_MODE_LABELS = {"コード伴奏（保持）": "chords", "検出した発音を保持": "notes"}
PLAYBACK_LABEL = "ギターオフ伴奏＋TABシンセ"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎸 tabs - ギターtab譜生成")
        self.geometry("960x780")
        self.minsize(720, 560)
        self.player = AudioPlayer()
        self.audio_path = ""
        self.midi_path = ""
        self.tab_text = ""
        self._tab_audio_paths = set()
        self._playback_sources = {}
        self._tab_timing = None
        self._build_ui()
        self._poll_job = self.after(50, self._poll_player)

    # ---------- UI構築 ----------

    def _build_ui(self):
        self._build_menu()

        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=4)

        ttk.Label(top, text="音声ファイル または YouTube URL:").pack(anchor="w")
        row = ttk.Frame(top)
        row.pack(fill="x")
        self.path_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="参照…", command=self._browse).pack(side="left", padx=4)

        opts = ttk.LabelFrame(self, text="オプション")
        opts.pack(fill="x", padx=8, pady=4)

        ttk.Label(opts, text="チューニング").grid(row=0, column=0, sticky="w", padx=4)
        self.tuning_var = tk.StringVar(value=list(TUNINGS)[0])
        ttk.Combobox(opts, textvariable=self.tuning_var, values=list(TUNINGS),
                     state="readonly", width=34).grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(opts, text="テンポ (BPM)").grid(row=0, column=2, sticky="w", padx=4)
        self.bpm_var = tk.IntVar(value=120)
        self.bpm_spin = ttk.Spinbox(opts, from_=40, to=240, textvariable=self.bpm_var, width=6)
        self.bpm_spin.grid(row=0, column=3)
        self.auto_bpm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="自動推定", variable=self.auto_bpm_var,
                        command=self._toggle_auto_bpm).grid(row=0, column=4, sticky="w")

        ttk.Label(opts, text="最小ノート信頼度").grid(row=1, column=0, sticky="w", padx=4)
        self.conf_var = tk.DoubleVar(value=0.3)
        ttk.Scale(opts, from_=0.05, to=0.8, variable=self.conf_var,
                  orient="horizontal", length=150).grid(row=1, column=1, sticky="w")
        self.capo_var = tk.IntVar(value=0)
        ttk.Label(opts, text="カポ(フレット)").grid(row=1, column=2, sticky="w")
        ttk.Spinbox(opts, from_=0, to=7, textvariable=self.capo_var, width=4).grid(row=1, column=3)

        self.sep_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="採譜に分離後のギター音源を使う (Demucs)",
                        variable=self.sep_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=4)

        self.split_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="リード / リズムに分けて生成", variable=self.split_var,
                        command=self._toggle_split).grid(row=2, column=2, columnspan=2, sticky="w", padx=4)

        ttk.Label(opts, text="リード音域の目安（補助）").grid(row=3, column=0, sticky="w", padx=4)
        self.lead_thr_var = tk.IntVar(value=64)
        self.lead_thr_scale = ttk.Scale(opts, from_=55, to=76, variable=self.lead_thr_var,
                                       orient="horizontal", length=150)
        self.lead_thr_lbl = ttk.Label(opts, text="E4 (64)")
        self.lead_thr_scale.grid(row=3, column=1, sticky="w")
        self.lead_thr_lbl.grid(row=3, column=2, sticky="w")
        self.lead_thr_scale.state(["disabled"])
        self.lead_thr_lbl.state(["disabled"])
        self.lead_thr_scale.bind("<Motion>", self._lead_thr_changed)

        ttk.Label(opts, text="最小ノート長 (ms)").grid(row=4, column=0, sticky="w", padx=4)
        self.note_len_var = tk.IntVar(value=70)
        ttk.Spinbox(opts, from_=30, to=300, increment=10, textvariable=self.note_len_var,
                    width=6).grid(row=4, column=1, sticky="w")
        ttk.Label(opts, text="リズム出力").grid(row=4, column=2, sticky="w")
        self.rhythm_mode_var = tk.StringVar(value="コード伴奏（保持）")
        self.rhythm_mode_box = ttk.Combobox(opts, textvariable=self.rhythm_mode_var,
                                           values=list(RHYTHM_MODE_LABELS), width=22, state="disabled")
        self.rhythm_mode_box.grid(row=4, column=3, sticky="w", padx=4)

        ttk.Label(opts, text="演奏難易度").grid(row=5, column=0, sticky="w", padx=4)
        self.difficulty_var = tk.StringVar(value="標準（最大4音）")
        ttk.Combobox(opts, textvariable=self.difficulty_var, values=list(DIFFICULTY_LABELS),
                     state="readonly", width=24).grid(row=5, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(opts, text="コード安定性（拍）").grid(row=6, column=0, sticky="w", padx=4)
        self.chord_beats_var = tk.IntVar(value=2)
        self.chord_beats_box = ttk.Combobox(opts, textvariable=self.chord_beats_var,
                                          values=(1, 2, 4), state="disabled", width=6)
        self.chord_beats_box.grid(row=6, column=1, sticky="w", padx=4, pady=4)

        self.suppress_chords_var = tk.BooleanVar(value=True)
        self.suppress_chords_check = ttk.Checkbutton(opts, text="コード打音をリードから除外",
                                                    variable=self.suppress_chords_var, state="disabled")
        self.suppress_chords_check.grid(row=6, column=2, columnspan=3, sticky="w", padx=4)
        ttk.Label(opts, text="リード開始（秒・0=自動）").grid(row=7, column=0, sticky="w", padx=4)
        self.lead_start_var = tk.DoubleVar(value=0.)
        self.lead_start_box = ttk.Spinbox(opts, textvariable=self.lead_start_var,
                                         from_=0, to=86400, increment=.5, width=10, state="disabled")
        self.lead_start_box.grid(row=7, column=1, sticky="w", padx=4, pady=4)
        self.audio_lead_var = tk.BooleanVar(value=True)
        self.audio_lead_check = ttk.Checkbutton(opts, text="音声で独立したリードを確認（実験的）",
                                               variable=self.audio_lead_var, state="disabled",
                                               command=self._toggle_split)
        self.audio_lead_check.grid(row=7, column=2, columnspan=3, sticky="w", padx=4)

        btns = ttk.Frame(self)
        btns.pack(fill="x", padx=8)
        self.run_btn = ttk.Button(btns, text="▶ タブ譜を生成", command=self._run)
        self.run_btn.pack(side="left", pady=6)
        ttk.Button(btns, text=" MIDIを保存", command=self._save_midi).pack(side="left", padx=6)
        ttk.Button(btns, text="♪ 楽譜を保存", command=self._save_sheet).pack(side="left", padx=6)
        ttk.Button(btns, text="📄 PDFで保存", command=self._save_pdf).pack(side="left", padx=6)
        ttk.Button(btns, text="クリップボードにコピー", command=self._copy_tab).pack(side="left")
        self._build_player()

    def _toggle_auto_bpm(self):
        self.bpm_spin.state(["disabled"] if self.auto_bpm_var.get() else ["!disabled"])

    def _toggle_split(self):
        self.rhythm_mode_box.configure(state="readonly" if self.split_var.get() else "disabled")
        self.chord_beats_box.configure(state="readonly" if self.split_var.get() else "disabled")
        self.lead_start_box.configure(state="normal" if self.split_var.get() else "disabled")
        self.suppress_chords_check.configure(state="normal" if self.split_var.get() and not self.audio_lead_var.get()
                                             else "disabled")
        self.audio_lead_check.configure(state="normal" if self.split_var.get() else "disabled")
        if self.split_var.get():
            self.lead_thr_scale.state(["!disabled"])
            self.lead_thr_lbl.state(["!disabled"])
            self._lead_thr_changed()
        else:
            self.lead_thr_scale.state(["disabled"])
            self.lead_thr_lbl.state(["disabled"])

    def _lead_thr_changed(self, *_):
        v = int(self.lead_thr_var.get())
        names = {55: "G3", 57: "A3", 59: "B3", 60: "C4", 62: "D4", 64: "E4", 65: "F4", 67: "G4", 69: "A4", 71: "B4", 72: "C5", 74: "D5", 76: "E5"}
        self.lead_thr_lbl.config(text=f"{names.get(v, 'MIDI ' + str(v))} ({v})")

    def _build_player(self):
        # ---- プレイヤー ----
        pl = ttk.LabelFrame(self, text="プレイヤー")
        pl.pack(fill="x", padx=8, pady=4)

        self.play_btn = ttk.Button(pl, text="⏵", width=3, command=self._toggle_play)
        self.play_btn.grid(row=0, column=0, padx=4)
        ttk.Button(pl, text="⏹", width=3, command=self._stop_play).grid(row=0, column=1)

        self.pos_var = tk.DoubleVar(value=0.0)
        self.seeking = False
        self.slider = ttk.Scale(pl, from_=0, to=100, variable=self.pos_var,
                                orient="horizontal", command=self._on_seek)
        self.slider.grid(row=0, column=2, sticky="ew", padx=6)
        pl.columnconfigure(2, weight=1)

        self.time_label = ttk.Label(pl, text="0:00 / 0:00", width=12)
        self.time_label.grid(row=0, column=3, padx=4)

        sources = ttk.Frame(pl)
        sources.grid(row=1, column=0, columnspan=4, sticky="ew", padx=4, pady=4)
        ttk.Label(sources, text="再生音源").pack(side="left")
        self.audio_source_var = tk.StringVar(value=PLAYBACK_LABEL)
        self.audio_source_box = ttk.Combobox(sources, textvariable=self.audio_source_var,
                                            values=(PLAYBACK_LABEL,), width=28, state="disabled")
        self.audio_source_box.pack(side="left", padx=6)
        self.audio_source_box.bind("<<ComboboxSelected>>", self._switch_audio_source)
        self.audio_file_var = tk.StringVar()
        ttk.Label(sources, textvariable=self.audio_file_var).pack(side="left", padx=4)

        prow = ttk.Frame(self)
        prow.pack(fill="x", padx=10, pady=(4, 0))
        self.progress = ttk.Progressbar(prow, maximum=100, value=0)
        self.progress.pack(side="left", fill="x", expand=True)
        self.pct_label = ttk.Label(prow, text="", width=6, anchor="e")
        self.pct_label.pack(side="left", padx=(6, 0))
        self.status = ttk.Label(self, text="", padding=(10, 2))
        self.status.pack(anchor="w")

        # ---- タブ譜エリア ----
        out = ttk.LabelFrame(self, text="タブ譜 (編集可能)")
        out.pack(fill="both", expand=True, padx=8, pady=4)

        font_bar = ttk.Frame(out)
        font_bar.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(font_bar, text="フォント").pack(side="left", padx=4)
        self.font_size = tk.IntVar(value=10)
        ttk.Spinbox(font_bar, from_=6, to=24, textvariable=self.font_size,
                    width=4, command=self._apply_font).pack(side="left")
        self.follow_cursor = tk.BooleanVar(value=True)
        ttk.Checkbutton(font_bar, text="再生位置を追従", variable=self.follow_cursor).pack(side="left", padx=8)
        ttk.Label(font_bar, text="追従パート").pack(side="left")
        self.follow_part = tk.StringVar(value="タブ譜")
        self.part_box = ttk.Combobox(font_bar, textvariable=self.follow_part, width=14, state="disabled")
        self.part_box.pack(side="left", padx=4)
        self.part_box.bind("<<ComboboxSelected>>", lambda event: self.playhead.select_part(self.follow_part.get()))

        self._tab_font = ("Consolas" if "Consolas" in tkfont.families(self)
                          else tkfont.nametofont("TkFixedFont").actual("family"))
        self.text = tk.Text(out, font=(self._tab_font, 10), wrap="none", undo=True)
        self.playhead = TabPlayhead(self.text, self._playhead_parts)
        hx = ttk.Scrollbar(out, orient="horizontal", command=self.text.xview)
        vy = ttk.Scrollbar(out, orient="vertical", command=self.text.yview)
        self.text.configure(xscrollcommand=lambda *args: (hx.set(*args), self.playhead.schedule_draw()),
                            yscrollcommand=lambda *args: (vy.set(*args), self.playhead.schedule_draw()))
        self.text.grid(row=1, column=0, sticky="nsew")
        vy.grid(row=1, column=1, sticky="ns")
        hx.grid(row=2, column=0, sticky="ew")
        out.rowconfigure(1, weight=1)
        out.columnconfigure(0, weight=1)

    def _build_menu(self):
        m = tk.Menu(self)
        filem = tk.Menu(m, tearoff=0)
        filem.add_command(label="音声を開く…", command=self._browse, accelerator="Ctrl+O")
        filem.add_command(label="タブを保存 (.txt)…", command=self._save_tab, accelerator="Ctrl+S")
        filem.add_command(label="MIDIを保存…", command=self._save_midi)
        filem.add_command(label="楽譜を保存 (.musicxml)…", command=self._save_sheet)
        filem.add_command(label="コード付きTAB譜をPDFで保存…", command=self._save_pdf)
        filem.add_separator()
        filem.add_command(label="終了", command=self.destroy)
        m.add_cascade(label="ファイル", menu=filem)

        editm = tk.Menu(m, tearoff=0)
        editm.add_command(label="すべて選択", command=lambda: self.text.tag_add("sel", "1.0", "end"))
        editm.add_command(label="コピー", command=self._copy_tab, accelerator="Ctrl+C")
        m.add_cascade(label="編集", menu=editm)
        self.config(menu=m)
        self.bind("<Control-o>", lambda e: self._browse())
        self.bind("<Control-s>", lambda e: self._save_tab())

    # ---------- 解析 ----------

    def _browse(self):
        f = filedialog.askopenfilename(filetypes=[
            ("音声ファイル", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac *.webm"),
            ("すべてのファイル", "*.*"),
        ])
        if f:
            self.path_var.set(f)
            self._set_playback_sources({})
            self.status.config(text="TABを生成すると、ギターオフ伴奏＋TABシンセで再生できます")

    def _run(self):
        src = self.path_var.get().strip()
        if not src:
            messagebox.showwarning("tabs", "音声ファイルまたはYouTube URLを入力してください")
            return
        self._stop_play()
        self.run_btn.config(state="disabled")
        self.progress.config(value=0)
        self.status.config(text="処理中…")
        threading.Thread(target=self._worker, args=(src,), daemon=True).start()

    def _worker(self, src):
        try:
            audio_sources = {}
            opts = TabOptions(
                tuning=TUNINGS[self.tuning_var.get()],
                bpm=float(self.bpm_var.get()),
                min_confidence=float(self.conf_var.get()),
                capo=int(self.capo_var.get()),
                difficulty=DIFFICULTY_LABELS[self.difficulty_var.get()],
                chord_beats=int(self.chord_beats_var.get()),
                suppress_chord_attacks=self.suppress_chords_var.get(),
                lead_start_seconds=float(self.lead_start_var.get()),
                audio_lead_detection=self.audio_lead_var.get(),
                rhythm_mode=RHYTHM_MODE_LABELS[self.rhythm_mode_var.get()],
            )
            tab_text, midi_path, sheet_paths, stem = generate(
                src, opts, use_separation=self.sep_var.get(),
                split_lead_rhythm=self.split_var.get(),
                lead_threshold=int(self.lead_thr_var.get()),
                min_note_length_ms=float(self.note_len_var.get()),
                auto_bpm=self.auto_bpm_var.get(),
                progress=lambda pct, msg: self.after(0, lambda: self._set_progress(pct, msg)),
                audio_sources=audio_sources,
                prepare_playback=True,
            )
            self.after(0, lambda: self._done(tab_text, midi_path, sheet_paths, stem, opts, audio_sources))
        except Exception as e:
            import traceback
            msg = str(e)
            tb = traceback.format_exc()
            self.after(0, lambda: self._fail(msg, tb))

    def _set_progress(self, pct, msg):
        if pct is not None:
            self.progress.config(value=max(0, min(100, pct)))
            self.pct_label.config(text=f"{min(100, max(0, int(pct)))}%")
        if msg:
            self.status.config(text=msg)

    def _done(self, tab_text, midi_path, sheet_paths, stem, opts, audio_sources=None):
        self.player.stop()
        self.progress.config(value=100)
        self.tab_text = tab_text
        self.midi_path = midi_path
        self.sheet_paths = [p for p in sheet_paths.split(";") if p]
        self.text.delete("1.0", "end")
        self.text.insert("1.0", tab_text)
        audio_sources = audio_sources or {}
        playback = audio_sources.get('playback')
        sources = {PLAYBACK_LABEL: playback} if playback else {}
        self._tab_audio_paths = {self._audio_key(p) for p in sources.values()}
        # Keep the actual generation tempo/offset, including automatic estimates;
        # later edits to the options form must not change the playback clock.
        self._tab_timing = PlaybackTiming.from_options(opts)
        self.run_btn.config(state="normal")
        self._set_playback_sources(sources)
        if not playback:
            self.status.config(text=audio_sources.get('playback_error', '再生音源がありません。TABを再生成してください。'))

    def _fail(self, msg, tb=""):
        self.run_btn.config(state="normal")
        self.progress.config(value=0)
        self.pct_label.config(text="0%")
        self.status.config(text="エラー")
        if tb:
            # 詳細をテキスト領域に表示(コピーして報告できます)
            self.text.delete("1.0", "end")
            self.text.insert("1.0", tb)
        messagebox.showerror("tabs", msg)

    # ---------- 保存/コピー ----------

    def _current_text(self) -> str:
        return self.text.get("1.0", "end-1c")

    def _save_tab(self):
        txt = self._current_text()
        if not txt.strip():
            messagebox.showinfo("tabs", "まずタブ譜を生成してください")
            return
        f = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("テキスト", "*.txt")],
            initialfile="tab.txt")
        if f:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(txt)
            self.status.config(text=f"保存しました: {f}")

    def _save_midi(self):
        paths = [p for p in self.midi_path.split(";") if p and os.path.exists(p)]
        if not paths:
            messagebox.showinfo("tabs", "MIDIデータがありません。まずタブ譜を生成してください")
            return
        for p in paths:
            if len(paths) > 1:
                kind = "リード" if "lead" in os.path.basename(p) else "リズム"
                if not messagebox.askyesno("tabs", f"{kind}ギターのMIDIを保存しますか?"):
                    continue
            f = filedialog.asksaveasfilename(
                defaultextension=".mid", filetypes=[("MIDI", "*.mid")],
                initialfile=("lead.mid" if "lead" in os.path.basename(p) else "rhythm.mid")
                            if len(paths) > 1 else "tab.mid")
            if f:
                with open(p, "rb") as src, open(f, "wb") as dst:
                    dst.write(src.read())
                self.status.config(text=f"MIDIを保存しました: {f}")

    def _save_sheet(self):
        paths = [p for p in self.sheet_paths if os.path.exists(p)]
        if not paths:
            messagebox.showinfo("tabs", "楽譜データがありません。まずタブ譜を生成してください")
            return
        for p in paths:
            if len(paths) > 1:
                kind = "リード" if "lead" in os.path.basename(p) else "リズム"
                if not messagebox.askyesno("tabs", f"{kind}ギターの楽譜を保存しますか?"):
                    continue
            f = filedialog.asksaveasfilename(
                defaultextension=".musicxml", filetypes=[("MusicXML", "*.musicxml")],
                initialfile=("lead.musicxml" if "lead" in os.path.basename(p) else "rhythm.musicxml")
                            if len(paths) > 1 else "sheet.musicxml")
            if f:
                with open(p, "rb") as src, open(f, "wb") as dst:
                    dst.write(src.read())
                self.status.config(text=f"楽譜を保存しました: {f}")

    def _save_pdf(self):
        paths = [p for p in self.sheet_paths if os.path.exists(p)]
        if not paths:
            messagebox.showinfo("tabs", "楽譜データがありません。まずタブ譜を生成してください")
            return
        from tabgen.tabpdf import tab_to_pdf
        for p in paths:
            if len(paths) > 1:
                kind = "リード" if "lead" in os.path.basename(p) else "リズム"
                if not messagebox.askyesno("tabs", f"{kind}ギターのコード付きTAB譜をPDFで保存しますか?"):
                    continue
            default = ("lead.pdf" if "lead" in os.path.basename(p) else "rhythm.pdf") \
                if len(paths) > 1 else "sheet.pdf"
            f = filedialog.asksaveasfilename(
                defaultextension=".pdf", filetypes=[("PDF", "*.pdf")], initialfile=default)
            if not f:
                continue
            try:
                self.status.config(text="PDF変換中…")
                self.update()
                tab_to_pdf(p, f)
                self.status.config(text=f"PDFを保存しました: {f}")
            except Exception as e:
                self.status.config(text="PDF保存に失敗しました")
                messagebox.showerror("tabs", f"PDFを保存できませんでした。\n{type(e).__name__}: {e}")
                return

    def _copy_tab(self):
        txt = self._current_text()
        if not txt.strip():
            return
        self.clipboard_clear()
        self.clipboard_append(txt)
        self.status.config(text="クリップボードにコピーしました")

    # ---------- プレイヤー ----------

    @staticmethod
    def _audio_key(path):
        return os.path.normcase(os.path.realpath(path))

    def _set_playback_sources(self, sources):
        self.player.stop()
        self._playback_sources = dict(sources)
        labels = list(sources)
        selected = labels[0] if labels else ""
        self.audio_source_box.configure(values=labels, state="readonly" if len(labels)>1 else "disabled")
        self.audio_source_var.set(selected)
        if selected:
            self._load_for_playback(sources[selected])
        else:
            self.player.data = None
            self.player.seek(0)
            self.audio_path = ""
            self.audio_file_var.set("")
            self.slider.config(to=0)
            self.playhead.set_timing(None)
            self._refresh_playback()

    def _switch_audio_source(self, event=None):
        path = self._playback_sources.get(self.audio_source_var.get())
        if path is None:
            return
        if self.audio_path and self._audio_key(path) == self._audio_key(self.audio_path):
            return
        position, resume = self.player.position, self.player.playing
        self.player.stop()  # Abort the old source before loading/starting another.
        if self._load_for_playback(path):
            self.player.seek(position)
            if resume and self.player.position < self.player.duration:
                if not self.player.play():
                    self.status.config(text=f"再生できません: {self.player.last_error}")
        self._refresh_playback()

    def _load_for_playback(self, path):
        ok = self.player.load(path)
        if ok:
            self.audio_path = path
            self.audio_file_var.set(os.path.basename(path))
            self.slider.config(to=self.player.duration)
            linked = self._audio_key(path) in self._tab_audio_paths
            self.playhead.set_timing(self._tab_timing if linked else None)
            self.status.config(text=f"再生対象: {self.audio_source_var.get()}")
        else:
            self.audio_path = ""
            self.audio_file_var.set("")
            self.slider.config(to=0)
            self.playhead.set_timing(None)
            self.status.config(text=f"音声を読み込めません: {self.player.last_error}")
        self._refresh_playback()
        return ok

    def _toggle_play(self):
        if self.player.data is None:
            return
        if self.player.playing:
            self.player.pause()
            self.play_btn.config(text="⏵")
        else:
            if self.player.position >= self.player.duration - 0.05:
                self.player.seek(0)
            if not self.player.play():
                self.status.config(text=f"再生できません: {self.player.last_error}")
        self._refresh_playback()

    def _stop_play(self):
        self.player.stop()
        self.player.seek(0)
        self._refresh_playback()

    def _on_seek(self, _value):
        if self.seeking:
            return
        self.player.seek(self.pos_var.get())
        self._refresh_playback()

    def _poll_player(self):
        self._refresh_playback()
        self._poll_job = self.after(50, self._poll_player)

    def _refresh_playback(self):
        pos, dur = self.player.position, self.player.duration
        self.seeking = True
        try:
            self.pos_var.set(pos)
        finally:
            self.seeking = False
        self.time_label.config(text=f"{self._fmt(pos)} / {self._fmt(dur)}")
        self.play_btn.config(text="⏸" if self.player.playing else "⏵")
        cursor = min(pos, max(0., dur-1e-6)) if dur else pos
        self.playhead.update(cursor, follow=self.follow_cursor.get())

    def _playhead_parts(self, parts, selected):
        self.part_box.configure(values=parts, state="readonly" if len(parts)>1 else "disabled")
        self.follow_part.set(selected)

    @staticmethod
    def _fmt(sec: float) -> str:
        sec = max(0, int(sec))
        return f"{sec // 60}:{sec % 60:02d}"

    # ---------- 表示 ----------

    def _apply_font(self):
        self.text.config(font=(self._tab_font, self.font_size.get()))
        self.playhead.schedule_draw()

    def destroy(self):
        if getattr(self, "_poll_job", None):
            self.after_cancel(self._poll_job)
            self._poll_job = None
        if hasattr(self, "playhead"):
            self.playhead.close()
        if hasattr(self, "player"):
            self.player.stop()
        super().destroy()


if __name__ == "__main__":
    App().mainloop()
