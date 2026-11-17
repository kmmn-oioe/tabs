# Codex CLI 引き継ぎ（2026-11-17）

## 最初に読むこと

- リポジトリ: https://github.com/kmmn-oioe/tabs
- 作業ブランチ: `improve-playable-tabs`。`main` ではない。
- 最後のアプリ変更: `3030f83`（周波数別のリード／リズム判定）。この文書を加えるコミットはその後に続く。
- ユーザーはGitHubと開発に不慣れ。日本語で、実行した結果と次の具体的な操作を簡潔に説明する。
- 最新コードの取得・テストを先に行い、過去の変更を再適用しない。クラウド環境の公開はしない。
- 過去の希望: 検証済み変更はGitHubの作業ブランチへ反映する。手元PCのGitHub認証やGit設定は別途必要。

## 目標

音声／YouTubeから、弾けるギターTABを自動生成するWindowsアプリ。
最終的にはWebサービスのように使いやすく、リード・リズムを正しく分けた自動採譜を目指す。
現在はPython/Tkinter。Webアプリへの移植はしていない。

## 実装済み

- Basic Pitchで音程検出。左右平均で消える音を補う左右差候補も使用。
- 難易度・同時発音・弦／フレット・運指の移動を考慮した整形。
- リードとリズムの分割、音声による独立声部の確認、反復・定位・倍音・位相による補完。
- 音声クロマによるコード推定。リズムは `chords`（コード保持）と `notes`（検出発音保持）の2種類。
- リズム候補の基本音と発音確認、検出遅れの拍補正、リード残存音の除外・引き渡し。
- 伴奏の反復と同音の空間モデルを使ったユニゾン復元。
- 同じ周波数に重なるリード基本音／伴奏倍音を、複素空間モデルと旋律で再判定。
- TABと同期する縦バー、一時停止・シーク。生成時のBPM・原点を固定して同期。
- 再生は **`no_guitar.wav` ＋最終TABの内蔵シンセ** の1本のみ。原曲／guitar.wavを別に同時再生しない。
- TAB・MIDI・MusicXML・コード名付きTAB PDF・シンセは同じ整形後ノートを使用。

## 分離とパート判定の違い

`tabgen/separate.py` は通常 `htdemucs_6s --two-stems guitar` を実行し、`guitar.wav` と
`no_guitar.wav` を使う。内部モデルはドラム／ベース／ボーカル／その他／ピアノ／ギターの6ステム。
失敗すると4ステムの `other.wav` へ代替するため、実際の使用モデルを確認する。
`other.wav` をギター専用分離の成功として採点しない。ギターオフ再生には実際の `no_guitar.wav` が必要。

リード／リズムの専用音源2本を出力する機能はまだない。今まで改善したのはギター音源内のTABパート判定。
「採譜に分離後のギター音源を使う」がOFFなら採譜には原曲を使う。
再生用のギターオフ伴奏は、この設定がOFFでも分離を実行して作る。

## 現在の採点値

「運命に賭けたい論理」の先頭10.3秒を解析し、TAB動画の第2〜6小節だけ採点。
参照: `benchmarks/cases/unmei_intro.json`。リード40音・リズム45音。
BPM147、原点0.047346939秒、標準、信頼度0.3、カポ1、通常チューニング、`notes`、発音±80ms。
正解BPMを与えた診断で、自動BPMや全曲の精度と混同しない。

| 条件 | リード一致／40 | リード対応なし | リズム一致／45 | リズム対応なし |
|---|---:|---:|---:|---:|
| 固定した6ステムguitar | 37 | 4 | 41 | 64 |
| 分離なし | 29 | 11 | 29 | 108 |

最後に確認した通常テストは **185件合格**。今後追加すれば件数は変わる。
別曲「羽が生えた理由」の `guitar_intro.wav`（20秒）について、保存した451音・613音の両候補で
生成パイプラインを通し、11秒より前の最終リードが0音であることを確認済み。

最新詳細:

- `benchmarks/results/unmei_frequency_role_refinement.md` / `.json`
- `benchmarks/results/unmei_unison_recovery.md` / `.json`
- `benchmarks/results/unmei_rhythm_voice_refinement.md` / `.json`
- `benchmarks/README.md`（採点条件とCLI）

この区間は繰り返し開発に使っている。未知曲・全曲での改善は保証しない。

## 未解決問題

1. 参照に対応しないリズム64音。全て誤検出とは限らず、他パート・倍音・残響・参照外の実演奏を区別する必要がある。
2. リード欠落3音（2音は候補段階にない、1音は空間モデルで拮抗）。リズム欠落4音も候補段階にない。
3. ユニゾンは反復・同音の録音例・別定位がある場合のみ。静的コード、モノラル、同じ定位、編成変化に弱い。
4. テンポ変化、揺れ、弦別サステイン、ベンド・スライド等の奏法は十分でない。
5. 分離の漏れ・混入、学習済み楽器分類ではないヒューリスティックの限界。
6. 比較対象の曲・区間・正解付き録音が不足。今の短区間だけへ条件を合わせない。
7. `tabgen/transcribe.py` が共通一時名 `tabgen_mono.wav` を使う。複数の実推論を同じ一時ディレクトリで並列実行しない。

## 主要ファイル

| ファイル | 担当 |
|---|---|
| `main.py` / `tabs.bat` | Windows GUI・起動 |
| `tabgen/pipeline.py` | 取得→分離→採譜→判定→整形→出力 |
| `tabgen/separate.py` / `audio.py` | Demucs・デコード・YouTube |
| `tabgen/transcribe.py` | Basic Pitch・ステレオ候補統合 |
| `lead_audio.py` / `split.py` | 独立声部確認・旋律追跡 |
| `rhythm_audio.py` / `part_roles.py` / `unison.py` | 発音確認・倍音とのパート判定・ユニゾン |
| `tab.py` / `rhythm.py` / `harmony.py` | 運指・リズム出力・コード |
| `player.py` / `playhead.py` / `synth.py` | 単一音声ストリーム・同期表示・TABシンセ |
| `tabpdf.py` / `sheet.py` / `export.py` | TAB PDF・MusicXML・MIDI |
| `evaluate.py` / `benchmarks/audit_rhythm.py` | 採点・候補への追跡 |

表の省略ファイル名は `tabgen/` 配下。

## Windowsでの初回確認

移行ZIPの準備と配置は `docs/WINDOWS_HANDOFF.md` を参照。
Python **3.11** で新しい仮想環境を作る。Linuxの `.venv` はコピーしない。
以下はプロジェクトのルートで実行する。既存 `.venv` がある場合は作り直す前に状態を確認する。

```bat
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

クラウドで確認した構成はPython3.11.17、numpy1.26.4、basic-pitch0.4.0、TensorFlow2.14.0、
Demucs4.0.1、torch/torchaudio2.5.1+cpu。Windowsで同じ機能が動くかは実機確認が必要。
`setuptools<81` は `pkg_resources` エラーの再発防止。numpyは2未満。
Demucs・torchがないとギターオフ再生を準備できない。CPU版の導入例はREADME参照。
GPU版は手元のGPU・ドライバを確認して対応するtorch/torchaudioをそろえる。

起動は `tabs.bat` または `.venv\Scripts\python.exe main.py`。
品質比較には次を設定する:

- 採譜に分離後のギター音源を使う: ON
- リード／リズムに分けて生成: ON
- 音声で独立したリードを確認: ON
- リズム出力: 検出した発音を保持

既定のリズム出力はコード保持なので、比較時に必ず変更する。

## 個別の検証データ（移行ZIPだけに含む）

`local-input/manifest.json` に元ファイル名・サイズ・SHA-256を記録している。

| 相対パス（local-input配下） | 内容 |
|---|---|
| `media/original.mp3` | 採点曲の原曲。参照JSONのSHA-256と一致する添付MP3 |
| `media/reference-tabs.mp4` | 上がリード、下がリズムの参照TAB動画 |
| `media/rhythm-only-intro.wav` | 別曲の20秒。約11秒より前はリードなし |
| `fixed/guitar.wav` | 比較に使った同じ10.3秒のギター分離結果 |
| `fixed/no_guitar.wav` | 上記と同じ試行のギターオフ伴奏 |
| `baseline/separated/` / `baseline/unseparated/` | 最新のdiagnostics、report、TAB |
| `intro/notes.json` / `intro/merged_notes.json` | イントロの保存済み音程候補 |

これらを公開GitHubへ追加しない。`local-input/` はignore済み。
Linuxの絶対パスが過去のレポートに残っていても、手元でそのパスを作らない。
移行後は上記の相対パスを使う。

固定した検出・分離を再利用した回帰確認（Demucs/Basic Pitchの実推論とは区別）:

```bat
.venv\Scripts\python.exe scripts\replay_fixed_benchmark.py
```

通常のモデルも再実行する比較:

```bat
.venv\Scripts\python.exe -m tabgen.evaluate run --reference benchmarks\cases\unmei_intro.json --audio local-input\media\original.mp3 --output benchmark-output\fresh --tempo-mode reference --rhythm-mode notes --separation
```

Demucsは乱数を使うため再分離で数値が変わり得る。固定入力との比較を先に行う。
自動BPMの評価では `--tempo-mode auto` を別に実施する。

## 直近の依頼・次の候補

直近はCLIへ移行するための資料・ソース・検証音源の準備。外部ライブラリの導入依頼はまだない。
調査済みリポジトリは `docs/REPOSITORY_CANDIDATES.md` を参照。まだインストール・比較していない。
次は評価曲の拡充、別環境での分離モデル比較、ギター専用採譜モデルの比較が候補。
既存のtorch2.5環境へ、別研究リポジトリの新しいtorch要求をそのまま上書きしない。

## CLIに最初に渡す依頼文

> AGENTS.md、docs/CODEX_HANDOFF.md、docs/WINDOWS_HANDOFF.mdを読んで引き継いでください。
> git statusとブランチを確認し、既存の変更を保持してください。まずWindowsのPython・GPU・Git環境と
> local-inputの配置を確認し、環境構築・テスト・固定入力での基準確認を進めてください。
> 過去の修正の再適用や、外部モデルの無断導入・本番公開はしないでください。
