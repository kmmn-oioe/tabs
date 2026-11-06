<img src="assets/icon.svg" width="64" height="64" alt="">

# 🎸tabs

音声ファイルやYouTubeのリンクから、曲のギターtab譜(タブ譜)を自動生成するWindowsデスクトップアプリです。

## 機能

- 🎵 音声ファイル(wav / mp3 / flac / m4a / ogg など)を読み込んで解析
- 📺 YouTube URL から音声をダウンロードして解析
- 🎚️ (任意)Demucs によるギターパート分離 → 精度アップ
- 🎼 音程検出は Spotify の **basic-pitch**
- 🎸 チューニング選択(レギュラー / ドロップD / 半音下 / Open G)、カポ位置指定
- 💾 タブ譜(.txt)と MIDI(.mid)で保存

## セットアップ(Windows)

Python 3.10 以降をインストールしてから:

```bat
py -3 -m pip install -r requirements.txt
tabs.bat
```

### ギターパート分離を使う場合(推奨・容量大)

```bat
py -3 -m pip install demucs torch
```

※ torch は数GBダウンロードされます。CPUでも動作しますが処理に時間がかかります。

## 使い方

1. `tabs.bat` を起動
2. 音声ファイルを「参照…」で選ぶか、YouTube URL を貼り付け
3. チューニングやBPMを設定して「タブ譜を生成」
4. 結果を確認し「タブを保存 (.txt)」

## exe 化(配布用)

```bat
py -3 -m pip install pyinstaller
pyinstaller --name tabs --noconsole --collect-all basic_pitch ^
  --collect-all pretty_midi --collect-all yt_dlp main.py
```

生成物は `dist/tabs/tabs.exe` です。

## 仕組み

```
音声 / YouTube
   └─ yt-dlp で音声取得(YouTubeの場合)
   └─ Demucs でギターステム分離(任意)
   └─ basic-pitch でノート検出(時刻・音程・信頼度)
   └─ チューニングに割り当ててtab譜テキスト化
```

## 制限

- ポリフォニックな解析(コードの完全な指板配置の最適化)は簡易実装です。同時発音の衝突時は後続ノートを優先します
- アコースティック/クリーンギター向けです。激しいディストーションや高速演奏は精度が落ちます

Created in DeepAI Code.
