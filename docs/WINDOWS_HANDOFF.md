# 手元のWindowsへソースと検証データを用意する

クラウドからPCへ直接書き込むことはしていません。移行ZIPをPCで展開し、付属スクリプトを実行して配置します。
この段階ではアプリやCodex CLIのインストール・起動は自動実行しません。

## 移行ZIPを使う（推奨）

1. このチャットの `tabs-cli-handoff-2026-11-17.zip` をダウンロードする。
2. 右クリック→「すべて展開」で、ドキュメント等の書き込み可能な場所へ展開する。ZIP内で直接実行しない。
3. 展開した `tabs-cli-handoff` 内の `START-HERE.txt` を読み、`prepare-windows.cmd` をダブルクリックする。
4. Git for Windowsがなければ、表示される https://git-scm.com/download/win から導入して再実行する。
5. 完了すると、同じ場所の `tabs` にGitの履歴付きソースと `local-input` の検証データがそろう。

構成:

```text
tabs-cli-handoff/
  START-HERE.txt
  prepare-windows.cmd
  tabs.git.bundle          # GitHubの作業ブランチを移すためのオフライン履歴
  local-input/             # 非公開の音源・動画・固定した検証データ
  PACKAGE-SHA256.txt
  tabs/                    # スクリプト実行後に作成される作業フォルダ
```

スクリプトは既存の `tabs` フォルダを上書きしません。既に存在すれば停止するので、
別の新しい場所へ展開してください。元のプロジェクトを削除・上書きする必要はありません。
GitHubへのpush時の認証は手元PC側で必要ですが、オフラインbundleからの配置には不要です。

Linux用の `.venv`、APIキー、Git認証情報、モデル重みは移行ZIPに含めません。
Pythonやモデルの構築は `CODEX_HANDOFF.md` の初回確認を参照してください。
Windows上でのスクリプト実行は、このクラウドからは実機確認できていません。

## 移行ZIPを取得できない場合

Git for Windowsが使える場合は、PowerShellで次を実行します。新しいフォルダ名を指定してください。

```powershell
git clone --branch improve-playable-tabs https://github.com/kmmn-oioe/tabs.git tabs-cli
Set-Location tabs-cli
New-Item -ItemType Directory -Force local-input/media
```

手元に保存している音源を次の名前で `local-input/media` にコピーします。

- 原曲MP3 → `original.mp3`
- 上がリード・下がリズムのTAB動画MP4 → `reference-tabs.mp4`
- 以前の `guitar_intro.wav` → `rhythm-only-intro.wav`

Gitを使わず [GitHubのソースZIP](https://github.com/kmmn-oioe/tabs/archive/refs/heads/improve-playable-tabs.zip)
を展開することもできますが、履歴やGitHubへのpushは別途Gitの設定が必要です。

GitHubには音源・固定分離結果・非公開の候補キャッシュは置いていません。
この代替手順だけでは固定入力リプレイ用のデータはそろいません。
手元の原曲で通常のモデル推論を実行できますが、再分離で比較値は変わることがあります。
元の添付ファイルを用意できない場合、新しい音源用に別の評価ケースを作り、既存結果と混同しないでください。

## 配置の確認

Git利用時は `tabs`（または `tabs-cli`）内で次を確認します。

```powershell
git branch --show-current
git log -1 --oneline
git status --short
Get-ChildItem local-input/media
```

ブランチは `improve-playable-tabs`。`AGENTS.md` と `docs/CODEX_HANDOFF.md` があることを確認します。
アプリ変更の基点は `3030f83` で、移行文書を追加した新しいコミットも含まれます。
検証データはignoreされるため `git status` に音源が表示されないのが正しい状態です。

Codex CLIを導入した後、この作業フォルダで `codex` を起動し、引き継ぎ文書末尾の依頼文を渡してください。
