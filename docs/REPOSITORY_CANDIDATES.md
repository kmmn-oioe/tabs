# 調査した外部リポジトリ（2026-11-17）

README・ライセンス・公開コミット履歴を調査済み。**導入・精度比較は未実施**。
コードと重み・学習データの利用条件を混同しない。

| 候補 | 用途 | 確認した点 |
|---|---|---|
| https://github.com/nomadkaraoke/python-audio-separator | 複数の音源分離モデルを比較するPython基盤 | MIT。CPU/GPU、Windowsの説明あり。最終確認コミット2026-10-29 |
| https://github.com/chenCFD/Guitar_Source_Separation | ギター／その他のMel-Band RoFormer比較 | コードMIT。READMEに学習済みモデル取得手順。リード／リズム専用ではない。重みの条件は要確認 |
| https://github.com/cwitkowitz/guitar-transcription-continuous | FretNet：弦・音程軌跡を扱うギター採譜 | MIT。最終コミット2023-06。研究実装、UsageにTODOあり。重みと依存関係を要確認 |
| https://github.com/cwitkowitz/guitar-transcription-with-inhibition | 弾きにくい音の組み合わせを抑制 | MIT。最終コミット2022-12。amt-toolsを使用する研究実装 |
| https://github.com/marl/GuitarSet | 注釈付きデータによる採譜評価 | コードMIT。データ条件は配布元で別途確認。既知の注釈誤りあり。2本の役割分離用正解ではない |
| https://github.com/robust-guitar-tabs/code | エフェクト・歪みに頑健な採譜の研究 | ルートCC0。依存コード・データは各条件を確認。最終コミット2024-06 |
| https://github.com/ZFTurbo/Music-Source-Separation-Training | 将来の専用分離モデル学習 | MIT。最終コミット2026-11-12。現READMEはtorch2.6以上。既存アプリvenvと分ける |
| https://github.com/CoderLine/alphaTab | Web向けTAB表示・音源同期・シンセ | コードMPL-2.0。最終確認コミット2026-11-12。Python GUIへはWebView等が必要。採譜モデルではない |

追加候補:

- https://github.com/cwitkowitz/amt-tools ：FretNet等の共通学習・評価基盤、MIT。
- https://github.com/yongyizang/SynthTab ：合成ギターの学習データ。READMEはデータをCC BY-NC 4.0と明記し、
  全体は約2TB。一方ルートLICENSEはCC BY 4.0なので対象別の条件を確認する。少量デモあり。
- https://github.com/NikolajChristensen/ncosis ：README上、MVSEPのLead/Rhythm Guitarモデルを外部APIで利用する例。
  ローカルモデル本体ではなく音源の外部送信・アカウントが必要。ルートでライセンス表示を確認できなかった。
- https://github.com/facebookresearch/demucs ：現在使用中。2025-01-01に本家がアーカイブ済み。

優先候補は、別venvでの分離モデル比較、評価曲・正解注釈の拡充、ギター専用採譜との比較。
専用のリード／リズム学習には、役割別の正解録音や重みの確保が必要。リポジトリを導入しただけで
未知曲の精度が上がるとは扱わず、現行Demucs＋Basic Pitchと同条件で比較する。
