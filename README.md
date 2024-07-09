# ネイル装着アプリ
## 仮想環境の構築

ltkinterのインストール
```
Ø% brew install python-tk

```

仮想環境の有効化
```
Ø% cd ~/ub_2023
Ø% cd gesture
Ø% source ./bin/activate
```

有効化完了.
続けてGUIアプリ用のライブラリインストール 
```
Ø(gesture)% pip install pysimplegui
Ø(gesture)% pip install py2app
```

## 実装

nailapp.pyを「gesture」フォルダへ移動
```
(gesture)% python nailapp.py

```

終了時はdeactivateを忘れずに! 同様に， 開始時のsource ./bin/activateも忘れずに!
