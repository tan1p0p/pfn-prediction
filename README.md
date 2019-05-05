# pfn2019 インターン課題2

``` bash
# docker利用時
docker build -t pfn2019 .
docker run -it pfn2019 /bin/bash

# 学習時
# src/train.py内にアノテーションファイルのパスを記述
python3 src/train.py

# 予測時
# src/predict.py内に画像フォルダ、出力先のパスを記述
python3 src/predict.py
```