# LDC-ONNX-Sample
[LDC: Lightweight Dense CNN for Edge Detection](https://github.com/xavysp/LDC)のPythonでのONNX推論サンプルです。<br>
ONNXに変換したモデルも同梱しています。<br>
変換自体を試したい方はColaboratoryで[LDC-Convert2ONNX.ipynb](LDC-Convert2ONNX.ipynb)を使用ください。<br>

# Requirement 
* OpenCV 4.5.3.56 or later
* onnxruntime 1.13.0 or later

# Demo
デモの実行方法は以下です。
```bash
python demo_onnx.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --image<br>
画像ファイルの指定 ※指定時はカメラデバイス・動画ファイルより優先<br>
デフォルト：指定なし
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：model/LDC_640x360.onnx

# Reference
* [xavysp/LDC](https://github.com/xavysp/LDC)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
FLW-Net-onnx2tf-sample is under [MIT License](LICENSE).

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[イギリス オックスフォードのクライスト・チャーチ学生寮外観](https://www2.nhk.or.jp/archives/movies/?id=D0002011220_00000&ref=search)を使用しています。
