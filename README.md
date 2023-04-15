> **Note**<br>
> 　ライセンスは以下Issueで問い合わせ中です。<br>
> 　このリポジトリのライセンスもIssue回答結果に従う予定です。<br>
> 　https://github.com/xavysp/LDC/issues/12

# LDC-ONNX-Sample
[LDC: Lightweight Dense CNN for Edge Detection](https://github.com/xavysp/LDC)のPythonでのONNX推論サンプルです。<br>
ONNXに変換したモデルも同梱しています。<br>
変換自体を試したい方はColaboratoryで[LDC-Convert2ONNX.ipynb](LDC-Convert2ONNX.ipynb)を使用ください。<br>

<img src="https://user-images.githubusercontent.com/37477845/232208727-ac5e23cb-db96-4790-88d2-945f4912aef3.jpg" loading="lazy" width="250px">　
<img src="https://user-images.githubusercontent.com/37477845/232208729-1aef71fb-55f2-493b-87af-6a6823022d26.png" loading="lazy" width="250px">　
<img src="https://user-images.githubusercontent.com/37477845/232208735-7923d7a6-833f-426f-b116-9e2296ba0c5b.png" loading="lazy" width="250px">　
<br>
1枚目：元画像　2枚目：Fuse画像(LDC_3840x2160.onnx使用)　3枚目：Average画像(LDC_3840x2160.onnx使用)

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
