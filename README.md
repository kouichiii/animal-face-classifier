# animal-face-classifier

## データについて

### データソース
このプロジェクトでは、Hugging Face Datasetsの犬猫画像データセットを使用しています。

**データセット URL**: https://huggingface.co/datasets/microsoft/cats_vs_dogs

### データのダウンロード方法
以下を実行
   ```bash
   python image-downloader.py
   ```
猫11740枚、犬11668枚の画像がダウンロードされます。
### データセット構造
```
images/
├── cats/     # 猫の画像
└── dogs/     # 犬の画像
```
