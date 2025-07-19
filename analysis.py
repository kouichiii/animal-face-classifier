# ステージ1：データ前処理
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import Model
from keras.applications import MobileNetV2

# ...（之前的常量定义保持不变）...

cat_size = 1000
dog_size = 1000
human_size = 200

def load_and_preprocess(image_paths):
    images = []
    for path in image_paths:
        # 3チャネルカラー画像として読み込み
        img = cv2.imread(path)
        if img is None:
            print(f"画像の読み込みに失敗しました: {path}")
            continue
            
        # リサイズと前処理
        img = cv2.resize(img, (224, 224))  # MobileNetV2に合わせたサイズ
        # MobileNetV2の前処理を適用
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        images.append(img)
    return np.array(images)

# データセット読み込み
print("画像をロード中...")
cat_images = load_and_preprocess([f"images/cats/{i}.jpg" for i in range(cat_size)])
dog_images = load_and_preprocess([f"images/dogs/{i}.jpg" for i in range(dog_size)])
human_images = load_and_preprocess([f"images/humans/{i}.jpg" for i in range(human_size)])
print("画像のロード完了")

# CNN特徴抽出器の初期化
def create_feature_extractor():
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3),
        pooling='avg'  # グローバル平均プーリングで固定サイズの出力
    )
    return Model(inputs=base_model.input, outputs=base_model.output)

# 特徴抽出
print("CNN特徴抽出中...")
feature_extractor = create_feature_extractor()

def extract_features(images):
    # バッチ処理で特徴抽出
    features = []
    batch_size = 32
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        features.append(feature_extractor.predict(batch))
    return np.vstack(features)

cat_features = extract_features(cat_images)
dog_features = extract_features(dog_images)
human_features = extract_features(human_images)
print("特徴抽出完了")

# ステージ2：PCA次元削減と特徴抽出
from sklearn.decomposition import PCA

# 動物データを統合（ラベル：猫=0，犬=1）
X_animals = np.vstack([cat_features, dog_features])
y_animals = np.array([0]*cat_size + [1]*dog_size)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_animals, y_animals, test_size=0.2, stratify=y_animals, random_state=42
)

# PCA次元削減（分散95%を保持）
pca = PCA(n_components=0.95)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# 人間データを同一空間に射影
X_human_pca = pca.transform(human_features)


# 寄与率プロット
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.show(block=False)

# 主成分可視化

print(f"Reduced dimensions: {X_train_pca.shape[1]}")

# ステージ3：SVMモデル訓練と分類
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# ハイパーパラメータチューニング（交差検証）
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
svm = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
svm.fit(X_train_pca, Y_train)

# 最適動物分類モデルの評価
print(f"Best parameters: {svm.best_params_}")
print(f"Training accuracy: {svm.score(X_test_pca, Y_test)}")

# 人間と動物の類似性予測
human_predictions = svm.predict(X_human_pca)  # 0=猫似，1=犬似

# ステージ4：結果の可視化

# 人間画像と類似結果を表示
def deprocess_mobilenetv2(x):
    """MobileNetV2の前処理を逆変換"""
    x = (x + 1) * 127.5  # [-1,1] → [0,255] に変換
    return np.clip(x, 0, 255).astype(np.uint8)

for i in range(human_size):
    print(f"Image {i}: {'Cat' if human_predictions[i]==0 else 'Dog'} similarity")

# 人間画像と類似結果を表示
fig, axes = plt.subplots(5, 6, figsize=(10, 12))
for i, ax in enumerate(axes.flat):
    # 前処理を逆変換 + BGR→RGB変換
    img = deprocess_mobilenetv2(human_images[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCVはBGR形式のためRGBに変換
    ax.imshow(img)
    ax.set_title(f"Resembles: {'Cat' if human_predictions[i]==0 else 'Dog'}")
    ax.axis('off')
plt.show(block=False)

# ステージ5：t-SNEによる可視化

from sklearn.manifold import TSNE

# 数据合并
X_all = np.vstack([X_animals, human_features])
y_all = np.array([0]*cat_size + [1]*dog_size + [2]*human_size)

# 特征降维
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X_all)

# 可视化
plt.figure(figsize=(10, 6))
colors = ['orange', 'blue', 'green']
labels = ['Cat', 'Dog', 'Human']
for i in range(3):
    plt.scatter(X_embedded[y_all==i, 0], X_embedded[y_all==i, 1], 
                label=labels[i], alpha=0.6)
plt.legend()
plt.title('t-SNE of CNN Features')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.grid(True)
plt.show()
