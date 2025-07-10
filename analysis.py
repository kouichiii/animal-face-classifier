# ステージ1：データ前処理
import cv2
import numpy as np
import matplotlib.pyplot as plt

cat_size = 100
dog_size = 100
human_size = 200

def load_and_preprocess(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # グレースケール化（チャネル干渉低減）
        img = cv2.resize(img, (128, 128))             # サイズ統一化
        img = cv2.equalizeHist(img)                   # ヒストグラム均等化（照明影響の軽減）
        images.append(img.flatten())                  # 1次元ベクトルに変換
    return np.array(images)

# データセット読み込み
cat_images = load_and_preprocess([f"images/cats/{i}.jpg" for i in range(cat_size)])
dog_images = load_and_preprocess([f"images/dogs/{i}.jpg" for i in range(dog_size)])
human_images = load_and_preprocess([f"images/humans/{i}.jpg" for i in range(human_size)])


# ステージ2：PCA次元削減と特徴抽出
from sklearn.decomposition import PCA

# 動物データを統合（ラベル：猫=0，犬=1）
X_train = np.vstack([cat_images, dog_images])
y_train = np.array([0]*cat_size + [1]*dog_size)

# PCA次元削減（分散95%を保持）
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)

# 人間データを同一空間に射影
X_human_pca = pca.transform(human_images)

# 寄与率プロット
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')

# 主成分可視化
n_components = pca.n_components_
eigenfaces = pca.components_.reshape((n_components, 128, 128))

plt.subplot(122)
plt.imshow(eigenfaces[0], cmap='gray')  # 第1主成分
plt.title('Top Principal Component')
plt.show(block=False)

print(f"Reduced dimensions: {X_train_pca.shape[1]}")

# ステージ3：SVMモデル訓練と分類
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# ハイパーパラメータチューニング（交差検証）
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
svm = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
svm.fit(X_train_pca, y_train)

# 人間と動物の類似性予測
human_predictions = svm.predict(X_human_pca)  # 0=猫似，1=犬似

# ステージ4：結果の可視化

# 人間画像と類似結果を表示
fig, axes = plt.subplots(5, 3, figsize=(10, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(human_images[i].reshape(128, 128), cmap='gray')
    ax.set_title(f"Resembles: {'Cat' if human_predictions[i]==0 else 'Dog'}")
    ax.axis('off')
plt.show()

