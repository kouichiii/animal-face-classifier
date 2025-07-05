from datasets import load_dataset
import os

print("この処理には数分かかります。")
ds = load_dataset("microsoft/cats_vs_dogs")

os.makedirs("images", exist_ok=True)
os.makedirs("images/dogs", exist_ok=True)
os.makedirs("images/cats", exist_ok=True)

cat_index = 0
dog_index = 0
print("ダウンロード中・・・")
for i in range(ds["train"].num_rows):
    img = ds["train"][i]["image"]
    label = ds["train"][i]["labels"]
    if label == 0:
        img.save(f"images/cats/{cat_index}.jpg")
        cat_index += 1
    else:
        img.save(f"images/dogs/{dog_index}.jpg")
        dog_index += 1