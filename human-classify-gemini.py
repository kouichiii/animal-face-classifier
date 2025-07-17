# Geminiを使って、images/humansの顔画像を分類します
# APIを使用するので注意してください

import json
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
import enum
from pydantic import BaseModel
from time import sleep
import prompts as PROMPTS

NUMBER_OF_IMAGES = 50  # Set the number of images to process



class CatOrDog(enum.Enum):
    CAT_LIKE = "cat_like"
    DOG_LIKE = "dog_like"


class Grade(enum.Enum):
    A_PLUS = "a+"
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    F = "f"

class Answer(BaseModel):
  reason: str
  result: CatOrDog

print("GeminiのAPIを使用します。よろしいですか？")
res = input("y/n: ")
while res not in ['y', 'n']:
    print("'y' か 'n'で答えてください。")
    res = input("y/n: ")
if res == 'n':
    print("プログラムを終了します。")
    exit(0)

api_key = os.getenv("GEMINI_API_KEY")
if api_key is None:
    print("GEMINI_API_KEY を入力してください")
    api_key = input("Enter your Gemini API key: ")

os.environ["GEMINI_API_KEY"] = api_key
try:
    client = genai.Client()
except Exception as e:
    print(f"Gemini APIのクライアントを作成できませんでした: {e}")
    exit(1)

results = {}
print("レートリミットを避けるために、各画像処理の間に6秒待機します")
for image_index in range(NUMBER_OF_IMAGES):
    image_name = f'{str(image_index)}.jpg'
    print(f"{image_name} を処理しています")
    image_path = (f"images/humans/{image_name}")
    if os.path.isfile(image_path):
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg',
                ),
                PROMPTS.CLASSIFY_PROMPT
            ],
            config={
                'response_mime_type': 'application/json',
                'response_schema': list[Answer],
            },
        )
        # Wait for the api request
        sleep(6)

        # Copy image to the appropriate directory
        response = json.loads(response.text)
        reason = response[0].get('reason', 'No reason provided')
        classification = response[0].get('result', 'unknown')
        if classification == CatOrDog.CAT_LIKE.value:
            results[image_name] = {
                "classification": CatOrDog.CAT_LIKE.value,
                "reason": reason
            }
        elif classification == CatOrDog.DOG_LIKE.value:
            results[image_name] = {
                "classification": CatOrDog.DOG_LIKE.value,
                "reason": reason
            }
        else:
            print(f"Image {image_name} classified as unknown: {classification}")

    else:
        print(f"Image {image_name} does not exist in the directory.")

result_path = "human-classified-results.json"
with open(result_path, 'w') as f:
    result_json = json.dumps(results, indent=4, ensure_ascii=False)
    f.write(result_json)
    f.close()

print("分類結果を human-classified-results.json に保存しました。")