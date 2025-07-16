# このプログラムは、Google Geminiを使用して人間の顔画像を「犬のような」または「猫のような」顔として分類するものです。
# 


import json
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
import enum
from pydantic import BaseModel
from time import sleep

NUMBER_OF_IMAGES = 250  # Set the number of images to process

PROMPT = """
# Task
You are an expert in facial analysis. Determine if the person in the image has a "dog-like" or "cat-like" face.

# Criteria
* **Dog-like features:** Round/drooping eyes, round face, larger/rounder nose, wide mouth, friendly impression.
* **Cat-like features:** Almond-shaped/upturned eyes, sharp jawline/triangular face, slender nose, small mouth, cool/mysterious impression.

# Instructions
1.  Analyze the image based on the criteria above.
2.  You must make a definitive choice between "dog_like" or "cat_like".
3.  Provide a clear reasoning for your decision.

# Output Format
Your response MUST be in the following format, and nothing else.

Decision: [dog_like or cat_like]
Reason: [Your detailed reasoning here.]
"""

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

client = genai.Client()

# Directories
human_dir = "images/humans"
humans_classified_dir = "images/humans-classified"
cat_dir = "images/humans-classified/cat-like"
dog_dir = "images/humans-classified/dog-like"

results = {}
print("Wait 6 seconds between each image processing to avoid rate limits")
for image_index in range(NUMBER_OF_IMAGES):
    image_name = f'{str(image_index)}.jpg'
    print(f"Processing image: {image_name}")
    image_path = os.path.join(human_dir, image_name)
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
                PROMPT
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

print("Classification completed. Each indexes have been saved")