from datasets import load_dataset
import os
import json
from PIL import Image

dataset = load_dataset("lmms-lab/COCO-Caption2017")

json_file_path = os.path.join("data", "coco_caption2017_1k.json")
image_dir = os.path.join("data", "images")
os.makedirs(image_dir, exist_ok=True)

examples = []
for item in dataset["val"]:
    image_path = os.path.join(image_dir, item["question_id"])    
    image = item["image"]
    image.save(image_path)

    question = item["question"]
    answer = item["answer"][0]
    example = {
        "image": item["question_id"],
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{question}",
            },
            {
                "from": "gpt",
                "value": answer
            }
        ]
    }
    examples.append(example)
    if len(examples) == 1000:
        break

with open(json_file_path, "w") as fout:
    json.dump(examples, fout, ensure_ascii=False, indent=2)