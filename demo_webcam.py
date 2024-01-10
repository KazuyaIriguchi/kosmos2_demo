from PIL import Image
import requests
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
import cv2
import numpy as np

model = Kosmos2ForConditionalGeneration.from_pretrained(
    "microsoft/kosmos-2-patch14-224"
)
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("can't open webcam")

ret, frame = cap.read()
cap.release()

image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

prompt = "<grounding> An image of"
inputs = processor(text=prompt, images=image, return_tensors="pt")

generated_ids = model.generate(
    pixel_values=inputs["pixel_values"],
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    image_embeds=None,
    image_embeds_position_mask=inputs["image_embeds_position_mask"],
    use_cache=True,
    max_new_tokens=64,
)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
processed_text = processor.post_process_generation(
    generated_text, cleanup_and_extract=False
)
print(f"{processed_text=}")

caption, entities = processor.post_process_generation(generated_text)
print(f"{caption=}")
print(f"{entities=}")
