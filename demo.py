from PIL import Image
import requests
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration

model = Kosmos2ForConditionalGeneration.from_pretrained(
    "microsoft/kosmos-2-patch14-224"
)
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
image = Image.open(requests.get(url, stream=True).raw)

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
