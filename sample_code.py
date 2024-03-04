from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# model_path = "liuhaotian/llava-v1.5-7b"

# tokenizer, model, image_processor, context_len = load_pretrained_model(
#     model_path=model_path,
#     model_base=None,
#     model_name=get_model_name_from_path(model_path)
# )

#2

model_path = "liuhaotian/llava-v1.5-7b"
prompt = "What are the things I should be cautious about when I visit here?"
# image_file = "https://llava-vl.github.io/static/images/view.jpg"
image_file = "https://scontent-sjc3-1.xx.fbcdn.net/v/t39.30808-6/414961043_2573484706145054_4797642975074305988_n.jpg?_nc_cat=100&ccb=1-7&_nc_sid=3635dc&_nc_ohc=gd2r2bk7sDgAX9WLfeI&_nc_ht=scontent-sjc3-1.xx&oh=00_AfBdDFP8AZpkJooikDB1nOscDkdJ-G3irkKszsAIrd059A&oe=659B2C33"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 128
})()

eval_model(args)
