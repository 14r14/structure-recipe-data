import os
import json

import pandas as pd

from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
from schema import Recipe

login()

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
hf = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 10},
)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
# hf = HuggingFacePipeline(pipeline=pipe)

df = pd.read_json("./recipes_v2.json")
df.head()
# pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")

# from dotenv import load_dotenv
# from parser import RecipeParser, MESSAGE, CHAT_PROMPT_TEMPLATE
