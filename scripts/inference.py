import argparse
import os
import pathlib
from typing import List

from dotenv import load_dotenv
import google.generativeai as genai
import openai
from openai import AzureOpenAI
from PIL import Image
from transformers import (AutoProcessor, Blip2Processor,
                          Blip2ForConditionalGeneration,
                          LlavaForConditionalGeneration)

import generator_prompt

MAX_TOKENS = 200

load_dotenv()


def infer_gemini(images: List[str], p_class: generator_prompt.Prompt):
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    for image in images:
        model = genai.GenerativeModel('gemini-pro-vision')

        cookie_picture = [{
            'mime_type': 'image/png',
            'data': pathlib.Path(image).read_bytes()
        }]
        retrieval_results = {"hits": []}

        message = p_class.prepare_message(retrieval_results)

        response = model.generate_content(model="gemini-pro-vision",
                                          content=[message, cookie_picture])
        print(f"Processed image: {image}")
        print(response.text)
        print("-" * 79)


def infer_gpt(images: List[str], p_class: generator_prompt.Prompt):
    azure_openai_api_version = os.environ["AZURE_OPENAI_API_VERSION"]
    azure_openai_api_base = os.environ["AZURE_OPENAI_API_BASE"]
    open_ai_api_key = os.environ["OPEN_AI_API_KEY"]
    deployment_name = os.environ["DEPLOYMENT_NAME"]

    client = AzureOpenAI(api_key=open_ai_api_key,
                         api_version=azure_openai_api_version,
                         azure_endpoint=azure_openai_api_base)

    for image in images:
        retrieval_results = {"hits": []}

        message = p_class.prepare_message(retrieval_results)
        encoded_image_url = p_class.encode_image_as_url(image)

        messages = [{
            "role":
            "user",
            "content": [{
                "type": "text",
                "text": message,
            }, {
                "type": "image_url",
                "image_url": {
                    "url": encoded_image_url
                }
            }]
        }]

        try:
            response = client.chat.completions.create(model=deployment_name,
                                                      messages=messages,
                                                      max_tokens=MAX_TOKENS)
            output = response.choices[0].message.content.lower(
            ) if response.choices[0].message.content else ""
        except openai.BadRequestError as e:
            print(f"Encountered {e}")
            output = ""

        print(f"Processed image: {image}")
        print(output)
        print("-" * 79)


def infer_llava(
    images: List[str],
    p_class: generator_prompt.Prompt,
    model_name: str = "llava-hf/llava-1.5-7b-hf",
):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, device_map="auto", low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

    prompts = []
    input_images = []

    for image_path in images:
        image = Image.open(image_path)
        input_images.append(image)

        retrieval_results = {"hits": []}
        message = p_class.prepare_message(retrieval_results)
        prompts.append(f"USER: <image>\n{message}\nASSISTANT:")

    inputs = processor(prompts,
                       images=input_images,
                       padding=True,
                       return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    for text, image_path in enumerate(generated_text, images):
        print(f"Processed image: {image_path}")
        print(text.split("ASSISTANT:")[-1])
        print("-" * 79)


def infer_blip(
    images: List[str],
    p_class: generator_prompt.Prompt,
    model_name: str = "Salesforce/blip2-flan-t5-xl",
):
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, device_map="auto", low_cpu_mem_usage=True)
    processor = Blip2Processor.from_pretrained(model_name, use_fast=True)

    prompts = []
    input_images = []

    for image_path in images:
        image = Image.open(image_path)
        input_images.append(image)

        retrieval_results = {"hits": []}
        message = p_class.prepare_message(retrieval_results)
        prompts.append(message)

    inputs = processor(prompts,
                       images=input_images,
                       padding=True,
                       return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    for text, image_path in enumerate(generated_text, images):
        print(f"Processed image: {image_path}")
        print(text)
        print("-" * 79)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default=False, help="Image path")
    parser.add_argument('--prompt_file', default=False, help="Prompt file")
    parser.add_argument('--model_name', default="gpt")
    args = parser.parse_args()

    infer_mapping = {
        "gpt": infer_gpt,
        "gemini": infer_gemini,
        "llava": infer_llava,
        "blip": infer_blip
    }

    image_path = args.image_path

    images = []
    if os.path.isdir(image_path):
        for file in os.listdir(image_path):
            images.append(os.path.join(image_path, file))
    else:
        images = [image_path]

    p_class = generator_prompt.Prompt(args.prompt_file)
    infer_mapping[args.model_name](images, p_class)


if __name__ == '__main__':
    main()
