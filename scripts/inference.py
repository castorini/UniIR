import argparse
from datetime import datetime
import json
import os
import pathlib
from tqdm import tqdm
from typing import Dict, List, Tuple

import jsonlines
from dotenv import load_dotenv
import google.generativeai as genai
import openai
from openai import AzureOpenAI
from PIL import Image
from transformers import (AutoProcessor, Blip2Processor,
                          Blip2ForConditionalGeneration,
                          LlavaForConditionalGeneration)

import generator_prompt

# TODO: make these customizable by adding to args
MAX_TOKENS = 200
MBIER_BASE_PATH = "/mnt/users/s8sharif/M-BEIR/"

load_dotenv()


def infer_gemini(
    images: List[str],
    p_class: generator_prompt.Prompt,
    retrieval_dict: Dict[str, Tuple[str, List[str]]],
):
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    outputs = []
    for image in images:
        model = genai.GenerativeModel('gemini-pro-vision')

        cookie_picture = [{
            'mime_type': 'image/png',
            'data': pathlib.Path(image).read_bytes()
        }]
        qid, retrieval_results = retrieval_dict.get(os.path.basename(image))
        message = p_class.prepare_message(retrieval_results)

        response = model.generate_content(model="gemini-pro-vision",
                                          content=[message, cookie_picture])
        print(f"Processed image: {image}")
        print(response.text)
        outputs.append({
            "qid": qid,
            "image": image,
            "prompt": message,
            "response": response.text
        })
        print("-" * 79)
    return outputs


def infer_gpt(
    images: List[str],
    p_class: generator_prompt.Prompt,
    retrieval_dict: Dict[str, Tuple[str, List[str]]],
):
    azure_openai_api_version = os.environ["AZURE_OPENAI_API_VERSION"]
    azure_openai_api_base = os.environ["AZURE_OPENAI_API_BASE"]
    open_ai_api_key = os.environ["OPEN_AI_API_KEY"]
    deployment_name = os.environ["DEPLOYMENT_NAME"]

    client = AzureOpenAI(api_key=open_ai_api_key,
                         api_version=azure_openai_api_version,
                         azure_endpoint=azure_openai_api_base)

    outputs = []
    for image in images:
        qid, retrieval_results = retrieval_dict.get(os.path.basename(image))
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
        outputs.append({
            "qid": qid,
            "image": image,
            "prompt": message,
            "response": output
        })
        print("-" * 79)
    return outputs


def infer_llava(
    images: List[str],
    p_class: generator_prompt.Prompt,
    retrieval_dict: Dict[str, Tuple[str, List[str]]],
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    bs: int = 4,
):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=os.environ["PYSERINI_CACHE"])
    processor = AutoProcessor.from_pretrained(
        model_name, use_fast=True, cache_dir=os.environ["PYSERINI_CACHE"])

    prompts = []
    input_images = []
    qids = []

    for image_path in images:
        image = Image.open(image_path)
        keep = image.copy()
        input_images.append(keep)
        image.close()

        qid, retrieval_results = retrieval_dict.get(os.path.basename(image_path))
        message = p_class.prepare_message(retrieval_results)
        prompts.append(f"USER: <image>\n{message}\nASSISTANT:")
        qids.append(qid)

    outputs = []
    for i in tqdm(range(0, len(prompts), bs), desc='Batching inputs'):
        inputs = processor(prompts[i:i + bs],
                           images=input_images[i:i + bs],
                           padding=True,
                           return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
        generated_text = processor.batch_decode(output,
                                                skip_special_tokens=True)
        for text, image_path, prompt, qid in tqdm(
                zip(generated_text, images[i:i + bs], prompts[i:i + bs], qids[i: i + bs])):
            print(f"Processed image: {image_path}")
            print(text.split("ASSISTANT:")[-1])
            outputs.append({
                "qid": qid,
                "image": image_path,
                "prompt": prompt,
                "response": text.split("ASSISTANT:")[-1]
            })
            print("-" * 79)

    return outputs


def infer_blip(
    images: List[str],
    p_class: generator_prompt.Prompt,
    retrieval_dict: Dict[str, Tuple[str, List[str]]],
    model_name: str = "Salesforce/blip2-flan-t5-xl",
    bs: int = 4,
):
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=os.environ["PYSERINI_CACHE"])
    processor = Blip2Processor.from_pretrained(
        model_name, use_fast=True, cache_dir=os.environ["PYSERINI_CACHE"])

    prompts = []
    input_images = []
    qids = []

    for image_path in images:
        image = Image.open(image_path)
        keep = image.copy()
        input_images.append(keep)
        image.close()

        qid, retrieval_results = retrieval_dict.get(os.path.basename(image))
        message = p_class.prepare_message(retrieval_results)
        prompts.append(message)
        qids.append(qid)

    inputs = processor(prompts,
                       images=input_images,
                       padding=True,
                       return_tensors="pt").to("cuda")
    outputs = []
    for i in tqdm(range(0, len(prompts), bs), desc='Batching inputs'):
        inputs = processor(prompts[i:i + bs],
                           images=input_images[i:i + bs],
                           padding=True,
                           return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
        generated_text = processor.batch_decode(output,
                                                skip_special_tokens=True)
        for text, image_path, prompt, qid in tqdm(
                zip(generated_text, images[i:i + bs], prompts[i:i + bs], qids[i:i + bs])):
            print(f"Processed image: {image_path}")
            print(text)
            outputs.append({
                "qid": qid,
                "image": image_path,
                "prompt": prompt,
                "response": text
            })
            print("-" * 79)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',
                        default=False,
                        help="Image path or dir")
    parser.add_argument('--prompt_file', default=False, help="Prompt file")
    parser.add_argument('--k', type=int, default=0, help="Number of retrieved examples included in the prompt")
    parser.add_argument('--model_name', default="gpt")
    parser.add_argument('--index',
                        default="full",
                        help="Add start end indices in x_y format")
    parser.add_argument('--output_dir', required=True,
                        help="Base directory to store llm outputs, the output dir would be: output_dir/'model_name'_outputs/'retriever_name'_k")
    parser.add_argument('--retrieved_results_path', required=True,
                        help='path to the jsonl file containing query + candidates pairs')
    parser.add_argument('--retriever_name', required=True,
                        help="Name of the retriever that has retrieved input candidates")
    args = parser.parse_args()
    if args.k == 0 and "-with-" in args.prompt_file:
        raise ValueError("Invalid template file for zero-shot inference.")
    elif args.k > 0 and "-without-" in args.prompt_file:
        raise ValueError("Invalid template file for few-shot inference.")

    infer_mapping = {
        "gpt": infer_gpt,
        "gemini": infer_gemini,
        "llava": infer_llava,
        "blip": infer_blip,
    }

    image_path = args.image_path

    basenames = []
    if os.path.isdir(image_path):
        files = os.listdir(image_path)
        if args.index == "full":
            start = 0
            end = len(files)
        else:
            temp = args.index.split("_")
            start = int(temp[0])
            end = int(temp[1])
        for file in files[start:end]:
            basenames.append(file)
    else:
        basenames = [os.path.basename(image_path)]

    # Storing only relevant retrieval info
    retrieval_dict = {}
    images = []
    with jsonlines.open(args.retrieved_results_path) as reader:
        for obj in tqdm(reader, desc='Reading docs'):
            image = obj["query"]["query_img_path"]
            if image:
                basename = os.path.basename(image)
                if basename in basenames:
                    candidates = []
                    for cand in obj.get("candidates"):
                        candidates.append(cand["txt"])
                    retrieval_dict[basename] = (obj["query"]["qid"], candidates)
                    images.append(os.path.join(MBIER_BASE_PATH, image))
        assert len(retrieval_dict) == len(images), "The number of images and queries should be equal"

    p_class = generator_prompt.Prompt(args.prompt_file, args.k)
    result = infer_mapping[args.model_name](images, p_class, retrieval_dict)
    result_dir = os.path.join(args.output_dir, f"{args.model_name}_outputs", f'{args.retriever_name}_k{args.k}')
    os.makedirs(result_dir, exist_ok=True)
    output_path = os.path.join(result_dir, f"{args.index}_{datetime.isoformat(datetime.now())}.json")
    with open(output_path, "w") as outfile:
        json.dump(result, outfile)
    print(f"Output file at: {output_path}")


if __name__ == '__main__':
    main()
