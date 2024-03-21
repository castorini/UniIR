import argparse
import requests
import os

from . import generator_prompt


def infer_gpt(images, p_class):
    api_key = os.environ["OPEN_API_KEY"]

    for image in images:
        retrieval_results = {"hits": []}

        message = p_class.prepare_message(retrieval_results)
        encoded_image = p_class.encode_image(image)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model":
            "gpt-4-vision-preview",
            "messages": [{
                "role":
                "user",
                "content": [{
                    "type": "text",
                    "text": f"{message}"
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }]
            }],
            "max_tokens":
            300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers=headers,
                                 json=payload)

        print(f"Processed image: {image}")
        print(response.json())
        print("-" * 79)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default=False, help="Image path")
    parser.add_argument('--prompt_file', default=False, help="Prompt file")
    parser.add_argument('--model_name', default="gpt")
    args = parser.parse_args()

    infer_mapping = {
        "gpt": infer_gpt,
    }

    image_path = args.image_path

    images = []
    if os.path.dirname(image_path):
        for file in os.listdir(image_path):
            images.append(os.path.join(image_path, file))
    else:
        images = [image_path]

    p_class = generator_prompt.Prompt(args.prompt_file)
    infer_mapping[args.model_name](images, p_class)


if __name__ == '__main__':
    main()
