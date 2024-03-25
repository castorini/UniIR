import argparse
import base64
from mimetypes import guess_type


class Prompt:

    def __init__(self, prompt_file) -> None:
        with open(prompt_file) as p:
            self.prompt_template = "".join(p.readlines()).strip()

    def prepare_message(self, retrieval_results):
        examples = ""
        for index, hit in enumerate(retrieval_results["hits"]):
            examples += f"Passage{index+1}: {hit['content']}\n"

        prompt = self.prompt_template.format(examples=examples,
                                             num=len(
                                                 retrieval_results["hits"]))
        return prompt

    def encode_image_as_url(self, image_path):
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_file', default=False, help="Prompt file")

    args = parser.parse_args()

    retrieval_results = {"hits": []}

    p = Prompt(args.prompt_file)
    print(p.prepare_message(retrieval_results))


if __name__ == '__main__':
    main()
