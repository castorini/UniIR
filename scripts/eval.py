import argparse
import json
import os
from tqdm import tqdm

import jsonlines
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

import candidate_retrieval as cr

RETRIEVAL_BASE_PATH = "/store2/scratch/sjupadhy/mbeir_mscoco_output"


def file_in_dir(dirname, extension):
    if os.path.isdir(dirname):
        files = os.listdir(dirname)
        result = [
            os.path.join(dirname, file) for file in files
            if file.endswith(extension)
        ]
    else:
        result = []
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate_path',
                        default=False,
                        help="Path to jsonl file containing the candidates")
    parser.add_argument('--result_dir', help="Result directory")
    args = parser.parse_args()

    res_files = file_in_dir(args.result_dir, ".json")
    print(f"Count of json file: {len(res_files)}")

    res = {}
    for file in res_files:
        with open(file, 'r') as file:
            data = json.load(file)
        for ind in data:
            res[ind["qid"]] = [ind["response"]]

    clu = cr.CandidateLookUp(args.candidate_path)
    gts = {}
    retrieval_jsonl_path = os.path.join(RETRIEVAL_BASE_PATH,
                                        "mbeir_mscoco_image_to_text.jsonl")
    with jsonlines.open(retrieval_jsonl_path) as reader:
        for obj in tqdm(reader, desc='Reading docs'):
            qid = obj["query"]["qid"] if obj["query"]["query_img_path"] else ""
            if qid in res:
                pos_cand = obj["query"]["pos_cand_list"]
                candidates = []
                for cand in pos_cand:
                    candidates.append(
                        clu.retrieve_candidate_txt_from_did(cand))
                gts[qid] = candidates
            if len(gts) == len(res):
                break
    print(f"ground truth count: {len(gts)}")

    res_filter = {}
    for file in res:
        if file in gts:
            res_filter[file] = res[file]
    print(
        f"Filtered files count: {len(res)} - {len(res_filter)} = {len(res) - len(res_filter)}"
    )

    scorers = [
        Bleu(),
        Cider(),
        Rouge(),
    ]
    scorers_names = [
        "Bleu",
        "Cider",
        "Rouge",
    ]
    result = {}
    for sc, scn in zip(scorers, scorers_names):
        score, scores = sc.compute_score(gts, res_filter)
        print(f"{scn}: {score}")
        result[scn] = score

    output_dir = os.path.join(args.result_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"metrics.json")
    with open(output_path, "w") as outfile:
        json.dump(result, outfile)
    print(f"Output file at: {output_path}")


if __name__ == '__main__':
    main()
