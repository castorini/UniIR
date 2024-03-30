import argparse
import json
import os
from tqdm import tqdm

import jsonlines
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider, CiderScorer
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

def convert_to_tokenizer_input_format(dictionary):
    new_dictionary = {}
    for id, captions in dictionary.items():
        captions_list = []
        for c in captions:
            captions_list.append({"image_id": id, "caption": c})
        new_dictionary[id] = captions_list
    return new_dictionary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate_path',
                        default=False,
                        required=True,
                        help="Path to jsonl file containing the candidates")
    parser.add_argument('--result_dir', required=True, help="Result directory")
    args = parser.parse_args()

    res_files = file_in_dir(args.result_dir, ".json")
    print(f"Count of json file: {len(res_files)}")

    res = {}
    for file in res_files:
        with open(file, 'r') as file:
            data = json.load(file)
        for ind in data:
            res[os.path.basename(ind["image"])] = [ind["response"]]

    clu = cr.CandidateLookUp(args.candidate_path)
    gts = {}
    retrieval_jsonl_path =  "/store2/scratch/s8sharif/UniIR/data/UniIR/retrieval_results/CLIP_SF/Large/Instruct/InBatch/run_files/mbeir_mscoco_task3_union_pool_test_k10_run_2024-03-27 15:28:49.276449.jsonl"

    with jsonlines.open(retrieval_jsonl_path) as reader:
        for obj in tqdm(reader, desc='Reading docs'):
            img =  os.path.basename(obj["query"]["query_img_path"])
            if img in res:
                pos_cand = obj["query"]["pos_cand_list"]
                candidates = []
                for cand in pos_cand:
                    candidates.append(
                        clu.retrieve_candidate_txt_from_did(cand))
                gts[img] = candidates
            else:
                assert False, "retrieved queries and llm queies must match"
            if len(gts) == len(res):
                break
    print(f"ground truth count: {len(gts)}")
    gts = convert_to_tokenizer_input_format(gts)
    res = convert_to_tokenizer_input_format(res)
    tokenizer = PTBTokenizer()
    _gts = tokenizer.tokenize(gts)
    _res = tokenizer.tokenize(res)

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
        score, _ = sc.compute_score(_gts, _res)
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
