import json

retrieved_cand_path = "/mnt/users/s8sharif/UniIR/data/UniIR/retrieval_results/BLIP_FF/Large/Instruct/InBatch/retrieved_candidates/mbeir_mscoco_task3_union_pool_test_k10_retrieved.jsonl"
retrieved_cands_dict = {}
with open(retrieved_cand_path, "r") as f:
    for line in f:
        obj = json.loads(line)
        retrieved_cands_dict[obj["query"]["qid"]] = obj
problem_qids = set()
for qid, obj in retrieved_cands_dict.items():
    for complement_cand in obj["complement_candidates"]:
        if complement_cand["img_path"] == obj["query"]["query_img_path"]:
            problem_qids.add(qid)
            break
for qid in problem_qids:
    print("\n" + retrieved_cands_dict[qid].__repr__() + "\n")
