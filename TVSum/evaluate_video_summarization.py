import json
import argparse
import numpy as np
import pandas as pd
from utils import *
from tqdm import tqdm
from evaluate import load

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="caption", choices=["caption", "transcript", "transcript2caption", "caption2transcript", "merge"]
    )
    parser.add_argument(
        "--pcot", type=int, default=0, choices=[0, 1]
    )
    parser.add_argument(
        "--merge_mode", type=str, default="ppl", choices=["ppl", "bertscore", "bleu"]
    )
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()

    anno_file_name = "./data/ydata-tvsum50-binary-anno.tsv"
    dict_labels = get_labels(anno_file_name)

    test_video_names = pd.read_pickle("./data/test_video_names.pkl")
    dict_prompts = get_raw_data(test_video_names)

    perplexity = load("perplexity", module_type="metric")
    bertscore = load("bertscore")
    bleu = load("google_bleu")

    dtype = f"PCoT_{args.data}" if args.pcot else args.data

    results = []
    # for vname, data in dict_labels.items():
    for vname in tqdm(test_video_names):
        data = dict_labels[vname]
        true_labels = np.vstack(data["labels"])
        fps = dict_labels[vname]["fps"]

        # if vname == "4wU_LUjG5Ic":
        if args.data == "merge":
            if args.pcot:
                cap_pred_labels = parse_caption_result(vname, "PCoT_caption", true_labels.shape[1], fps)
                trans2cap_pred_labels = parse_caption_result(vname, "PCoT_transcript2caption", true_labels.shape[1], fps)
            else:
                cap_pred_labels = parse_caption_result(vname, "caption", true_labels.shape[1], fps)
                trans2cap_pred_labels = parse_caption_result(vname, "transcript2caption", true_labels.shape[1], fps)
            
            if args.merge_mode == "ppl":
                pred_labels = merge_data_with_ppl(perplexity, dict_prompts[vname]["caption"], dict_prompts[vname]["transcript"], cap_pred_labels, trans2cap_pred_labels)
            elif args.merge_mode == "bertscore":
                pred_labels = merge_data_with_bertscore(bertscore, dict_prompts[vname]["caption"], dict_prompts[vname]["transcript"], cap_pred_labels, trans2cap_pred_labels, threshold=args.threshold)
            elif args.merge_mode == "bleu":
                pred_labels = merge_data_with_bleu(bleu, dict_prompts[vname]["caption"], dict_prompts[vname]["transcript"], cap_pred_labels, trans2cap_pred_labels, threshold=args.threshold)

        if args.data in ["caption", "transcript2caption"]:
            pred_labels = parse_caption_result(vname, dtype, true_labels.shape[1], fps)

        if args.data in ["transcript", "caption2transcript"]:
            pred_labels = parse_transcript_result(vname, dtype, true_labels.shape[1], fps)

        f_score = evaluate_summary(pred_labels, true_labels, "avg")
        # print(vname, f_score)
        results.append(f_score)

    print(np.mean(results))

if __name__ == '__main__':
    main()