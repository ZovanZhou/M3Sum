import os
import re
import cv2
import json
import numpy as np
import pandas as pd
from datetime import datetime
from generate_summary import generate_summary

def evaluate_summary(predicted_summary, user_summary, eval_method):
    max_len = max(len(predicted_summary),user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary

    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G
        
        # Compute precision, recall, f-score
        precision = sum(overlapped) / sum(S) if sum(S) != 0 else 0
        recall = sum(overlapped) / sum(G)
        if (precision + recall == 0):
            f_scores.append(0)
        else:
            f_scores.append(2*precision*recall*100/(precision+recall))

    if eval_method == 'max':
        return max(f_scores)
    else:
        return sum(f_scores)/len(f_scores)

def parse_caption_result(vname, dtype, n_frames, fps):
    dict_pred = {}
    with open(f"./data/{dtype}/{vname}.txt", "r") as fr:
        for line in fr.readlines():
            line = line.strip()
            if line:
                if "Frame" in line and len(line) > 5:
                    if "{" in line:
                        if "}" in line:
                            results = json.loads(line)
                            for k, v in results.items():
                                fid = int(k.split("Frame")[1])
                                if fid not in dict_pred:
                                    dict_pred[fid] = v
                                else:
                                    dict_pred[fid] = (dict_pred[fid] + v) // 2
                        else:
                            results = re.findall(r"\"Frame\d+\": \d", line)
                            for text in results:
                                fid, score = text.split(":")
                                fid = int(fid.split("\"")[1][5:])
                                score = int(score.strip())
                                if fid not in dict_pred:
                                    dict_pred[fid] = score
                                else:
                                    dict_pred[fid] = (dict_pred[fid] + score) // 2
                    else:
                        results = re.findall(r"\"Frame\d+\": \d", line)
                        if len(results) == 0 and len(line) == len(line.split(":")[0]) + 3 and "\"" not in line:
                            text = line.split("Frame")[1]
                            try:
                                fid, score = [int(ele.strip()) for ele in text.split(":")]
                                dict_pred[fid] = score
                            except Exception:
                                pass
                        elif results:
                            text = results[0]
                            fid, score = text.split(":")
                            fid = int(fid.split("\"")[1][5:])
                            score = int(score.strip())
                            dict_pred[fid] = score
                else:
                    results = re.findall(r"\"\d+\": \d", line)
                    if results:
                        text = results[0]
                        fid, score = text.split(":")
                        fid = int(fid.split("\"")[1])
                        score = int(score.strip())
                        dict_pred[fid] = score
    positions = []
    scores = []
    max_frame_id = max(list(dict_pred.keys()))
    shot_bound = pd.read_pickle(f"./data/shot_bounds.pkl")[vname]

    for k, v in dict_pred.items():
        scores.append(v)
        positions.append((k-1)*2*fps)

    summary = generate_summary([shot_bound], [scores], [n_frames], np.array([positions], dtype=np.int32))
    return summary[0]

def parse_transcript_result(vname, dtype, n_frames, fps):
    dict_pred = {}

    with open(f"./data/{dtype}/{vname}.txt", "r") as fr:
        for line in fr.readlines():
            line = line.strip()
            result = re.findall(r"\"Frame\d+-\d+\": \d", line) if "\"" in line else re.findall(r"Frame\d+-\d+: \d", line)
            if len(result):
                frange, score = result[0].split(": ")
                start_frame, end_frame = [int(ele) for ele in re.findall(r"\d+-\d+", frange)[0].split("-")]
                score = int(score.strip(","))
                for i in range(start_frame, end_frame):
                    dict_pred[i] = score
            else:
                result = re.findall(r"\"Frame\d+\": \d", line)
                if len(result):
                    text = result[0]
                    fid, score = text.split(":")
                    fid = int(fid.split("\"")[1][5:])
                    score = int(score.strip())
                    dict_pred[fid] = score
                else:
                    result = re.findall(r"Frame\d+: \d", line)
                    if len(result):
                        text = result[0]
                        fid, score = text.split(":")
                        fid = int(fid[5:])
                        score = int(score.strip())
                        dict_pred[fid] = score
    if len(dict_pred) == 0:
        return [0]

    positions = []
    scores = []
    max_frame_id = max(list(dict_pred.keys())) + 1
    shot_bound = pd.read_pickle(f"./data/shot_bounds.pkl")[vname]

    for k, v in dict_pred.items():
        scores.append(v)
        positions.append(k*fps*2)

    summary = generate_summary([shot_bound], [scores], [n_frames], np.array([positions], dtype=np.int32))
    return summary[0]

def get_video_fps(vname):
    cap = cv2.VideoCapture(vname)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    cap.release()
    cv2.destroyAllWindows()
    return fps

def get_caption_data(vname):
    caption = []
    dict_caption = {}
    cnt = 1
    with open(f"./TVSum_caption/{vname}.txt", "r") as fr:
        for line in fr.readlines():
            if r"\"" in line:
                line = line.replace(r"\"", "")
            text = line.split("\"")[1]
            if "'" in text:
                text = text.replace("'", r"\'")
            sentence = f"Frame{cnt}: {text}"
            caption.append(sentence)
            dict_caption[cnt] = text
            cnt += 1
    return caption, dict_caption

def get_transcript_data(vname):
    transcript = []
    dict_transcript = {}

    raw_data = []
    with open(f"./TVSum_transcript/{vname}.srt", "r", encoding="utf-8") as fr:
        for line in fr.readlines():
            line = line.strip()
            if line:
                raw_data.append(line)

    # tmp_data = ()
    for i in range(len(raw_data) // 3):
        _, time_slot, text = raw_data[i * 3 : (i + 1) * 3]
        start_time, end_time = [datetime.strptime(ele.split(",")[0], "%H:%M:%S") for ele in time_slot.split(" --> ")]
        
        base_time = datetime.strptime("00:00:00", "%H:%M:%S")
        
        start_time = (start_time - base_time).seconds
        start_frame = start_time // 2
        if start_frame * 2 < start_time:
            start_frame += 1

        end_time = (end_time - base_time).seconds
        end_frame = end_time // 2
        if end_frame * 2 < end_time:
            end_frame += 1

        sentence = f"Frame{start_frame}-{end_frame}: {text}"
        transcript.append(sentence)
        for j in range(start_frame, end_frame + 1):
            dict_transcript[j] = text

    return transcript, dict_transcript

def merge_data_transcript2caption(dict_transcript, dict_caption):
    caption = []
    for k in dict_caption.keys():
        text = f"Frame{k}: {dict_caption[k]}"
        if (k - 1) in dict_transcript:
            text += ". " + dict_transcript[k-1]
        caption.append(text)
    return caption

def merge_data_caption2transcript(transcript, dict_caption):
    extended_transcript = []
    for text in transcript:
        frame_range = text.split(":")[0].strip("Frame")
        start_frame, end_frame = [int(ele) for ele in frame_range.split("-")]
        for i in range(start_frame, end_frame):
            if (i+1) in dict_caption:
                text += dict_caption[i+1] + " "
        extended_transcript.append(text)
    return extended_transcript

def get_raw_data(video_names):
    dict_prompts = {}
    for vname in video_names:
        title = vname.split(".")[0] if "." in vname else vname
        dict_prompts[title] = {}
        
        caption, dict_caption = get_caption_data(title)
        dict_prompts[title]["caption"] = caption

        transcript, dict_transcript = get_transcript_data(title)
        dict_prompts[title]["transcript"] = transcript

        transcript2caption = merge_data_transcript2caption(dict_transcript, dict_caption)
        dict_prompts[title]["transcript2caption"] = transcript2caption

        caption2transcript = merge_data_caption2transcript(transcript, dict_caption)
        dict_prompts[title]["caption2transcript"] = caption2transcript
    return dict_prompts

def get_labels(anno_file_name):
    raw_data = np.loadtxt(anno_file_name, delimiter="\t", dtype="str")
    dict_labels = {}
    for i in range(len(raw_data)):
        fname, dtype, labels = raw_data[i].tolist()
        if fname not in dict_labels:
            dict_labels[fname] = {}
            dict_labels[fname]["dtype"] = dtype
            dict_labels[fname]["labels"] = []
        parsed_labels = labels.split(',')
        tmp_labels = []
        fps = get_video_fps(f"./videos/{fname}.mp4")
        dict_labels[fname]["fps"] = fps
        for j in range(len(parsed_labels)):
            tmp_labels.append(int(parsed_labels[j]))
        dict_labels[fname]["labels"].append(np.expand_dims(np.array(tmp_labels), axis=0))
    return dict_labels

def calculate_ppl(perplexity, texts):
    try:
        dict_ppl = perplexity.compute(model_id='gpt2', add_start_token=False, predictions=texts)
        ppl_score = round(dict_ppl["mean_perplexity"], 2)
    except Exception:
        ppl_score = np.inf
    return ppl_score

def merge_data_with_ppl(perplexity, caption, transcript, cap_pred_labels, trans2cap_pred_labels):
    caption_ppl = calculate_ppl(perplexity, caption)
    transcript_ppl = calculate_ppl(perplexity, transcript)
    if caption_ppl < transcript_ppl:
        pred_labels = cap_pred_labels
    else:
        pred_labels = trans2cap_pred_labels 
    return pred_labels

def merge_data_with_bertscore(bertscore, caption, transcript, cap_pred_labels, trans2cap_pred_labels, threshold: float = 0.69):
    caption = [" ".join(caption)]
    transcript = [" ".join(transcript)]
    results = bertscore.compute(predictions=transcript, references=caption, model_type="distilbert-base-uncased")
    F1 = results["f1"][0]
    if F1 > threshold:
        pred_labels = cap_pred_labels
    else:
        pred_labels = trans2cap_pred_labels
    return pred_labels

def merge_data_with_bleu(bleu, caption, transcript, cap_pred_labels, trans2cap_pred_labels, threshold: float = 0.69):
    caption = [" ".join(caption)]
    transcript = [" ".join(transcript)]
    results = bleu.compute(predictions=transcript, references=caption)
    score = results["google_bleu"]
    if score > threshold:
        pred_labels = cap_pred_labels
    else:
        pred_labels = trans2cap_pred_labels
    return pred_labels