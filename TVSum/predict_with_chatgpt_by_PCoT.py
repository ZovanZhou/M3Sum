import os
import argparse
import numpy as np
from utils import *
from tqdm import tqdm
from chatgpt import ChatGPT

def construct_process_prompt(video_info_prompt, summarization_content, frames_content):
    process_prompt = r"""Please help me score the frames for selecting important frames. Here is the process you will follow:
1. Firstly, you will read the descriptions of frames across different ranges.
2. After that, you will review each frame's descriptions and establish a relationship between them.
3. The range for scoring the frames should be 1 to 5; a higher score represents greater significance. Using the frame descriptions, you must assign scores to each frame. It is suggested to allocate more low scores and fewer higher scores.
4. Lastly, you must output the frame scores in JSON format, excluding any descriptions of frames.
The presented descriptions of frames are as follows:

"""

    return video_info_prompt + "The video content is \"" + summarization_content + "\". " + process_prompt + frames_content

def construct_summarization_prompt(frames_content):
    summarize_prompt = r"""Please summarize the important information based on all descriptions of frames. Making your answer is concise and accurate. The descriptions of frames are presented as follows:

"""
    return summarize_prompt + frames_content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="caption", choices=["caption", "transcript", "transcript2caption", "caption2transcript"]
    )
    parser.add_argument(
        "--batch_size", type=int, default=120,
    )
    args = parser.parse_args()

    dict_info = {}
    with open("./data/ydata-tvsum50-info.tsv", "r") as fr:
        for line in fr.readlines():
            category, video_id, title, _, _ = line.strip().split("\t")
            if "\"" in title:
                title = title.split("\"")[1]
            if "\'" in title:
                title = title.replace("'", r"\'")
            dict_info[video_id] = {
                "category": category,
                "title": title
            }

    video_names = os.listdir("./videos")
    dict_prompts = get_raw_data(video_names)

    batch_size = args.batch_size
    chatgpt = ChatGPT(api_key="xxxxx")

    for vname in tqdm(video_names):
        title = vname.split(".")[0]
        video_info_prompt = "The video type is " + dict_info[title]["category"] + r" and the title of the video is \"" + dict_info[title]["title"] + r"\". "
        
        if args.data == "caption":
            contents = dict_prompts[title]["caption"]

        if args.data == "transcript":
            transcripts = dict_prompts[title]["transcript"]
            contents = transcripts if len(transcripts) else dict_prompts[title]["caption"]

        if args.data == "transcript2caption":
            contents = dict_prompts[title]["transcript2caption"]

        if args.data == "caption2transcript":
            contents = dict_prompts[title]["caption2transcript"]
        
        video_summarization_content = ""
        for i in range(0, len(contents) // batch_size + 1):
            prompt = construct_summarization_prompt(video_summarization_content + "\n".join(contents[i*batch_size: (i+1) * batch_size]))
            video_summarization_content = chatgpt.respond(prompt)
        # print(video_summarization_content)

        save_result_path = f"./data/PCoT_{args.data}"
        if not os.path.exists(save_result_path):
            os.mkdir(save_result_path)

        with open(f"{save_result_path}/{title}.txt", "w", encoding="utf-8") as fw:
            for i in range(0, len(contents) // batch_size + 1):
                prompt = construct_process_prompt(video_info_prompt, video_summarization_content, "\n".join(contents[i * batch_size: (i + 1) * batch_size]))
                # print(prompt)
                response = chatgpt.respond(prompt)
                if response:
                    fw.write(response)
                    fw.write("\n")
                else:
                    tqdm.write(vname)
                    fw.write(prompt)
                    fw.write("\n")

if __name__ == "__main__":
    main()