import os
import argparse
from utils import *
from tqdm import tqdm
from chatgpt import ChatGPT

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
    with open("./data/summe-anno.tsv", "r") as fr:
        for line in fr.readlines():
            video_id, _ = line.strip().split("\t")
            title = video_id
            if "_" in title:
                title = title.replace("'", r"\'")
            dict_info[video_id] = {
                "title": title
            }

    video_names = os.listdir("./videos")
    dict_prompts = get_raw_data(video_names)

    process_prompt = r"""Please help me score the frames for selecting important frames. Here is the process you will follow:
1. Firstly, you will read the descriptions of frames across different ranges.
2. After that, you will review each frame's descriptions and establish a relationship between them.
3. The range for scoring the frames should be 1 to 5; a higher score represents greater significance. Using the frame descriptions, you must assign scores to each frame. It is suggested to allocate more low scores and fewer higher scores.
4. Lastly, you must output the frame scores in JSON format, excluding any descriptions of frames.
The presented descriptions of frames are as follows:

"""

    batch_size = args.batch_size
    chatgpt = ChatGPT(api_key="xxxxx")

    for vname in tqdm(video_names):
        title = vname.split(".")[0]
        video_info_prompt = "The title of the video is \"" + dict_info[title]["title"] + r"\". "
        
        if args.data == "caption":
            contents = dict_prompts[title]["caption"]

        if args.data == "transcript":
            transcripts = dict_prompts[title]["transcript"]
            contents = transcripts if len(transcripts) else dict_prompts[title]["caption"]

        if args.data == "transcript2caption":
            contents = dict_prompts[title]["transcript2caption"]

        if args.data == "caption2transcript":
            contents = dict_prompts[title]["caption2transcript"]

        save_result_path = f"./data/{args.data}"
        if not os.path.exists(save_result_path):
            os.mkdir(save_result_path)

        with open(f"{save_result_path}/{title}.txt", "w", encoding="utf-8") as fw:
            for i in range(0, len(contents) // batch_size + 1):
                prompt = video_info_prompt + process_prompt + "\n".join(contents[i * batch_size: (i + 1) * batch_size])
                response = chatgpt.respond(prompt)
                # system_prompt = video_info_prompt + process_prompt
                # user_prompt = "\n".join(contents[i * batch_size: (i + 1) * batch_size])
                # response = chatgpt.respond(system_prompt, user_prompt)
                if response:
                    fw.write(response)
                    fw.write("\n")
                else:
                    tqdm.write(vname)
                    fw.write(prompt)
                    fw.write("\n")

if __name__ == "__main__":
    main()