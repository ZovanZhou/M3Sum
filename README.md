# M3Sum

[Title] M3Sum: A Novel Unsupervised Language-guided Video Summarization

## Preparation

1. Clone the repo to your local.
2. Download Python version: 3.7.12
3. Open the shell or cmd in this repo folder. Run this command to install necessary packages.
```cmd
pip install -r requirements.txt
```
4. Change the value **api_key** in the code to your OpenAI api key.
5. Download the videos from this [link](https://pan.baidu.com/s/1Hs_3ofxA9KkenYwA56tV1A), and the extraction code is 1234. Put the "videos" folder into the specific datasets.

## Experiments

1. Inference: You can input the following command to train the model. There are different choices for some hyper-parameters shown in square barckets. The meaning of these parameters are shown in the following tables.

predict_with_chatgpt.py or predict_with_chatgpt_by_CoT.py
|  Parameters | Value | Description|
|  ----  | ----  | ---- |
|  data  | string | Different data for video summarization. You can choose "caption, transcript, transcript2caption" |
| batch_size | int | Number of frames for inference one time |


```cmd
cd ./TVSum
python predict_with_chatgpt.py \
    --data caption \
    --batch_size 120 \

python predict_with_chatgpt.py \
    --data transcript \
    --batch_size 30 \

python predict_with_chatgpt.py \
    --data transcript2caption \
    --batch_size 20 \
```

2. Evaluation: After predicting the scores of the frames, we need to calculate the F1 scores of the prediction results. You can input the following command to evaluate the model. There are different choices for some hyper-parameters shown in square barckets. The meaning of these parameters are shown in the following tables.

evaluate_video_summarization.py
|  Parameters | Value | Description|
|  ----  | ----  | ---- |
|  data  | string | Different data for video summarization. You can choose "caption, transcript, transcript2caption, merge" |
| pcot | int | Whether to use the prediction results of progressive CoT |
| merge_mode | string | Different alignment metrics. You can choose "ppl, bertscore, bleu" |
| threshold | float | The threshold values for alignment |

> P.S. When ''**data**'' parameter is set to "merge", the frame scores of "caption" and "transcript2caption" are merged by different alignment metrics. This mode is the alignment module in our paper.

```cmd
cd ./TVSum

% evaluate the results of standard prompting
python evaluate_video_summarization.py \
    --data caption \

python evaluate_video_summarization.py \
    --data transcript \

python evaluate_video_summarization.py \
    --data transcript2caption \

% evaluate the results of progressive CoT
python evaluate_video_summarization.py \
    --data caption \
    --pcot 1 \

python evaluate_video_summarization.py \
    --data transcript \
    --pcot 1 \

python evaluate_video_summarization.py \
    --data transcript2caption \
    --pcot 1 \

% evaluate the results of standard prompting with alignment
python evaluate_video_summarization.py \
    --data merge \
    --merge_mode bertscore \
    --threshold 0.6 \

% evaluate the results of progressive CoT with alignment
python evaluate_video_summarization.py \
    --data merge \
    --pcot 1 \
    --merge_mode bertscore \
    --threshold 0.6 \
```