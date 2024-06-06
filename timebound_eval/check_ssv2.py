"""Evaluates on SSv2."""
import os
import sys
from glob import glob
from tqdm import tqdm

import torch
from languagebind import (
    LanguageBindVideo,
    LanguageBindVideoTokenizer,
    LanguageBindVideoProcessor,
)

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def get_video_path(video_dir, video_id, ext="webm"):
    paths = glob(os.path.join(video_dir, f"*/{video_id}.{ext}"))
    assert len(paths) == 1
    return paths[0]


def text_correct(sim):
    """
    Given a 2x2 similarity matrix, computes text score.

    Based on WinoGround's evaluation code.
    """
    return sim[0, 0] > sim[0, 1] and sim[1, 1] > sim[1, 0]


def video_correct(sim):
    """
    Given a 2x2 similarity matrix, computes video score.

    Based on WinoGround's evaluation code.
    """
    return sim[0, 0] > sim[1, 0] and sim[1, 1] > sim[0, 1]


def group_correct(sim):
    """
    Given a 2x2 similarity matrix, computes group score.

    Based on WinoGround's evaluation code.
    """
    return text_correct(sim) and video_correct(sim)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    pretrained_ckpt = 'LanguageBind/LanguageBind_Video_FT'
    model = LanguageBindVideo.from_pretrained(pretrained_ckpt)
    tokenizer = LanguageBindVideoTokenizer.from_pretrained(pretrained_ckpt)

    # Had to change video backend due to issues with decord
    model.config.vision_config.video_decode_backend = "opencv"

    video_process = LanguageBindVideoProcessor(model.config, tokenizer)
    model.to(device)
    model.eval()

    # Load data
    csv_path = "/scratch/shared/nfs2/piyush/datasets/SSv2/metadata/time_antonyms-validation.csv"
    df = pd.read_csv(csv_path)

    data_dir = "/scratch/shared/beegfs/shared-datasets/SomethingSomething-V2/"
    video_dir = os.path.join(data_dir, "videos")

    iterator = tqdm(df.iterrows(), total=len(df))
    text_corrects = []
    video_corrects = []
    group_corrects = []
    failed = []
    for i, row in iterator:
        row = row.to_dict()
        video_path_x = get_video_path(video_dir, row["id_x"])
        video_path_y = get_video_path(video_dir, row["id_y"])
        label_x = row["label_x"]
        label_y = row["label_y"]
        # iterator.set_postfix(video_path=video_path)

        # Get embeddings
        try:
            data = video_process(
                [video_path_x, video_path_y],
                [label_x, label_y],
                return_tensors='pt',
            )
        
            # Port to device
            for key in data:
                data[key] = data[key].to(device)

            with torch.no_grad():
                out = model(**data)
        except:
            failed.append((row["id_x"], row["id_y"]))
            continue
        
        sim = out.text_embeds @ out.image_embeds.T
        sim = sim.cpu().numpy()
        text_corrects.append(text_correct(sim))
        video_corrects.append(video_correct(sim))
        group_corrects.append(group_correct(sim))
    
    print("Number of failed samples:", len(failed))
    

    # Compute final metrics
    text_corrects = np.array(text_corrects)
    video_corrects = np.array(video_corrects)
    group_corrects = np.array(group_corrects)

    print("Text score:", text_corrects.mean())
    print("Video score:", video_corrects.mean())
    print("Group score:", group_corrects.mean())
