import warnings
warnings.filterwarnings("ignore")

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


import sys
sys.path.append("../TimeBound.v1/")
import video_language.datasets.epic as epic
import video_language.tasks.metrics as metrics
import shared.utils as su


def gather_features(df_pair, df_main):
    video_features_path = ".languagebind_epic_video_features.pt"
    text_features_path = ".languagebind_epic_text_features.pt"

    if os.path.exists(video_features_path):
        video_features = torch.load(video_features_path)
        text_features = torch.load(text_features_path)
    else:
        # Step 1: Compute video features for all IDs
        ids_a = df_pair.id_a.unique()
        ids_b = df_pair.id_b.unique()
        ids = np.unique(np.concatenate([ids_a, ids_b]))

        iterator = su.log.tqdm_iterator(ids, desc="Computing features")
        video_features = dict()
        text_features = dict()
        for narration_id in iterator:
            video_id = "_".join(narration_id.split("_")[:2])
            row = df_main[df_main.narration_id == narration_id].iloc[0].to_dict()
            label = f"{row['verb']} {row['noun']}"
            st = row["start_sec"]
            et = row["stop_sec"]
            video_path = epic.get_video_path_basic(video_id)
            assert os.path.exists(video_path), \
                f"Video path {video_path} does not exist"

            try:
                data = video_process(
                    images=[video_path],
                    text=[label],
                    starts=[st],
                    ends=[et],
                    return_tensors='pt',
                )
                for key in data:
                    data[key] = data[key].to(device)
                with torch.no_grad():
                    out = model(**data)
                text_feat = out.text_embeds.cpu()[0].numpy()
                video_feat = out.image_embeds.cpu()[0].numpy()
                video_features[narration_id] = video_feat
                text_features[narration_id] = text_feat
            except:
                print("Failed to process video", narration_id)
                continue

        # Save video features
        torch.save(video_features, video_features_path)
        torch.save(text_features, text_features_path)

    return video_features, text_features


def gather_results(df_pair, video_features, text_features):
    N = len(df_pair)
    results = []
    iterator = su.log.tqdm_iterator(range(N), desc="Evaluating")
    for i in iterator:
        row = df_pair.iloc[i].to_dict()
        id_a = row["id_a"]
        id_b = row["id_b"]
        if id_a not in video_features or id_b not in video_features:
            continue

        vid_a = video_features[id_a]
        vid_b = video_features[id_b]
        txt_a = text_features[id_a]
        txt_b = text_features[id_b]
        vid = np.stack([vid_a, vid_b])
        txt = np.stack([txt_a, txt_b])

        sim = (vid @ txt.T)
        r = metrics.get_scores(sim)
        results.append(r)
    results = pd.DataFrame(results)
    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    pretrained_ckpt = 'LanguageBind/LanguageBind_Video_FT'
    model = LanguageBindVideo.from_pretrained(pretrained_ckpt)
    tokenizer = LanguageBindVideoTokenizer.from_pretrained(pretrained_ckpt)

    # Had to change video backend due to issues with decord
    # model.config.vision_config.video_decode_backend = "opencv"
    model.config.vision_config.video_decode_backend = "decord"

    video_process = LanguageBindVideoProcessor(model.config, tokenizer)
    model.to(device)
    model.eval()

    # Load CSVs
    paths = epic.get_paths()
    split = "validation"
    df_main = epic.load_main_csv(paths, split)
    _, df_time = epic.load_time_csv(paths, split)
    df_pair = epic.load_pair_csv(paths, split)

    # Debug
    debug = False
    if debug:
        narration_id = "P02_12_312"
        video_id = "_".join(narration_id.split("_")[:2])
        row = df_main[df_main.narration_id == narration_id].iloc[0].to_dict()
        label = f"{row['verb']} {row['noun']}"
        st = row["start_sec"]
        et = row["stop_sec"]
        video_path = epic.get_video_path_basic(video_id)
        assert os.path.exists(video_path)
        data = video_process(
            images=[video_path],
            text=[label],
            starts=[st],
            ends=[et],
            return_tensors='pt',
        )
        for key in data:
            data[key] = data[key].to(device)
        with torch.no_grad():
            out = model(**data)
        import ipdb; ipdb.set_trace()

    # Get features
    video_features, text_features = gather_features(df_pair, df_main)

    # Get results
    results = gather_results(df_pair, video_features, text_features)

    # Save results
    os.makedirs("results", exist_ok=True)
    print(results.mean())
    results.mean().to_csv("results/scores_epic_languagebind.csv")
