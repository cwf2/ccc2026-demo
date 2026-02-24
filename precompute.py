#!/usr/bin/env python3
"""
Offline precomputation script.

Run this once on your local machine to generate speech_score.tsv, then upload
that file to your OSF project (uz6hg) before deploying to Streamlit Community
Cloud.

Usage:
    python precompute.py

Requirements (local only, not needed on Streamlit Cloud):
    pip install osfclient pandas numpy scikit-learn
"""

import os
import pandas as pd
import numpy as np
from osfclient import OSF
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

LOCAL_PATH = "data"
OSF_PROJECT = "uz6hg"
N_COMPONENTS = 3


def download_data(osf_project=OSF_PROJECT, local_path=LOCAL_PATH):
    osf = OSF()
    project = osf.project(osf_project)
    for remote_file in project.storage("osfstorage").files:
        if remote_file.name == "tokens.tsv":
            print("Downloading tokens.tsv from OSF...")
            dest = os.path.join(local_path, "tokens.tsv")
            with open(dest, "wb") as f:
                remote_file.write_to(f)
            print(f"  Saved to {dest}")
            return
    raise FileNotFoundError("tokens.tsv not found in OSF project")


def load_tokens(local_path=LOCAL_PATH):
    token_file = os.path.join(local_path, "tokens.tsv")
    if not os.path.exists(token_file):
        download_data(local_path=local_path)
    tokens = pd.read_csv(token_file, delimiter="\t", dtype=str)

    # collapse elided and unelided forms
    tokens.loc[tokens["lemma"] == "δʼ", "lemma"] = "δέ"
    tokens.loc[tokens["lemma"] == "τʼ", "lemma"] = "τε"
    tokens.loc[tokens["lemma"] == "ἀλλʼ", "lemma"] = "ἀλλά"
    tokens.loc[tokens["lemma"] == "ἄρʼ", "lemma"] = "ἄρα"
    tokens.loc[tokens["lemma"] == "ἐπʼ", "lemma"] = "ἐπί"
    tokens.loc[tokens["lemma"] == "οὐδʼ", "lemma"] = "οὐδέ"

    tokens.loc[tokens["work"] == "Sack of Troy", "pref"] = " "
    return tokens


def top_features(tokens, col="lemma", n=100):
    feat_count = tokens.loc[~(tokens["pos"] == "PUNCT"), col].value_counts()
    return feat_count.head(n).index


def compute_speech_score(tokens, feature_set, col="lemma",
                         sample_size=1000, window_size=500):
    # --- training samples ---
    print("  Building training samples...")
    nr_mask = tokens["speaker"].isna()
    sp_mask = tokens["speaker"].notna() & tokens["speaker"].ne("Odysseus-Apologue")

    nara_group_ids = pd.Series("oth", index=tokens.index)
    nara_group_ids[nr_mask] = "nar"
    nara_group_ids[sp_mask] = "spk"

    auth_group_ids = tokens["work"].str.slice(0, 4)
    group_ids = auth_group_ids + "-" + nara_group_ids

    sample_ids = pd.Series(index=tokens.index, dtype=str)
    for group in group_ids.unique():
        n_toks = sum(group_ids == group)
        sample_ids.loc[group_ids == group] = (
            np.random.permutation(n_toks) // sample_size
        )
    sample_ids = group_ids + "-" + sample_ids.map(lambda f: f"{int(f):03d}")

    tokens_per_sample = tokens.groupby(sample_ids).size()
    train_samples = (
        pd.get_dummies(tokens[col])[feature_set]
        .groupby(sample_ids)
        .agg("sum")
    )
    train_samples = train_samples.div(tokens_per_sample, axis=0) * 1000

    # --- PCA ---
    print("  Fitting PCA...")
    pca_model = PCA(n_components=N_COMPONENTS)
    train_pca = pd.DataFrame(
        data=pca_model.fit_transform(train_samples),
        columns=["PC1", "PC2", "PC3"],
        index=train_samples.index,
    )

    # --- logistic regression on nar/spk only ---
    print("  Fitting classifier...")
    mask = ~train_pca.index.str.contains("-oth-")
    X = train_pca.loc[mask, ["PC1", "PC2"]].values
    y = train_pca.index[mask].str.contains("spk").astype(int)
    clf = LogisticRegression()
    clf.fit(X, y)

    # --- rolling window test ---
    print("  Computing rolling window features...")
    feat_dummies = pd.get_dummies(tokens[col])[feature_set]

    roll_sum = (
        feat_dummies
        .rolling(window=window_size, center=True,
                 min_periods=int(window_size * 0.7))
        .agg("sum")
        .fillna(0)
        .astype(int)
    )
    tokens_per_window = (
        tokens[col]
        .rolling(window=window_size, center=True,
                 min_periods=int(window_size * 0.7))
        .agg("count")
        .fillna(0)
        .astype(int)
    )
    test_samples = roll_sum.div(tokens_per_window, axis=0) * 1000
    valid = test_samples.dropna()

    print("  Projecting test samples...")
    test_pca = pd.DataFrame(
        data=pca_model.transform(valid),
        columns=["PC1", "PC2", "PC3"],
        index=valid.index,
    )

    proj = test_pca[["PC1", "PC2"]].values @ clf.coef_.T + clf.intercept_
    speech_score = pd.DataFrame(
        dict(
            work=tokens.loc[test_pca.index, "work"],
            pref=tokens.loc[test_pca.index, "pref"],
            line=tokens.loc[test_pca.index, "line"],
            speech_id=tokens.loc[test_pca.index, "speech_id"],
            score=proj.flatten(),
        ),
        index=test_pca.index,
    )
    return speech_score


if __name__ == "__main__":
    os.makedirs(LOCAL_PATH, exist_ok=True)

    print("Loading tokens...")
    tokens = load_tokens()
    print(f"  {len(tokens):,} tokens loaded.")

    print("Selecting top features...")
    feature_set = top_features(tokens)

    print("Computing speech scores...")
    speech_score = compute_speech_score(tokens, feature_set)

    out_path = os.path.join(LOCAL_PATH, "speech_score.tsv")
    speech_score.to_csv(out_path, sep="\t")
    print(f"\nSaved {len(speech_score):,} rows to {out_path}")

    print("\nNext steps:")
    print("  1. Upload data/speech_score.tsv to OSF project uz6hg")
    print("  2. Deploy streamlit_app.py to Streamlit Community Cloud")
