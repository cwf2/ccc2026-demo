# 
# import statements
#

import streamlit as st
import os
from osfclient import OSF
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sns

LOCAL_PATH = "data"
OSF_PROJECT = "uz6hg"
N_COMPONENTS = 3


#
# function definitions
#

# download data from OSF
def download_data(osf_project=OSF_PROJECT, local_path=LOCAL_PATH):
    osf = OSF()
    project = osf.project(osf_project)

    for remote_file in project.storage('osfstorage').files:
        if remote_file.name == "tokens.tsv":
            with open(os.path.join(local_path, "tokens.tsv"), "wb") as local_file:
                remote_file.write_to(local_file)


# load tokens
def load_tokens(local_path=LOCAL_PATH):

    token_file = os.path.join(local_path, "tokens.tsv")
    
    # check local copy first
    try:
        tokens = pd.read_csv(token_file, delimiter="\t", dtype=str)
    except:
        # download remote
        download_data(local_path=local_path)
        tokens = pd.read_csv(token_file, delimiter="\t", dtype=str)
    
    # collapse elided and unelided forms
    tokens.loc[tokens["lemma"]=="δʼ", "lemma"] = "δέ"
    tokens.loc[tokens["lemma"]=="τʼ", "lemma"] = "τε"
    tokens.loc[tokens["lemma"]=="ἀλλʼ", "lemma"] = "ἀλλά"
    tokens.loc[tokens["lemma"]=="ἄρʼ", "lemma"] = "ἄρα"
    tokens.loc[tokens["lemma"]=="ἐπʼ", "lemma"] = "ἐπί"
    tokens.loc[tokens["lemma"]=="οὐδʼ", "lemma"] = "οὐδέ"
        
    tokens.loc[tokens["work"]=="Sack of Troy", "pref"] = " "
    
    
    return tokens


# select top features
def top_features(tokens, col="lemma", n=100):

    # corpus-wide count for all non-punctuation features, from most frequent to least
    feat_count = tokens.loc[~(tokens["pos"]=="PUNCT"), col].value_counts()

    # a list of the top lemmas
    top_feat = feat_count.head(n).index

    return top_feat


# train on random samples classified by work, speech/narrative
def run_training(tokens, feature_set, col="lemma", sample_size=1000):
    
    # narratological groups
    nr_mask = tokens["speaker"].isna()
    sp_mask = tokens["speaker"].notna() & tokens["speaker"].ne("Odysseus-Apologue")

    # default group is "other"
    nara_group_ids = pd.Series("oth", index=tokens.index)
    nara_group_ids[nr_mask] = "nar"
    nara_group_ids[sp_mask] = "spk"

    # authorship groups
    auth_group_ids = tokens["work"].str.slice(0,4)

    # combined two-factor group
    group_ids = auth_group_ids + "-" + nara_group_ids

    # sample labels
    sample_ids = pd.Series(index=tokens.index)
    for group in group_ids.unique():
        n_toks = sum(group_ids==group)
        sample_ids.loc[group_ids==group] = np.random.permutation(n_toks) // sample_size
    sample_ids = group_ids + "-" + sample_ids.map(lambda f: f"{int(f):03d}")

    # calculate sample sizes
    tokens_per_sample = tokens.groupby(sample_ids).size()

    # generate feature tallies
    samples = (pd.get_dummies(tokens["lemma"])[feature_set]
        .groupby(sample_ids)
        .agg("sum")
    )

    # normalize as freq / 1000 words
    samples = samples.div(tokens_per_sample, axis=0) * 1000
    
    # pca
    pca_model = PCA(n_components=N_COMPONENTS)
    pca = pd.DataFrame(
        data = pca_model.fit_transform(samples), 
        columns = ["PC1", "PC2", "PC3"],
        index = samples.index,
    )
    
    # fit on nar/spk only
    mask = ~pca.index.str.contains("-oth-")
    X = pca.loc[mask, ["PC1", "PC2"]].values
    y = pca.index[mask].str.contains("spk").astype(int)

    # linear classifier
    clf = LogisticRegression()
    clf.fit(X, y)
    
    # store results
    st.session_state["train_samples"] = samples
    st.session_state["train_pca"] = pca
    st.session_state["pca_model"] = pca_model
    st.session_state["clf"] = clf


def clear_training():
    st.session_state["train_samples"] = None
    st.session_state["train_pca"] = None
    st.session_state["pca_model"] = None
    st.session_state["clf"] = None


# test on rolling window samples
def run_testing(tokens, feature_set, pca_model, clf, col="lemma", window_size=500):
    
    # feature tallies
    samples = (
        pd.get_dummies(tokens[col])[feature_set]
        .rolling(
            window = window_size, 
            center = True,
            min_periods = int(window_size * 0.7))
        .agg("sum")
        .fillna(0)
        .astype(int)
    )
    
    # sample sizes for freq calculations
    tokens_per_sample = (tokens[col]
        .rolling(
            window = window_size,
            center = True,
            min_periods = int(window_size * 0.7)
        )
        .agg("count")
        .fillna(0)
        .astype(int)
    )
    
    # get frequencies
    samples = samples.div(tokens_per_sample, axis=0) * 1000
    
    # project using pca model
    pca_model = st.session_state.get("pca_model")
    pca = pd.DataFrame(
        data = pca_model.transform(samples.dropna()),
        columns = ["PC1", "PC2", "PC3"],
        index = samples.dropna().index,
    )
    
    # calculate speechiness score
    clf = st.session_state.get("clf")
    X = pca[["PC1", "PC2"]].values
    proj = X @ clf.coef_.T + clf.intercept_
    speech_score = pd.DataFrame(dict(
        work = tokens.loc[pca.index, "work"],
        pref = tokens.loc[pca.index, "pref"],
        line = tokens.loc[pca.index, "line"],
        speech_id = tokens.loc[pca.index, "speech_id"],
        score = proj.flatten(), 
        ),
        index = pca.index
    )
    
    # store results
    st.session_state["test_samples"] = samples
    st.session_state["test_pca"] = pca
    st.session_state["speech_score"] = speech_score


def clear_testing():
    st.session_state["test_samples"] = None
    st.session_state["test_pca"] = None
    st.session_state["speech_score"] = None
    

#
# load data
# 

if ("tokens" not in st.session_state) or st.session_state["tokens"] is None:
    st.session_state["tokens"] = load_tokens()
    st.session_state["feature_set"] = top_features(
        tokens = st.session_state["tokens"],
    )
    clear_training()
    clear_testing()
    st.rerun()

#
# training set
#

for k in ["train_samples", "train_pca", "pca_model", "clf"]:
    if (k not in st.session_state) or st.session_state[k] is None:
        run_training(
            tokens = st.session_state["tokens"], 
            feature_set = st.session_state["feature_set"],
        )
        clear_testing()
        st.rerun()

#
# testing set
#

for k in ["test_samples", "test_pca", "speech_score"]:
    if (k not in st.session_state) or st.session_state[k] is None:
        run_testing(
            tokens = st.session_state["tokens"],
            feature_set = st.session_state["feature_set"],
            pca_model = st.session_state["pca_model"],
            clf = st.session_state["clf"],
        )
        st.rerun()

#
# plot
#

speech_score = st.session_state["speech_score"]


work = st.selectbox(label="Work", options=speech_score["work"].unique())
pref = st.selectbox(label="Book", options=speech_score.loc[speech_score["work"]==work, "pref"].unique())
mask = (speech_score["work"] == work) & (speech_score["pref"] == pref)

g = sns.relplot(
    x = speech_score[mask].index,
    y = speech_score[mask].score,
    kind = "line",
)
g.set(
    title = (work + " " + pref),
    xlabel = "line",
)
g.figure.set_size_inches(8, 4)

# set x-tics
xmin = speech_score[mask].index.min()
xmax = speech_score[mask].index.max()
ymin = speech_score["score"].min()
ymax = speech_score["score"].max()

x_ticks = []
x_tick_labels = []
for idx in speech_score[mask].index:
    ln = speech_score.loc[idx, "line"]
    try:
        if int(ln) % 50 == 0:
            if ln not in x_tick_labels:
                x_ticks.append(idx)
                x_tick_labels.append(ln)
    except ValueError:
        continue
g.ax.plot([xmin,xmax], [0,0], "k--", lw=1)
g.set(
    xticks = x_ticks,
    xticklabels = x_tick_labels,
    xlim = (xmin, xmax),
    ylim = (ymin, ymax),
)

speech_data = speech_score.dropna(subset=["speech_id"])
for sid, group in speech_data.groupby("speech_id"):
    g.ax.axvspan(group.index.min(), group.index.max(), alpha=0.15, 
                        color="gray", linewidth=0)

st.pyplot(g)