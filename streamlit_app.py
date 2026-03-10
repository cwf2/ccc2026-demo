#
# import statements
#

import streamlit as st
import os
from osfclient import OSF
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

LOCAL_PATH = "data"
OSF_PROJECT = "uz6hg"
N_COMPONENTS = 3
SAMPLE_SIZE = 1000
WINDOW_SIZE = 500
N_FEATURES = 100


#
# function definitions
#

@st.cache_data
def load_tokens(local_path=LOCAL_PATH):
    """Load tokens, downloading from OSF if needed."""
    token_file = os.path.join(local_path, "tokens.tsv")

    if not os.path.exists(token_file):
        with st.spinner("Downloading data from OSF..."):
            osf = OSF()
            project = osf.project(OSF_PROJECT)
            os.makedirs(local_path, exist_ok=True)
            for remote_file in project.storage("osfstorage").files:
                if remote_file.name == "tokens.tsv":
                    with open(token_file, "wb") as f:
                        remote_file.write_to(f)
                    break

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


def run_training(tokens, col="lemma",
                         feature_set=None,
                         sample_size=SAMPLE_SIZE):

    # --- feature selection ---
    if feature_set is None:
        feat_count = tokens.loc[~(tokens["pos"] == "PUNCT"), col].value_counts()
        feature_set = feat_count.head(N_FEATURES).index

    # --- training samples ---
    nr_mask = tokens["speaker"].isna()
    sp_mask = tokens["speaker"].notna() & tokens["speaker"].ne("Odysseus-Apologue")

    nara_group_ids = pd.Series("oth", index=tokens.index)
    nara_group_ids[nr_mask] = "nar"
    nara_group_ids[sp_mask] = "spk"

    auth_group_ids = tokens["work"].str.slice(0, 4)
    group_ids = auth_group_ids + "-" + nara_group_ids

    # use seed to reset random number generator every time samples are calculated
    rng = np.random.default_rng(st.session_state["seed"])

    sample_ids = pd.Series(index=tokens.index, dtype=str)
    for group in group_ids.unique():
        n_toks = sum(group_ids == group)
        sample_ids.loc[group_ids == group] = (
            rng.permutation(n_toks) // sample_size
        )
    sample_ids = group_ids + "-" + sample_ids.map(lambda f: f"{int(f):03d}")

    tokens_per_sample = tokens.groupby(sample_ids).size()

    # filter-first: only create dummies for the top-N lemmas
    filtered = tokens[col].where(tokens[col].isin(feature_set))
    train_samples = (
        pd.get_dummies(filtered)
        .reindex(columns=feature_set, fill_value=0)
        .groupby(sample_ids)
        .agg("sum")
    )
    train_samples = train_samples.div(tokens_per_sample, axis=0) * 1000

    # --- PCA ---
    pca_model = PCA(n_components=N_COMPONENTS)
    train_pca = pd.DataFrame(
        data=pca_model.fit_transform(train_samples),
        columns=["PC1", "PC2", "PC3"],
        index=train_samples.index,
    )

    # --- logistic regression on nar/spk only ---
    mask = ~train_pca.index.str.contains("-oth-")
    X = train_pca.loc[mask, ["PC1", "PC2"]].values
    y = train_pca.index[mask].str.contains("spk").astype(int)
    clf = LogisticRegression()
    clf.fit(X, y)
    
    st.session_state["col"] = col
    st.session_state["feature_set"] = feature_set
    st.session_state["auth_group_ids"] = auth_group_ids
    st.session_state["nara_group_ids"] = nara_group_ids
    st.session_state["train_sample_ids"] = sample_ids
    st.session_state["train_samples"] = train_samples
    st.session_state["pca_model"] = pca_model
    st.session_state["train_pca"] = train_pca
    st.session_state["clf"] = clf
    

def compute_speech_score(tokens, window_size=WINDOW_SIZE):
    """Compute rolling speech scores from token table."""
    
    col = st.session_state["col"]
    feature_set = st.session_state["feature_set"]
    pca_model = st.session_state["pca_model"]
    filtered = tokens[col].where(tokens[col].isin(feature_set))
    clf = st.session_state["clf"]

    # --- rolling window test (filter-first optimization) ---
    feat_dummies = (
        pd.get_dummies(filtered)
        .reindex(columns=feature_set, fill_value=0)
    )

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


#
# load data and compute
#

if "seed" not in st.session_state:
    st.session_state["seed"] = 19631123171620

tokens = load_tokens()

feat_count = tokens.loc[~(tokens["pos"] == "PUNCT"), "lemma"].value_counts().head(N_FEATURES)

with st.sidebar:
    st.header("Settings")

    seed_input = st.number_input(
        "Seed (-1 = random)",
        value=-1,
        min_value=-1,
        max_value=2**128 - 1,
        step=1,
    )
    st.caption(f"Current seed: {st.session_state['seed']}")

    if st.button("Recalculate"):
        editor_state = st.session_state.get("feature_editor", {})
        edited_rows = editor_state.get("edited_rows", {})
        selected = [
            lemma
            for i, lemma in enumerate(feat_count.index)
            if edited_rows.get(i, {}).get("include", True)
        ]
        st.session_state["selected_features"] = selected
        actual_seed = int(np.random.randint(0, 2**31)) if seed_input == -1 else seed_input
        st.session_state["seed"] = actual_seed
        for key in ["col", "speech_score"]:
            st.session_state.pop(key, None)
        st.rerun()

    feat_df = pd.DataFrame({
        "lemma": feat_count.index,
        "frequency": feat_count.values,
        "include": True,
    })
    st.data_editor(
        feat_df,
        key="feature_editor",
        column_config={
            "include": st.column_config.CheckboxColumn("Include"),
        },
        use_container_width=True,
        height=400,
        hide_index=True,
        disabled=["lemma", "frequency"],
    )

if "col" not in st.session_state:
    with st.spinner("Training model..."):
        feature_set = st.session_state.get("selected_features")
        if feature_set is not None:
            feature_set = pd.Index(feature_set)
        run_training(tokens, feature_set=feature_set)

if "speech_score" not in st.session_state:
    with st.spinner("Computing speech scores..."):
        st.session_state["speech_score"] = compute_speech_score(tokens)

speech_score = st.session_state["speech_score"]

#
# plot
#

tab_model, tab_view = st.tabs(["Model", "View"])

with tab_model:
        
    # plot
    auth_labels = (st.session_state["auth_group_ids"]
                    .groupby(st.session_state["train_sample_ids"])
                    .agg("first")
                    .values)
    nara_labels = (st.session_state["nara_group_ids"]
                    .groupby(st.session_state["train_sample_ids"])
                    .agg("first")
                    .values)

    g = sns.relplot(data=st.session_state["train_pca"],
        x = "PC1",
        y = "PC2",
        hue = auth_labels,
        style = nara_labels,
    )
    
    # add linear decision boundary to plot
    
    # get the axis limits from the plot
    xlim = g.ax.get_xlim()
    
    # solve for y at each x endpoint: coef[0]*x + coef[1]*y + intercept = 0
    #   => y = -(coef[0]*x + intercept) / coef[1]
    w = st.session_state["clf"].coef_[0]
    b = st.session_state["clf"].intercept_[0]
    xs = np.array(xlim)
    ys = -(w[0] * xs + b) / w[1]

    g.ax.plot(xs, ys, "k--", linewidth=1)
    g.ax.set_xlim(xlim)  # restore limits so the line doesn't expand the plot
    
    st.pyplot(g.figure)
    

with tab_view:
    work = st.selectbox(label="Work", options=speech_score["work"].unique())
    pref = st.selectbox(
        label="Book",
        options=speech_score.loc[speech_score["work"] == work, "pref"].unique(),
    )

    mask = (speech_score["work"] == work) & (speech_score["pref"] == pref)
    data = speech_score[mask]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data.index, data["score"])
    ax.set_title(work + " " + pref)
    ax.set_xlabel("line")

    xmin = data.index.min()
    xmax = data.index.max()
    ymin = speech_score["score"].min()
    ymax = speech_score["score"].max()

    x_ticks = []
    x_tick_labels = []
    for idx in data.index:
        ln = data.loc[idx, "line"]
        try:
            if int(ln) % 50 == 0:
                if ln not in x_tick_labels:
                    x_ticks.append(idx)
                    x_tick_labels.append(ln)
        except ValueError:
            continue

    ax.plot([xmin, xmax], [0, 0], "k--", lw=1)
    ax.set(
        xticks=x_ticks,
        xticklabels=x_tick_labels,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
    )

    speech_data = data.dropna(subset=["speech_id"])
    for sid, group in speech_data.groupby("speech_id"):
        ax.axvspan(group.index.min(), group.index.max(), alpha=0.15,
                   color="gray", linewidth=0)

    st.pyplot(fig)
