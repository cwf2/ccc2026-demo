#
# import statements
#

import streamlit as st
import os
from osfclient import OSF
import pandas as pd
import matplotlib.pyplot as plt

LOCAL_PATH = "data"
OSF_PROJECT = "uz6hg"


#
# function definitions
#

@st.cache_data
def load_speech_score(local_path=LOCAL_PATH):
    """Load pre-computed speech scores, downloading from OSF if needed."""
    score_file = os.path.join(local_path, "speech_score.tsv")

    if not os.path.exists(score_file):
        with st.spinner("Downloading data from OSF..."):
            osf = OSF()
            project = osf.project(OSF_PROJECT)
            os.makedirs(local_path, exist_ok=True)
            for remote_file in project.storage("osfstorage").files:
                if remote_file.name == "speech_score.tsv":
                    with open(score_file, "wb") as f:
                        remote_file.write_to(f)
                    break

    speech_score = pd.read_csv(score_file, delimiter="\t", index_col=0, dtype={
        "work": str,
        "pref": str,
        "line": str,
        "speech_id": str,
        "score": float,
    })
    speech_score.loc[speech_score["work"]=="Sack of Troy", "pref"] = " "
    
    return speech_score


#
# load data
#

speech_score = load_speech_score()

#
# plot
#

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
