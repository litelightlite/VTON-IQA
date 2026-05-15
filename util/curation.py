import os
import json
import hashlib
from collections import Counter
from typing import Literal, Optional, List, Dict, Union, Tuple
import krippendorff
import numpy as np
import pandas as pd
from util.commons import VTONQBENCH_ROOT, VTON_MODEL_IDS 

ANSWER_COLS = [str(i) for i in range(1, 51)]
DUMMIES = {
    "v1": {
        5: "cc+lower_body+dc+images+050517_0_050741_1_ladivton.png",
        15: "cc+upper_body+dc+images+049166_0_048954_1_ladivton.png",
        25: "cc+lower_body+dc+images+050300_0_050256_1_ladivton.png",
        35: "cc+upper_body+vitonhd+images+14520_00_13234_00_gpvton.png",
        45: "cc+dresses+dc+images+053459_0_053544_1_ladivton.png"
        },
    "v2": {
        5: "cc+dresses+dc+051994_0_053552_1_nanobanana_051994_1_any2any.png",
        15: "cc+lower_body+dc+050244_0_050889_1_nanobanana_050244_1_qwen_edit.png",
        25: "cc+upper_body+dc+050182_0_049952_1_nanobanana_050182_1_sdviton.png",
        35: "cc+upper_body+dc+050182_0_049952_1_nanobanana_050182_1_sdviton.png",
        45: "cc+upper_body+dc+050173_0_048723_1_nanobanana_050173_1_ladi.png"  
    }
    }

ATTENTION_EXPECTED = {
    "v1": {
        "5": {"expected": 0, "tol": 0},  
        #"15": {"expected": 0, "tol": 0},
        "25": {"expected": 0, "tol": 0},
        "35": {"expected": 0, "tol": 1}, 
        "45": {"expected": 0, "tol": 1}, 
        },
    "v2": {
    "5": {"expected": 0, "tol": 0},  
    "15": {"expected": 0, "tol": 0},
    "25": {"expected": 0, "tol": 0},
    "35": {"expected": 0, "tol": 1}, 
    "45": {"expected": 0, "tol": 1}, 
    }}

def longest_run_length(values: np.ndarray) -> int:
    if len(values) == 0:
        return 0
    max_run = 1
    cur_run = 1
    for i in range(1, len(values)):
        if pd.isna(values[i]) or pd.isna(values[i-1]):
            cur_run = 1
            continue
        if values[i] == values[i-1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    return max_run

def row_longstring_ratio(row: pd.Series) -> float:
    arr = row.values
    return longest_run_length(arr) / len(arr)


def attention_pass_row(row: pd.Series, spec: dict) -> bool:
    for col, cfg in spec.items():
        exp = cfg["expected"]
        tol = cfg.get("tol", 0)
        v = row.get(col, np.nan)
        if pd.isna(v) or abs(v - exp) > tol:
            return False
    return True

def convert_label_id_to_score(label_id: Literal[0, 1, 2]) -> float:
    if label_id == 0:
        return 1.0
    elif label_id == 1:
        return 2.0
    elif label_id == 2:
        return 3.0
    else:
        return np.nan

THETA_MAJ = 0.60
N_MIN     = 10
TAU       = 0.20

def majority_with_margin(vals, theta=THETA_MAJ):
    vs = [v for v in vals if pd.notna(v)]
    if len(vs) == 0: 
        return None, 0.0
    c = Counter(vs)
    maj_label, maj_cnt = c.most_common(1)[0]
    ratio = maj_cnt / len(vs)
    if ratio >= theta:
        return maj_label, ratio
    return None, ratio

def flag_trolls_one_form(df_form, theta=THETA_MAJ, n_min=N_MIN, tau=TAU):

    raters = df_form['response_id'].tolist()
    M = len(raters)
    Q = ANSWER_COLS

    N_eff = {r: 0 for r in raters}
    N_opp = {r: 0 for r in raters}

    for q in Q:
        col = df_form[q].astype('float').values

        for i, r in enumerate(raters):
            others = np.delete(col, i)
            maj, ratio = majority_with_margin(others, theta=theta)
            if maj is None:
                continue

            v = col[i]
            if np.isnan(v):
                continue

            N_eff[r] += 1

            if (maj == 2 and v == 0) or (maj == 0 and v == 2):
                N_opp[r] += 1

    rows = []
    for r in raters:
        ne = N_eff[r]
        no = N_opp[r]
        p  = (no / ne) if ne > 0 else np.nan
        flagged = (ne >= n_min) and (p is not np.nan) and (p >= tau)
        rows.append({"response_id": r, "N_eff": ne, "N_opp": no, "p_opp": p, "flagged": flagged})

    return pd.DataFrame(rows)


def calc_bootstrapped_krippendorff_alpha(
    df_answer: pd.DataFrame, 
    key_col: str = "form_id",
    target_cols: List[str] = [f"{i}" for i in range(1,51)],
    n_bootstrap: int = 10, 
    n_sample: int = 3
    ):
    
    alphas_bootstrapped = {}
    for _, df in df_answer.groupby(key_col):
        reliability_data = df[target_cols].values   
        tmp_alphas = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(reliability_data.shape[0], n_sample, replace=False)
            alpha = krippendorff.alpha(reliability_data=reliability_data[idx, :], level_of_measurement="ordinal")
            tmp_alphas.append(alpha)
        alphas_bootstrapped[df["form_id"].values[0]] = np.mean(tmp_alphas)

    return alphas_bootstrapped


def load_reward_dataset_as_df(
    curation: bool = True,
    drop_dummies: bool = True,
    min_vote_num: int = 3,
    min_alpha: Optional[float] = None,
    version: Literal["v1", "v2"] = "v2"
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[int, float]]]:
    
    df_answer = pd.read_csv(os.path.join(VTONQBENCH_ROOT, "answers.csv"))

    if curation:
        
        longstring_ratio = df_answer[ANSWER_COLS].apply(lambda r: row_longstring_ratio(r), axis=1)
        within_var = df_answer[ANSWER_COLS].var(axis=1, skipna=True)
        att_pass = df_answer.apply(lambda r: attention_pass_row(r, ATTENTION_EXPECTED[version]), axis=1)

        MAX_LONGSTRING_RATIO = 0.8
        MIN_WITHIN_VAR = 0.30

        flag_longstring = longstring_ratio > MAX_LONGSTRING_RATIO
        flag_low_var = within_var < MIN_WITHIN_VAR
        flag_att_fail = ~att_pass

        exclude_flag = flag_longstring | flag_low_var | flag_att_fail

        df_answer = df_answer.loc[~exclude_flag].reset_index(drop=True)

        results = []
        for fid, sub in df_answer.groupby("form_id"):
            out = flag_trolls_one_form(sub)
            out["form_id"] = fid
            results.append(out)

        troll_df = pd.concat(results, ignore_index=True)
        df_answer = df_answer[~df_answer["response_id"].isin(troll_df[troll_df["flagged"]]["response_id"].values)]
    
    arr = df_answer["form_id"].value_counts()
    df_answer = df_answer[df_answer["form_id"].isin(arr[min_vote_num <= arr].index)]
    
    if min_alpha is not None:
        alphas = calc_bootstrapped_krippendorff_alpha(df_answer, n_bootstrap=10, n_sample=3)
        df_answer = df_answer[~df_answer["form_id"].isin([k for k, v in alphas.items() if v < min_alpha])]
    
    df_answer = df_answer.melt(
        id_vars=["form_id", "answer_timestamp", "gender", "age", "experience", "response_id"], 
        value_vars=[str(i) for i in range(1, 51)],
        var_name="answer_count",
        value_name="label_id"
    ).astype({"answer_count": int})

    df_image =  pd.read_csv(os.path.join(VTONQBENCH_ROOT, "images.csv"))
    
    df_answer = df_answer[["form_id", "answer_count", "response_id", "label_id"]].merge(df_image, on=["form_id", "answer_count"], how="inner").sort_values(by=["form_id", "answer_count"], ascending=[True, True]).reset_index(drop=True)
    df_answer = df_answer.rename(columns={"answer_count": "problem_id"})
        
    df_answer["score"] = df_answer.apply(lambda r: convert_label_id_to_score(r["label_id"]) , axis=1)

    df_reward = df_answer.groupby(["form_id", "problem_id"])[["score"]].mean().reset_index().rename(columns={"score": "reward"})
    
    df_meta = df_answer[["form_id", "problem_id", "garment_type", "dataset_type", "vton_model_id", "person_path", "garment_path", "ref_person_path", "vton_path"]].drop_duplicates().reset_index(drop=True)
    df_meta = df_meta.merge(df_answer.groupby(["form_id", "problem_id"]).size().reset_index(name="n_annotation"), on=["form_id", "problem_id"], how="left")
    df_reward = df_reward.merge(df_meta, on=["form_id", "problem_id"], how="left")
    
    df_vote = df_answer.groupby(["form_id", "problem_id"])[["response_id", "label_id"]].agg(list).rename(columns={"response_id": "annotator_ids", "label_id": "answer_ids"})
    df_reward = df_reward.merge(df_vote, on=["form_id", "problem_id"], how="left")    

    df_count = df_answer.groupby(["form_id", "problem_id", "label_id"]).size().reset_index(name="count")

    df_pivot = df_count.pivot_table(
        index=["form_id", "problem_id"],
        columns="label_id",
        values="count",
        fill_value=0
    ).reset_index()

    df_pivot["votes"] = df_pivot[[0, 1, 2]].apply(lambda row: [int(x) for x in row.values], axis=1)

    df_pivot = df_pivot[["form_id", "problem_id", "votes"]]
            
    df_reward = df_reward.merge(df_pivot, on=["form_id", "problem_id"], how="left")
    
    if drop_dummies:
        df_reward = df_reward[df_reward["vton_model_id"].isin(VTON_MODEL_IDS)].reset_index(drop=True)
    
    def stable_hash(row):
        s = json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    df_reward["id"] = df_reward.apply(stable_hash, axis=1)
    
    if min_alpha is not None:
        return df_reward, alphas
    else:
        return df_reward
