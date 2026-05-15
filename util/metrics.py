import copy
import tqdm
import math
import random
import itertools
import statistics
from typing import Literal, Tuple, List, Dict, Optional, Any, Union
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import brier_score_loss, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import torch
import torch.nn.functional as F
from util.visualize import plot_calibration_curve, plot_probability_alignment_curve, plot_pairwise_acc_alignment_curve, plot_score_distribution
from dataset import convert_label_id_to_score, BaseRewardDataset


def bce_with_logits_by_group(
    prediction: torch.Tensor,   # [N] or [1,N]
    answer: torch.Tensor,       # [N] or [1,N]
    group_id: torch.Tensor,     # [N] or [1,N]
    model_temperature: float,
    human_temperature: float = 0.65,
    reduction: str = "mean",    # "mean" | "sum" | "none"
) -> Tuple[torch.Tensor, int]:
    """
    各 group_id 内の (i<j) 全ペアについて
        logit = (pred_i - pred_j)
        label = sigmoid((ans_i - ans_j) / t)
    として BCE-with-logits を計算する。
    tie (ans_i == ans_j) も除外せず学習に含める。
    戻り値: (loss, 使用ペア数)
    """
    pred = prediction.flatten()
    ans  = answer.flatten()
    gid  = group_id.flatten()

    device = pred.device
    all_losses = []

    for g in gid.unique(sorted=False):
        idx = (gid == g).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue

        pairs = torch.combinations(idx, r=2)  # [P,2]
        i, j = pairs[:, 0], pairs[:, 1]

        t = torch.sigmoid((ans[i] - ans[j]) / human_temperature)  # [P] ∈ (0,1)

        logit = (pred[i] - pred[j]) / model_temperature

        # BCE with logits
        loss_ij = F.binary_cross_entropy_with_logits(logit, t, reduction="none")
        all_losses.append(loss_ij)

    if len(all_losses) == 0:
        if reduction == "none":
            return torch.empty(0, device=device), 0
        else:
            return torch.tensor(0.0, device=device), 0

    losses = torch.cat(all_losses, dim=0)  # [総ペア数]

    if reduction == "mean":
        return losses.mean(), losses.numel()
    elif reduction == "sum":
        return losses.sum(), losses.numel()
    elif reduction == "none":
        return losses, losses.numel()
    else:
        raise ValueError("reduction must be 'mean' | 'sum' | 'none'")


def bootstrap_annotation(
    reward_dataset_path: str,
    scaler: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]] = None,
    n_bootstrap: int = 10,
    ) -> List[List[Dict[str, Any]]]:
    
    def to_raw_result(g1: List[Dict[str, Any]], g2: List[Dict[str, Any]]):
        g1d = pd.DataFrame(g1).groupby(["person", "garment"]).agg(list).reset_index().to_dict(orient="records")
        g2d = pd.DataFrame(g2).groupby(["person", "garment"]).agg(list).reset_index().to_dict(orient="records")
        test_results_raw = []
        for human, pred in zip(g1d, g2d):
            human["prediction"] = copy.deepcopy(pred["reward"])
            test_results_raw.append(human)
        return test_results_raw
    
    dataset = BaseRewardDataset(reward_dataset_path=reward_dataset_path).dataset
    
    bootstrapped_data = []
    for _ in tqdm.tqdm(range(n_bootstrap), total=n_bootstrap, desc="Bootstrapping human rewards"):
        g1, g2 = [], []
        for form_id in sorted(list(set([data["form_id"] for data in dataset]))):
            targets = [data for data in dataset if data["form_id"]==form_id]
            ids = [i for i in range(sum(targets[0]["votes"]))]
            random.shuffle(ids)
            for target in targets:
                target_g1 = copy.deepcopy(target)
                target_g1["reward"] = statistics.mean([convert_label_id_to_score(target["answer_ids"][i]) for i in ids[:len(ids)//2]])
                target_g1["answer_ids"] = [target["answer_ids"][i] for i in ids[:len(ids)//2]]
                target_g1["annotator_ids"] = [target["annotator_ids"][i] for i in ids[:len(ids)//2]]
                target_g1["votes"] = [len([answer_id for answer_id in target_g1["answer_ids"] if answer_id==i]) for i in range(3)]
                g1.append(target_g1)

                target_g2 = copy.deepcopy(target)
                target_g2["reward"] = statistics.mean([convert_label_id_to_score(target["answer_ids"][i]) for i in ids[len(ids)//2:]])
                target_g2["answer_ids"] = [target["answer_ids"][i] for i in ids[len(ids)//2:]]
                target_g2["annotator_ids"] = [target["annotator_ids"][i] for i in ids[len(ids)//2:]]
                target_g2["votes"] = [len([answer_id for answer_id in target_g2["answer_ids"] if answer_id==i]) for i in range(3)]
                g2.append(target_g2)

        if scaler is not None:
            g1, g2 = pd.DataFrame(g1), pd.DataFrame(g2)
            g1["reward"], g2["reward"] = scaler.fit_transform(g1[["reward"]]), scaler.fit_transform(g2[["reward"]])
            g1, g2 = g1.to_dict(orient="records"), g2.to_dict(orient="records")
        else:
            g1, g2 = pd.DataFrame(g1), pd.DataFrame(g2)
            g1, g2 = g1.to_dict(orient="records"), g2.to_dict(orient="records")
        
        bootstrapped_data.append(to_raw_result(g1, g2))      
    
    return bootstrapped_data


def evaluate_bootstrapped_ranking_corr(
    bootstrapped_data: List[List[Dict[str, Any]]]
    ) -> Tuple[float, float, float, float]:

    pearsonr_seq, spearmanr_seq = [], []
    for test_results_raw in bootstrapped_data:
        r1 = list(itertools.chain.from_iterable([data["reward"] for data in test_results_raw]))
        r2 = list(itertools.chain.from_iterable([data["prediction"] for data in test_results_raw]))
        pearsonr_seq.append(pearsonr(r1, r2)[0]), spearmanr_seq.append(spearmanr(r1, r2)[0])
    
    return statistics.mean(pearsonr_seq), statistics.stdev(pearsonr_seq), statistics.mean(spearmanr_seq), statistics.stdev(spearmanr_seq) 


def evaluate_bootstrapped_r2score(
    bootstrapped_data: List[List[Dict[str, Any]]]
    ) -> Tuple[float, float]:

    r2_seq = []
    for test_results_raw in bootstrapped_data:
        r1 = list(itertools.chain.from_iterable([data["reward"] for data in test_results_raw]))
        r2 = list(itertools.chain.from_iterable([data["prediction"] for data in test_results_raw]))
        r2_seq.append(r2_score(r1, r2))
    
    return statistics.mean(r2_seq), statistics.stdev(r2_seq)


def evaluate_bootstrapped_pairwise_accuracy(
    bootstrapped_data: List[List[Dict[str, Any]]]
    ) -> Tuple[float, float, float, float, float, float, float, float]:
    
    micro_acc_seq, macro_acc_seq, label_tie_rate_seq, pred_tie_rate_seq = [], [], [], []
    for test_results_raw in bootstrapped_data:
        micro_acc, macro_acc, label_tie_rate, pred_tie_rate = evaluate_micro_macro_pairwise_accuracy(test_results_raw, eps_label=1e-8, eps_pred=1e-4)
        micro_acc_seq.append(micro_acc), macro_acc_seq.append(macro_acc)
        label_tie_rate_seq.append(label_tie_rate), pred_tie_rate_seq.append(pred_tie_rate)
    
    return statistics.mean(micro_acc_seq), statistics.stdev(micro_acc_seq), statistics.mean(macro_acc_seq), statistics.stdev(macro_acc_seq), statistics.mean(label_tie_rate_seq), statistics.stdev(label_tie_rate_seq), statistics.mean(pred_tie_rate_seq), statistics.stdev(pred_tie_rate_seq)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))  

        
def groupby_difficulty(
    test_results_raw: List[Dict[str, Any]],
    model_temperature: float,
    human_temperature: float = 0.65,
    interval: float=0.05
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
    
    def to_difficulty_analysis_format(
        test_results_raw: List[Dict[str, Any]], 
        model_temperature: float,
        human_temperature: float = 0.9,
        ) -> List[Dict[str, Any]]: 
        
        content = []
        for data in test_results_raw:
            for i, j in itertools.combinations(range(len(data["reward"])), 2):
                record = {
                    "person": data["person"], 
                    "garment": data["garment"],
                    "vton_x": data["vton"][i],
                    "vton_y": data["vton"][j],
                    "vton_model_id_x": data["vton_model_id"][i],
                    "vton_model_id_y": data["vton_model_id"][j],
                    "reward_x": data["reward"][i],
                    "reward_y": data["reward"][j],
                    "prediction_x": data["prediction"][i],
                    "prediction_y": data["prediction"][j],
                    "votes_x": data["votes"][i],
                    "votes_y": data["votes"][j],
                    "reward_proba_gt_winner_win": max(sigmoid((data["reward"][i]-data["reward"][j]) / human_temperature), 1.0-sigmoid((data["reward"][i]-data["reward"][j]) / human_temperature)),
                    "prediction_proba_gt_winner_win": sigmoid((data["prediction"][i]-data["prediction"][j]) / model_temperature) if data["reward"][j] < data["reward"][i] else 1.0 - sigmoid((data["prediction"][i]-data["prediction"][j]) / model_temperature),
                    "garment_type": data["garment_type"][0],
                    }
                content.append(record)
                
        return content
    
    content = to_difficulty_analysis_format(test_results_raw, model_temperature, human_temperature)
    
    lbs = [round(i, 3) for i in np.arange(0.5, 1.0, interval)]
    content_gb_diff = []
    for lb in lbs:
        filtered_content = [data for data in content if (lb < data["reward_proba_gt_winner_win"]) and (data["reward_proba_gt_winner_win"] <= lb+interval)]
        if 0 < len(filtered_content):
            p_mean = statistics.mean([data["reward_proba_gt_winner_win"] for data in filtered_content])
            content_gb_diff.append({"p_min": lb, "p_max": lb+interval, "p_mean": p_mean, "content": filtered_content})
    
    return content_gb_diff, lbs


def evaluate_probability_alignment(
    test_results_raw: List[Dict[str, Any]],
    model_temperature: float,
    human_temperature: float = 0.65,
    interval: float=0.05
    ) -> Tuple[List[float], List[float], List[float]]:
    
    content_gb_diff, lbs = groupby_difficulty(test_results_raw, model_temperature, human_temperature, interval)
    palign_means, palign_stdevs, n_instances = [], [], []
    for i in range(len(content_gb_diff)):
        if 0 < len(content_gb_diff[i]["content"]):
            probas = [data["prediction_proba_gt_winner_win"] for data in content_gb_diff[i]["content"]]
            palign_means.append(statistics.mean(probas)), palign_stdevs.append(statistics.stdev(probas)), n_instances.append(len(probas))
    
    return palign_means, palign_stdevs, lbs, n_instances


def evaluate_bootstrapped_probability_alignment(
    bootstrapped_data: List[List[Dict[str, Any]]],
    model_temperature: float,
    human_temperature: float = 0.65,
    interval: float=0.05
    ) -> Tuple[List[float], List[List[float]], List[List[float]], List[float]]:
    
    palign_means, palign_stds = [], []
    for test_results_raw in bootstrapped_data:
        means, stdevs, lbs = evaluate_probability_alignment(test_results_raw, model_temperature, human_temperature, interval)
        palign_means.append(means), palign_stds.append(stdevs)

    bootstrapped_palign_mean = np.mean(np.array(palign_means), axis=0).tolist()
    
    return bootstrapped_palign_mean, palign_means, palign_stds, lbs


def evaluate_pairwise_acc_alignment(
    test_results_raw: List[Dict[str, Any]],
    model_temperature: float,
    human_temperature: float = 0.65,
    interval: float = 0.05,
    eps_reward: float = 1e-8,
    eps_pred: float = 1e-4,
    ) -> Tuple[List[float], List[float]]:
    """
    Returns:
      accs: List[float]   # 各 difficulty bin の pairwise accuracy
      lbs:  List[float]   # 各 bin の label
    Scoring:
      - reward diff が小さい (|dx| <= eps_reward) → 無視
      - prediction diff が小さい (|dy| <= eps_pred) → 0.5 点
      - 符号一致 → 1.0
      - 符号不一致 → 0.0
    """

    content_gb_diff, lbs = groupby_difficulty(
        test_results_raw,
        model_temperature,
        human_temperature,
        interval,
    )

    accs = []

    for i in range(len(content_gb_diff)):
        contents = content_gb_diff[i]["content"]
        if len(contents) == 0:
            continue

        score = 0.0
        total = 0

        for data in contents:
            dx = data["reward_x"] - data["reward_y"]
            dy = data["prediction_x"] - data["prediction_y"]

            # reward diff が小さい → 無視
            if abs(dx) <= eps_reward or math.isnan(dx):
                continue

            # prediction tie → 0.5 点
            if abs(dy) <= eps_pred or math.isnan(dy):
                score += 0.5
                total += 1
                continue

            # 通常の符号一致判定
            total += 1
            if dx * dy > 0:
                score += 1.0
            # else: 0 点

        if total > 0:
            accs.append(score / total)

    return accs, lbs


def evaluate_bootstrapped_pairwise_acc_alignment(
    bootstrapped_data: List[List[Dict[str, Any]]],
    model_temperature: float,
    human_temperature: float = 0.65,
    interval: float = 0.05,
    eps_reward: float = 1e-8,
    eps_pred: float = 1e-4,
    ) -> Tuple[List[float], List[List[float]], List[List[float]], List[float]]:
    
    accs_seq = []
    for test_results_raw in bootstrapped_data:
        accs, lbs = evaluate_pairwise_acc_alignment(test_results_raw, model_temperature, human_temperature, interval, eps_reward, eps_pred)
        accs_seq.append(accs)

    bootstrapped_accs = np.mean(np.array(accs_seq), axis=0).tolist()
    
    return bootstrapped_accs, accs_seq, lbs

def evaluate_expected_calibration_error(
    test_results_raw: List[Dict[str, Any]],
    model_temperature: float,
    n_bins: int = 10
    ) -> float:
    
    probas, targets = [], []
    for r in test_results_raw:
        if 1 < len(r["prediction"]): 
            for (i, si), (j, sj) in itertools.combinations(enumerate(r["prediction"]), 2):
                probas.append(sigmoid((si-sj) / model_temperature))
                targets.append(0 if r["reward"][i] < r["reward"][j] else 1)
                
    probas, targets = np.array(probas), np.array(targets)
    bins = np.linspace(0, 1, n_bins+1)
    binids = np.digitize(probas, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            acc = np.mean(targets[mask])
            conf = np.mean(probas[mask])
            ece += np.sum(mask) * abs(acc - conf)
    ece /= len(targets)
    ece = float(ece)
    
    return ece


def evaluate_brier_score(
    test_results_raw: List[Dict[str, Any]],
    model_temperature: float,
    ) -> float:
    
    probas, targets = [], []
    for r in test_results_raw:
        if 1 < len(r["prediction"]): 
            for (i, si), (j, sj) in itertools.combinations(enumerate(r["prediction"]), 2):
                probas.append(sigmoid((si-sj) / model_temperature))
                targets.append(0 if r["reward"][i] < r["reward"][j] else 1)
                
    probas, targets = np.array(probas), np.array(targets)
    brier = brier_score_loss(targets, probas)
    
    return brier


def evaluate_micro_macro_pairwise_accuracy(
    test_results_raw: List[Dict[str, Any]],
    eps_label: float = 1e-8,
    eps_pred: float = 1e-4,
    prediction_key: Literal["prediction", "ssim", "lpips"] = "prediction"
    ) -> Tuple[float, float, float, float]:
    """
    Returns:
      micro_acc
      macro_acc
      label_tie_rate
      pred_tie_rate
    """
    def pairwise_counts(
        labels: List[float],
        predictions: List[float],
        eps_label: float = 1e-8,
        eps_pred: float = 1e-4,
    ) -> Tuple[float, int, int, int]:
        
        def is_tie(x: float, y: float, eps: float) -> bool:
            if math.isnan(x) or math.isnan(y):
                return True
            return abs(x - y) <= eps
        
        assert len(labels) == len(predictions)

        correct = 0.0
        total = 0
        label_ties = 0
        pred_ties = 0

        for i, j in itertools.combinations(range(len(labels)), 2):
            ai, aj = float(labels[i]), float(labels[j])

            # label tie
            if is_tie(ai, aj, eps_label):
                label_ties += 1
                continue

            bi, bj = float(predictions[i]), float(predictions[j])

            # prediction tie
            if is_tie(bi, bj, eps_pred):
                pred_ties += 1
                correct += 0.5
                total += 1
                continue

            total += 1
            if (ai < aj and bi < bj) or (ai > aj and bi > bj):
                correct += 1.0

        return correct, total, label_ties, pred_ties
    
    correct_all = 0.0
    total_all = 0
    label_ties_all = 0
    pred_ties_all = 0
    all_pairs = 0

    macro_terms = []
    macro_label_ties = []
    macro_pred_ties = []

    pairs = []
    for r in test_results_raw:
        pairs.append((r["reward"], r[prediction_key]))
        
    for labels, predictions in pairs:
        n = len(labels)
        num_all_pairs = n * (n - 1) // 2
        all_pairs += num_all_pairs

        c, t, lt, pt = pairwise_counts(
            labels,
            predictions,
            eps_label=eps_label,
            eps_pred=eps_pred,
        )

        correct_all += c
        total_all += t
        label_ties_all += lt
        pred_ties_all += pt

        if t > 0:
            macro_terms.append(c / t)
            macro_pred_ties.append(pt / t)

        if num_all_pairs > 0:
            macro_label_ties.append(lt / num_all_pairs)

    micro_acc = (correct_all / total_all) if total_all > 0 else float("nan")
    macro_acc = (sum(macro_terms) / len(macro_terms)) if macro_terms else float("nan")

    label_tie_rate = (label_ties_all / all_pairs) if all_pairs > 0 else float("nan")
    pred_tie_rate = (pred_ties_all / total_all) if total_all > 0 else float("nan")

    return micro_acc, macro_acc, label_tie_rate, pred_tie_rate


PairType = Literal["KK", "KU", "UU"]

def evaluate_pair_micro_and_type_macro(
    test_results_raw: List[Dict[str, Any]],
    target_model_pairs: List[Tuple[str, str, PairType]],
    eps_label: float = 1e-8,
    eps_pred: float = 1e-4,
    ) -> Dict[str, Any]:
    """
    Compute:
      - pair_micro: micro-accuracy per model-pair
      - type_macro: macro-average of pair_micro accuracies per pair type (KK/KU/UU),
                    plus type-level aggregated counts (total/label_ties/pred_ties/coverage)
      - pair_macro: macro-average of pair_micro accuracies over ALL pairs (type ignored),
                    plus overall aggregated counts

    Rules:
      - label tie (|y_i - y_j| <= eps_label OR NaN label) -> invalid comparison (skipped)
      - pred tie (|s_i - s_j| <= eps_pred) -> counted as 0.5 correct
      - pred NaN -> invalid comparison (skipped)
      - Outputs use None instead of NaN when undefined.
    """

    def normalize(a: str, b: str) -> Tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def safe_div(num: float, den: int):
        return None if den == 0 else (num / den)

    valid_types = {"KK", "KU", "UU"}

    # Map normalized pair -> type
    pair_to_type: Dict[Tuple[str, str], str] = {}
    for a, b, t in target_model_pairs:
        if t not in valid_types:
            raise ValueError(f"Unknown pair type: {t}")
        aa, bb = normalize(str(a), str(b))
        if (aa, bb) in pair_to_type and pair_to_type[(aa, bb)] != t:
            raise ValueError(f"Pair {(aa, bb)} assigned multiple types.")
        pair_to_type[(aa, bb)] = t

    # Per-pair accumulators
    pair_correct: Dict[Tuple[str, str], float] = {p: 0.0 for p in pair_to_type}
    pair_total: Dict[Tuple[str, str], int] = {p: 0 for p in pair_to_type}
    pair_label_ties: Dict[Tuple[str, str], int] = {p: 0 for p in pair_to_type}
    pair_pred_ties: Dict[Tuple[str, str], int] = {p: 0 for p in pair_to_type}
    pair_coverage: Dict[Tuple[str, str], int] = {p: 0 for p in pair_to_type}

    # ---- main aggregation loop ----
    for r in test_results_raw:
        model_ids = r["vton_model_id"]
        rewards = r["reward"]
        preds = r["prediction"]

        index = {str(m): i for i, m in enumerate(model_ids)}

        for (a, b), _ptype in pair_to_type.items():
            if a not in index or b not in index:
                continue

            pair_coverage[(a, b)] += 1

            i, j = index[a], index[b]
            ai, aj = float(rewards[i]), float(rewards[j])

            # label tie / invalid label -> invalid comparison
            if math.isnan(ai) or math.isnan(aj) or abs(ai - aj) <= eps_label:
                pair_label_ties[(a, b)] += 1
                continue

            bi, bj = float(preds[i]), float(preds[j])

            # invalid prediction -> invalid comparison (skipped)
            if math.isnan(bi) or math.isnan(bj):
                continue

            pair_total[(a, b)] += 1

            # pred tie -> 0.5 credit
            if abs(bi - bj) <= eps_pred:
                pair_pred_ties[(a, b)] += 1
                pair_correct[(a, b)] += 0.5
            else:
                ok = (ai < aj and bi < bj) or (ai > aj and bi > bj)
                if ok:
                    pair_correct[(a, b)] += 1.0

    # ---- build pair_micro ----
    pair_micro: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for p, ptype in pair_to_type.items():
        total = pair_total[p]
        micro = safe_div(pair_correct[p], total)

        pair_micro[p] = {
            "type": ptype,
            "micro_acc": micro,  # None if total==0
            "total": total,      # valid comparisons (label tie & pred NaN removed)
            "label_ties": pair_label_ties[p],  # invalid comparisons due to label ties
            "pred_ties": pair_pred_ties[p],    # ties among valid comparisons
            "coverage": pair_coverage[p],      # times both models were present in a sample
            "pred_tie_rate": safe_div(pair_pred_ties[p], total),
        }

    # ---- type macro (平均はペア単位) + type-level aggregated counts ----
    type_to_micros: Dict[str, List[float]] = {"KK": [], "KU": [], "UU": []}
    type_agg: Dict[str, Dict[str, int]] = {
        "KK": {"total": 0, "label_ties": 0, "pred_ties": 0, "coverage": 0, "num_pairs": 0},
        "KU": {"total": 0, "label_ties": 0, "pred_ties": 0, "coverage": 0, "num_pairs": 0},
        "UU": {"total": 0, "label_ties": 0, "pred_ties": 0, "coverage": 0, "num_pairs": 0},
    }

    for (_a, _b), stats in pair_micro.items():
        t = stats["type"]
        type_agg[t]["total"] += int(stats["total"])
        type_agg[t]["label_ties"] += int(stats["label_ties"])
        type_agg[t]["pred_ties"] += int(stats["pred_ties"])
        type_agg[t]["coverage"] += int(stats["coverage"])
        type_agg[t]["num_pairs"] += 1

        if stats["micro_acc"] is not None:
            type_to_micros[t].append(float(stats["micro_acc"]))

    type_macro: Dict[str, Dict[str, Any]] = {}
    for t in ["KK", "KU", "UU"]:
        micros = type_to_micros[t]
        macro = (sum(micros) / len(micros)) if micros else None

        tot = type_agg[t]["total"]
        type_macro[t] = {
            # Macro-average across pairs (each pair equally weighted)
            "macro_acc": macro,
            "num_pairs_with_defined_acc": len(micros),
            "num_pairs_total": type_agg[t]["num_pairs"],

            # Aggregated counts
            "total": tot,
            "label_ties": type_agg[t]["label_ties"],
            "pred_ties": type_agg[t]["pred_ties"],
            "coverage": type_agg[t]["coverage"],

            # Helpful rates (None if undefined)
            "pred_tie_rate": safe_div(type_agg[t]["pred_ties"], tot),
            "label_tie_rate_over_coverage": safe_div(type_agg[t]["label_ties"], type_agg[t]["coverage"]),
        }

    # ---- pair macro (type ignored): macro-average across ALL pairs ----
    all_micros: List[float] = []
    overall_agg = {"total": 0, "label_ties": 0, "pred_ties": 0, "coverage": 0, "num_pairs": 0}

    for (_a, _b), stats in pair_micro.items():
        overall_agg["total"] += int(stats["total"])
        overall_agg["label_ties"] += int(stats["label_ties"])
        overall_agg["pred_ties"] += int(stats["pred_ties"])
        overall_agg["coverage"] += int(stats["coverage"])
        overall_agg["num_pairs"] += 1

        if stats["micro_acc"] is not None:
            all_micros.append(float(stats["micro_acc"]))

    pair_macro = {
        "macro_acc": (sum(all_micros) / len(all_micros)) if all_micros else None,
        "num_pairs_with_defined_acc": len(all_micros),
        "num_pairs_total": overall_agg["num_pairs"],
        "total": overall_agg["total"],
        "label_ties": overall_agg["label_ties"],
        "pred_ties": overall_agg["pred_ties"],
        "coverage": overall_agg["coverage"],
        "pred_tie_rate": safe_div(overall_agg["pred_ties"], overall_agg["total"]),
        "label_tie_rate_over_coverage": safe_div(overall_agg["label_ties"], overall_agg["coverage"]),
    }

    return {
        "pair_micro": pair_micro,
        "type_macro": type_macro,
        "pair_macro": pair_macro,
    }


    
def evaluate_human_performance(
    bootstrapped_sample: List[Dict[str, Any]], 
    model_temperature: float = 0.65,
    human_temperature: float = 0.65,
    ) -> Tuple[Dict[str, Any], Figure, Figure, Figure, Figure]:
        
    micro_acc, macro_acc, label_tie_rate, pred_tie_rate = evaluate_micro_macro_pairwise_accuracy(bootstrapped_sample, eps_label=1e-8, eps_pred=1e-4)

    r1 = list(itertools.chain.from_iterable([data["reward"] for data in bootstrapped_sample]))
    r2 = list(itertools.chain.from_iterable([data["prediction"] for data in bootstrapped_sample]))
    spearman, p_spearman = spearmanr(r1, r2)
    pearson, p_pearson = pearsonr(r1, r2)
    
    r2_score_value = r2_score(r1, r2)
        
    interval = 0.05
    palign_means, palign_stdevs, lbs = evaluate_probability_alignment(bootstrapped_sample, model_temperature, human_temperature, interval)
    fig_p_align, _ = plot_probability_alignment_curve(palign_means, palign_stdevs, lbs, label="model")
    
    acc_alignments, lbs = evaluate_pairwise_acc_alignment(bootstrapped_sample, model_temperature, human_temperature, interval)
    fig_acc_align, _ = plot_pairwise_acc_alignment_curve(acc_alignments, lbs, label="model")
    
    fig_score_dist, _ = plot_score_distribution(bootstrapped_sample)
    
    n_bins = 10
    ece = evaluate_expected_calibration_error(bootstrapped_sample, model_temperature, n_bins)
    brier = evaluate_brier_score(bootstrapped_sample, model_temperature)
    fig_calibration, _ = plot_calibration_curve(bootstrapped_sample, model_temperature, ece, brier, n_bins)
        
    metrics = {
        "spearman": {"value": float(spearman), "p_value": float(p_spearman)}, 
        "pearson": {"value": float(pearson), "p_value": float(p_pearson)}, 
        "r2": {"value": float(r2_score_value)},
        "micro_pairwise_acc": {"value": float(micro_acc), "label_tie_rate": float(label_tie_rate), "pred_tie_rate": float(pred_tie_rate)},
        "macro_pairwise_acc": {"value": float(macro_acc), "label_tie_rate": float(label_tie_rate), "pred_tie_rate": float(pred_tie_rate)},
        "ece": {"value": float(ece)},
        "brier": {"value": float(brier)},
        "probability_align": {"value": palign_means, "stdev": palign_stdevs, "lbs": [float(l) for l in lbs]},
        "acc_alignments": {"value": acc_alignments, "lbs": [float(l) for l in lbs]},
        }
    
    summary = {"metrics": metrics, "raw": bootstrapped_sample}
        
    return summary, fig_p_align, fig_acc_align, fig_calibration, fig_score_dist
