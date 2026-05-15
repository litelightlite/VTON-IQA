import os
import uuid
import tqdm
import json
import copy
import joblib
import logging
import statistics
from typing import Dict, Any, List, Literal, Tuple
from omegaconf import OmegaConf, DictConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from scorer.configuration_vtoniqa import VTONScorerConfig
from scorer.modeling_vtoniqa import VTONScorer
from util.metrics import evaluate_micro_macro_pairwise_accuracy, bce_with_logits_by_group, evaluate_probability_alignment, evaluate_pairwise_acc_alignment, evaluate_expected_calibration_error, evaluate_brier_score, bootstrap_annotation, evaluate_human_performance
from util.visualize import setup_mpl, plot_calibration_curve, plot_probability_alignment_curve, plot_pairwise_acc_alignment_curve, plot_score_distribution
from util.commons import set_random_seed, create_logger, create_config, get_exec_cmd_as_str, get_now, is_ampera_gpu_available, VTONQBENCH_ROOT
from dataset import GroupedRewardDataset

HUMAN_TEMPERATURE = 0.65

torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def evaluate(
    model: VTONScorer, 
    loader: DataLoader, 
    loss_type: Literal["regression", "ranking", "ranking+regression"]
    ) -> Tuple[Dict[str, Any], Figure, Figure, Figure]:
    
    model.eval()
    
    preds, targets, group_ids = np.array([]), np.array([]), np.array([])
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if is_ampera_gpu_available() else torch.float32, enabled=True), torch.no_grad():
        for batch in tqdm.tqdm(loader, total=len(loader), desc="evaluation"):
            out = model(batch["person"].squeeze(0), batch["garment"].squeeze(0), batch["vton"].squeeze(0))
            preds = np.concatenate([preds, out.to('cpu').detach().numpy().copy()])
            targets = np.concatenate([targets, batch["reward"].squeeze(0).to('cpu').detach().numpy().copy()])
            group_ids = np.concatenate([group_ids, batch["group_id"].long().squeeze(0).to('cpu').detach().numpy().copy()])
    
    spearman, p_spearman = spearmanr(preds, targets)
    pearson, p_pearson = pearsonr(preds, targets)
    r2 = r2_score(targets, preds)
    rmse = float(np.mean((preds - targets) ** 2) ** 0.5)
    
    if loss_type in ["regression"]:
        loss = rmse
    elif loss_type in ["ranking"]:
        loss, _ = bce_with_logits_by_group(
            torch.tensor(preds), torch.tensor(targets), torch.tensor(group_ids), model_temperature=model.get_temperature().item(), human_temperature=HUMAN_TEMPERATURE, reduction="mean"
        )
        loss = float(loss.item())
    elif loss_type in ["ranking+regression"]:
        ranking_loss, _ = bce_with_logits_by_group(
            torch.tensor(preds), torch.tensor(targets), torch.tensor(group_ids), model_temperature=model.get_temperature().item(), human_temperature=HUMAN_TEMPERATURE, reduction="mean"
        )
        loss = 0.5 * float(ranking_loss.item()) + 0.5 * rmse
    
    raw = copy.deepcopy(loader.dataset.dataset.dataset)
    for pred, group_id in zip(preds, group_ids):
        if "prediction" not in raw[int(group_id)].keys():
            raw[int(group_id)]["prediction"] = [float(pred)]
        else:
            raw[int(group_id)]["prediction"].append(float(pred))
    
    micro_acc, macro_acc, label_tie_rate, pred_tie_rate = evaluate_micro_macro_pairwise_accuracy(raw, eps_label=1e-8, eps_pred=1e-4)
    
    interval = 0.05
    palign_means, palign_stdevs, lbs, n_instances = evaluate_probability_alignment(raw, model.get_temperature().item(), HUMAN_TEMPERATURE, interval)
    fig_p_align, _ = plot_probability_alignment_curve(palign_means, palign_stdevs, lbs, label="model", n_instances=n_instances)
    
    acc_alignments, lbs = evaluate_pairwise_acc_alignment(raw, model.get_temperature().item(), HUMAN_TEMPERATURE, interval)
    fig_acc_align, _ = plot_pairwise_acc_alignment_curve(acc_alignments, lbs, label="model")
    
    fig_score_dist, _ = plot_score_distribution(raw)
    
    n_bins = 10
    ece = evaluate_expected_calibration_error(raw, model.get_temperature().item(), n_bins)
    brier = evaluate_brier_score(raw, model.get_temperature().item())
    fig_calibration, _ = plot_calibration_curve(raw, model.get_temperature().item(), ece, brier, n_bins)
    
    metrics = {
        "spearman": {"value": float(spearman), "p_value": float(p_spearman)}, 
        "pearson": {"value": float(pearson), "p_value": float(p_pearson)}, 
        "r2": {"value": float(r2)},
        "micro_pairwise_acc": {"value": float(micro_acc), "label_tie_rate": float(label_tie_rate), "pred_tie_rate": float(pred_tie_rate)},
        "macro_pairwise_acc": {"value": float(macro_acc), "label_tie_rate": float(label_tie_rate), "pred_tie_rate": float(pred_tie_rate)},
        "rmse": {"value": float(rmse)}, 
        "loss": {"value": float(loss)},
        "ece": {"value": float(ece)},
        "brier": {"value": float(brier)},
        "probability_align": {"value": palign_means, "stdev": palign_stdevs, "lbs": [float(l) for l in lbs]},
        "acc_alignments": {"value": acc_alignments, "lbs": [float(l) for l in lbs]},
        }
    
    summary = {"metrics": metrics, "raw": raw}
    
    model.train()
    
    return summary, fig_p_align, fig_acc_align, fig_calibration, fig_score_dist

def display_summaries(
    phase: str,
    information: str,
    summaries: Dict[str, Any], 
    logger: logging.Logger
    ) -> None:
    logger.info(
        f"[{phase}]"
        f"[{information}] "
        f"{phase.lower()} "
        f"spearman={summaries['metrics']['spearman']['value']:.4f}, "
        f"pearson={summaries['metrics']['pearson']['value']:.4f}, "
        f"r2={summaries['metrics']['r2']['value']:.4f}, "
        f"micro_pairwise_acc={summaries['metrics']['micro_pairwise_acc']['value']:.4f}, "
        f"macro_pairwise_acc={summaries['metrics']['macro_pairwise_acc']['value']:.4f}, "
        f"ece={summaries['metrics']['ece']['value']:.4f}, "
        f"brier={summaries['metrics']['brier']['value']:.4f}"
        )

def format_metrics(metrics: dict, sig: int = 4) -> str:
    lines = []
    for name, stats in metrics.items():
        mean = stats["mean"]
        stdev = stats["stdev"]
        line = f"{name:20s}: {mean:.{sig}g} ± {stdev:.{sig}g}"
        lines.append(line)
    return "\n".join(lines)

def evaluate_bs_summary(bs_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    
    spearman_seq, pearson_seq, r2_seq, macro_pairwise_acc_seq, micro_pairwise_acc_seq, ece_seq, brier_seq = [], [], [], [], [], [], []
    for bs in bs_summaries:
        spearman_seq.append(float(bs["metrics"]["spearman"]["value"]))
        pearson_seq.append(float(bs["metrics"]["pearson"]["value"]))
        r2_seq.append(float(bs["metrics"]["r2"]["value"]))
        macro_pairwise_acc_seq.append(float(bs["metrics"]["macro_pairwise_acc"]["value"]))
        micro_pairwise_acc_seq.append(float(bs["metrics"]["micro_pairwise_acc"]["value"]))
        ece_seq.append(float(bs["metrics"]["ece"]["value"]))
        brier_seq.append(float(bs["metrics"]["brier"]["value"]))
    
    bs_metrics = {
        "spearman": {"mean": statistics.mean(spearman_seq), "stdev": statistics.stdev(spearman_seq)}, 
        "pearson": {"mean": statistics.mean(pearson_seq), "stdev": statistics.stdev(pearson_seq)}, 
        "r2": {"mean": statistics.mean(r2_seq), "stdev": statistics.stdev(r2_seq)},
        "macro_pairwise_acc": {"mean": statistics.mean(macro_pairwise_acc_seq), "stdev": statistics.stdev(macro_pairwise_acc_seq)},
        "micro_pairwise_acc": {"mean": statistics.mean(micro_pairwise_acc_seq), "stdev": statistics.stdev(micro_pairwise_acc_seq)},
        "ece": {"mean": statistics.mean(ece_seq), "stdev": statistics.stdev(ece_seq)},
        "brier": {"mean": statistics.mean(brier_seq), "stdev": statistics.stdev(brier_seq)},
     }  
    
    return bs_metrics


def filter_records(
    records: List[Dict[str, Any]],
    target_vton_model_ids: List[str],
    target_dataset_types: List[str],
) -> List[Dict[str, Any]]:
    target_vton_model_ids = set(target_vton_model_ids)
    target_dataset_types = set(target_dataset_types)

    filtered = []
    for r in records:
        vton_ids = set(r.get("vton_model_id", []))
        dataset_types = set(r.get("dataset_type", []))

        if vton_ids & target_vton_model_ids and dataset_types & target_dataset_types:
            filtered.append(r)

    return filtered

def run_half(cfg: DictConfig, save_root: str, logger: logging.Logger):

    setup_mpl()
    set_random_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(cfg.scorer_model_root, "target_scaler.pkl"), 'rb') as f:
        scaler = joblib.load(f)
    
    # ===== Model =====
    vtoniqa = VTONScorer(VTONScorerConfig(**json.load(open(os.path.join(cfg.scorer_model_root, "config.json")))))
    vtoniqa.load_state_dict(load_file(os.path.join(cfg.scorer_model_root, "model.safetensors"))) 
    vtoniqa.to(device) 
    vtoniqa.eval()
    
    if cfg.bs_annos_path is None:
        bs_annos = bootstrap_annotation(
            reward_dataset_path=os.path.join(VTONQBENCH_ROOT, "test.json"), 
            n_bootstrap=cfg.n_bootstrap
            )
    else:
        bs_annos = json.load(open(cfg.bs_annos_path))
    
    bs_summary, bs_summary_human = [], []
    for bootstrap_id, bs_anno in tqdm.tqdm(enumerate(bs_annos), total=cfg.n_bootstrap, desc="Evaluating bootstrap samples"):
        df = pd.DataFrame(bs_anno)
        df = df.explode([
            'form_id', 'problem_id', 'dataset_type', 'vton_model_id',
            'garment_type', 'vton', 'reward', 'votes',
            'annotator_ids', 'answer_ids', 'ssim', 'lpips',
        ])
        df["person"] = df.apply(lambda row: "/".join(row["person"].split("/")[5:]), axis=1)
        df["garment"] = df.apply(lambda row: "/".join(row["garment"].split("/")[5:]), axis=1)
        df["vton"] = df.apply(lambda row: "/".join(row["vton"].split("/")[5:]), axis=1)
        df = df.rename(columns={"person": "person_path", "garment": "garment_path", "vton": "vton_path"})
        
        local_save_root = os.path.join(save_root, str(bootstrap_id))
        os.makedirs(local_save_root, exist_ok=True)
        
        with open(os.path.join(local_save_root, "half_test.json"), "w") as f:
            json.dump(df.to_dict(orient="records"), f)
        
        image_size = (512, 512)
        test_data_half = GroupedRewardDataset(
            expand_2_square=True, align_image_size=True, image_size=image_size,
            reward_dataset_path=os.path.join(local_save_root, "half_test.json"), 
            target_vton_model_ids=cfg.test_target_vton_model_ids, target_dataset_types=cfg.test_target_dataset_types,
            standardization_type=None, with_metadata=False
        )
        test_data_half.set_scaler(scaler)
        test_data_half_loader = DataLoader(
            test_data_half, batch_size=1, shuffle=False, num_workers=os.cpu_count()-6, pin_memory=True, pin_memory_device="cuda:0", persistent_workers=True, prefetch_factor=4
        )
        test_summary, fig_p_align, fig_acc_align, fig_calibration, fig_score_dist = evaluate(vtoniqa, test_data_half_loader, cfg.loss_type)
        bs_summary.append(test_summary)
        display_summaries(phase="TEST HALF（MODEL）", information=f"{bootstrap_id}/{cfg.n_bootstrap}", summaries=test_summary, logger=logger)
        
        with open(os.path.join(local_save_root, "test_summary.json"), "w") as f:
            json.dump([test_summary], f, indent=4) 
        
        fig_p_align.savefig(os.path.join(local_save_root, "test_probability_alignment_curve.png"))
        fig_acc_align.savefig(os.path.join(local_save_root, "test_acc_alignment_curve.png"))
        fig_calibration.savefig(os.path.join(local_save_root, "test_calibration_curve.png"))
        fig_score_dist.savefig(os.path.join(local_save_root, "valid_score_distribution.png"))
        plt.clf()
        plt.close()
        
        bs_anno_filtered = filter_records(bs_anno, cfg.test_target_vton_model_ids, cfg.test_target_dataset_types)
        test_summary_human, fig_p_align_human, fig_acc_align_human, fig_calibration_human, fig_score_dist_human  = evaluate_human_performance(bs_anno_filtered, model_temperature=0.65, human_temperature=0.65)
        display_summaries(phase="TEST HALF（HUMAN）", information=f"{bootstrap_id}/{cfg.n_bootstrap}", summaries=test_summary_human, logger=logger)
        bs_summary_human.append(test_summary_human)
        with open(os.path.join(local_save_root, "test_summary_human.json"), "w") as f:
            json.dump([test_summary_human], f, indent=4) 
        
        fig_p_align_human.savefig(os.path.join(local_save_root, "test_probability_alignment_curve_human.png"))
        fig_acc_align_human.savefig(os.path.join(local_save_root, "test_acc_alignment_curve_human.png"))
        fig_calibration_human.savefig(os.path.join(local_save_root, "test_calibration_curve_human.png"))
        fig_score_dist_human.savefig(os.path.join(local_save_root, "valid_score_distribution_human.png"))
        plt.clf()
        plt.close()
    
    bs_summary_mean = evaluate_bs_summary(bs_summary)
    logger.info("================ MODEL PERFORMANCE ================")
    logger.info(format_metrics(bs_summary_mean))
    with open(os.path.join(save_root, "bs_summary.json"), "w") as f:
        json.dump([bs_summary_mean], f, indent=4)
    
    logger.info("================ HUMAN PERFORMANCE ================")
    bs_summary_human_mean = evaluate_bs_summary(bs_summary_human)
    logger.info(format_metrics(bs_summary_human_mean))
    with open(os.path.join(save_root, "bs_summary_human.json"), "w") as f:
        json.dump([bs_summary_human_mean], f, indent=4)


def run_full(cfg: DictConfig, save_root: str, logger: logging.Logger):
    
    setup_mpl()
    set_random_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(cfg.scorer_model_root, "target_scaler.pkl"), 'rb') as f:
        scaler = joblib.load(f)
    
    # ===== Model =====
    vtoniqa = VTONScorer(VTONScorerConfig(**json.load(open(os.path.join(cfg.scorer_model_root, "config.json")))))
    vtoniqa.load_state_dict(load_file(os.path.join(cfg.scorer_model_root, "model.safetensors"))) 
    vtoniqa.to(device) 
    vtoniqa.eval()
    
    image_size = (512, 512)
    test_data = GroupedRewardDataset(
        expand_2_square=True, align_image_size=True, image_size=image_size,
        reward_dataset_path=os.path.join(VTONQBENCH_ROOT, "test.json"), 
        target_vton_model_ids=cfg.test_target_vton_model_ids, target_dataset_types=cfg.test_target_dataset_types,
        standardization_type=None, with_metadata=False
    )
    test_data.set_scaler(scaler)
    
    test_loader = DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=os.cpu_count()-6, pin_memory=True, pin_memory_device="cuda:0", persistent_workers=True, prefetch_factor=4
    )

    test_summary, fig_p_align, fig_acc_align, fig_calibration, fig_score_dist = evaluate(vtoniqa, test_loader, cfg.loss_type)
    
    with open(os.path.join(save_root, "test_summary.json"), "w") as f:
        json.dump([test_summary], f, indent=4) 
    
    fig_p_align.savefig(os.path.join(save_root, "test_probability_alignment_curve.png"))
    fig_acc_align.savefig(os.path.join(save_root, "test_acc_alignment_curve.png"))
    fig_calibration.savefig(os.path.join(save_root, "test_calibration_curve.png"))
    fig_score_dist.savefig(os.path.join(save_root, "test_score_distribution.png"))
    plt.clf()
    plt.close()

    display_summaries("TEST", information="full", summaries=test_summary, logger=logger)

def main():

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
        
    cfg = create_config()
    cfg["cmd"] = get_exec_cmd_as_str()
    cfg["start_time"] = get_now()
    cfg["train_job_id"] = str(uuid.uuid4())
    
    set_random_seed(cfg.seed)
    save_root = os.path.join("outputs", cfg.test_type, cfg["scorer_model_root"].replace("/", "+"), cfg['start_time'])
    save_root = save_root+f"_{cfg.save_root_postfix}" if cfg.save_root_postfix is not None else save_root
    os.makedirs(save_root, exist_ok=True)    

    OmegaConf.save(cfg, os.path.join(save_root, "config.yaml"))
    
    logger = create_logger(name=__name__, file_name=os.path.join(save_root, "stdout.log"))
    logger.info(f"{OmegaConf.to_yaml(cfg)}")
    logger.info(f"save_root={save_root}")
    logger.info("Testing VTON scorer By Bootstrapped Data...")
    
    if cfg.test_type=="half":
        run_half(cfg, save_root, logger)
    elif cfg.test_type=="full":
        run_full(cfg, save_root, logger)
    
if __name__=="__main__":
    main()
