import os
import gc
import tqdm
import uuid
import json
import joblib
import random
import logging
import itertools
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from safetensors.torch import load_file
from omegaconf import OmegaConf, DictConfig
from scorer.configuration_vtoniqa import VTONScorerConfig
from scorer.modeling_vtoniqa import VTONScorer
from util.metrics import evaluate_pair_micro_and_type_macro
from util.visualize import setup_mpl, plot_pair_micro_heatmap_blocks
from util.commons import set_random_seed, create_logger, create_config, get_exec_cmd_as_str, is_ampera_gpu_available, get_now, VTONQBENCH_ROOT
from dataset import RewardRegressionDataset, RewardComparisonDataset, GroupedRewardDataset, VTON_MODEL_IDS
from test import evaluate

HUMAN_TEMPERATURE = 0.65

torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
    
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.seed
    np.random.seed(seed % (2**32))
    random.seed(seed)
    torch.manual_seed(seed)
    
def display_summaries(
    phase: str,
    epoch: int,
    max_epochs: int,
    summaries: Dict[str, Any], 
    logger: logging.Logger
    ) -> None:
    logger.info(
        f"[{phase}]"
        f"[Epoch {epoch}/{max_epochs}] "
        f"{phase.lower()} "
        f"loss={summaries['metrics']['loss']['value']:.4f}, "
        f"rmse={summaries['metrics']['rmse']['value']:.4f}, "
        f"spearman={summaries['metrics']['spearman']['value']:.4f}, "
        f"pearson={summaries['metrics']['pearson']['value']:.4f}, "
        f"r2={summaries['metrics']['r2']['value']:.4f}, "
        f"micro_pairwise_acc={summaries['metrics']['micro_pairwise_acc']['value']:.4f}, "
        f"macro_pairwise_acc={summaries['metrics']['macro_pairwise_acc']['value']:.4f}, "
        f"ece={summaries['metrics']['ece']['value']:.4f}, "
        f"brier={summaries['metrics']['brier']['value']:.4f}"
        )

def setup_requires_grad(model: VTONScorer):
    
    for param in model.backbone.model.parameters():
        param.requires_grad = False
    
    model_type = model.backbone.model.config.model_type
    
    if model_type in ["dinov3_vit"]:
        n_layer = len(model.backbone.model.layer)
        for i in range(n_layer-model.config.n_mid_s_attn, n_layer):
            for param in model.backbone.model.layer[i].parameters():
                param.requires_grad = True
        model.backbone.model.norm.requires_grad = True

    if model.config.n_mid_c_attn > 0:
        for param in model.backbone.mid_p2v_blocks.parameters():
            param.requires_grad = True
        for param in model.backbone.mid_v2p_blocks.parameters():
            param.requires_grad = True
        for param in model.backbone.mid_c2v_blocks.parameters():
            param.requires_grad = True
        for param in model.backbone.mid_v2c_blocks.parameters():
            param.requires_grad = True
    
    return model


def train(cfg: DictConfig, save_root: str, logger: logging.Logger):
    
    setup_mpl()
    set_random_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Model =====
    model = VTONScorer(VTONScorerConfig(**cfg.model))
    model = setup_requires_grad(model)
    
    logger.info("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)
    
    params = []
    params.append({"params": [p for name, p in model.backbone.named_parameters() if p.requires_grad and name.startswith("model")], "lr": 3e-5})
    params.append({"params": [p for name, p in model.backbone.named_parameters() if p.requires_grad and not name.startswith("model")], "lr": 1e-4})
    params.append({"params": [model.scorer.alpha], "lr": 1e-4})
    params.append({"params": [model.scorer.a, model.scorer.b], "lr": 1e-4})

    if cfg.model.temperature_scale is None:
        params.append({"params": [model.log_t], "lr": 1e-4})
    
    optimizer = torch.optim.AdamW(params)
    model.to(device)
    
    image_size = (512, 512)
    # ===== Dataset =====
    if cfg.loss_type in ["regression"]:
        train_data = RewardRegressionDataset(
            expand_2_square=True, align_image_size=True, image_size=image_size,
            reward_dataset_path=os.path.join(VTONQBENCH_ROOT, "train.json"), 
            target_vton_model_ids=cfg.train_target_vton_model_ids, target_dataset_types=cfg.train_target_dataset_types,
            standardization_type=cfg.standardization_type, with_metadata=False, real_augmentation=cfg.real_augmentation
        )
        indices = sorted(random.sample(range(len(train_data)), int(len(train_data)*cfg.dataset_use_ratio)))
        train_data = Subset(train_data, indices)
    elif cfg.loss_type in ["ranking", "ranking+regression"]:
        train_data = RewardComparisonDataset(
            expand_2_square=True, align_image_size=True, image_size=image_size,
            reward_dataset_path=os.path.join(VTONQBENCH_ROOT, "train.json"), 
            target_vton_model_ids=cfg.train_target_vton_model_ids, target_dataset_types=cfg.train_target_dataset_types,
            standardization_type=cfg.standardization_type, with_metadata=False, temperature=HUMAN_TEMPERATURE,
            real_augmentation=cfg.real_augmentation
        )
        indices = sorted(random.sample(range(len(train_data)), int(len(train_data)*cfg.dataset_use_ratio)))
        train_data = Subset(train_data, indices)
    else:
        raise ValueError(f"Unknown task={cfg.task}")

    valid_data = GroupedRewardDataset(
        expand_2_square=True, align_image_size=True, image_size=image_size,
        reward_dataset_path=os.path.join(VTONQBENCH_ROOT, "valid.json"), 
        target_vton_model_ids=cfg.valid_target_vton_model_ids, target_dataset_types=cfg.valid_target_dataset_types,
        standardization_type=None, with_metadata=False
    )
    valid_data.set_scaler(train_data.dataset.get_scaler())
    test_data = GroupedRewardDataset(
        expand_2_square=True, align_image_size=True, image_size=image_size,
        reward_dataset_path=os.path.join(VTONQBENCH_ROOT, "test.json"), 
        target_vton_model_ids=cfg.test_target_vton_model_ids, target_dataset_types=cfg.test_target_dataset_types,
        standardization_type=None, with_metadata=False
    )
    test_data.set_scaler(train_data.dataset.get_scaler())
    
    known_vton_model_ids = set(cfg.train_target_vton_model_ids).union(set(cfg.valid_target_vton_model_ids))
    unknown_vton_model_ids = set(VTON_MODEL_IDS) - known_vton_model_ids
    if 0 < len(unknown_vton_model_ids):
        test_data_unknown_models = GroupedRewardDataset(
            expand_2_square=True, align_image_size=True, image_size=image_size,
            reward_dataset_path=os.path.join(VTONQBENCH_ROOT, "test.json"), 
            target_vton_model_ids=sorted(list(unknown_vton_model_ids)), target_dataset_types=cfg.test_target_dataset_types,
            standardization_type=None, with_metadata=False
        )
        test_data_unknown_models.set_scaler(train_data.dataset.get_scaler())
        
        test_data_known_models = GroupedRewardDataset(
            expand_2_square=True, align_image_size=True, image_size=image_size,
            reward_dataset_path=os.path.join(VTONQBENCH_ROOT, "test.json"), 
            target_vton_model_ids=sorted(list(known_vton_model_ids)), target_dataset_types=cfg.test_target_dataset_types,
            standardization_type=None, with_metadata=False
        )
        test_data_known_models.set_scaler(train_data.dataset.get_scaler())
    else:
        test_data_unknown_models = None
        test_data_known_models = None
    
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    
    train_loader = DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=os.cpu_count()-4, pin_memory=True, worker_init_fn=worker_init_fn, generator=g, pin_memory_device="cuda:0", persistent_workers=True, prefetch_factor=8
    )
    valid_loader = DataLoader(
        valid_data, batch_size=1, shuffle=False, num_workers=os.cpu_count()-6, pin_memory=True, pin_memory_device="cuda:0", persistent_workers=True, prefetch_factor=4
    )
    test_loader = DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=os.cpu_count()-6, pin_memory=True, pin_memory_device="cuda:0", persistent_workers=True, prefetch_factor=4
    )
    
    joblib.dump(train_data.dataset.get_scaler(), os.path.join(save_root, "target_scaler.pkl"))

    # ===== Train loop =====
    global_step = 0
    best_metric, best_epoch = float("inf"), 0
    best_dir = os.path.join(save_root, "best")
    model.save_pretrained(best_dir)
    loss_ratio, tolerance, summaries = 0.5, 0, []
    for epoch in range(1, cfg.max_epochs + 1):
        if epoch == 1:
            valid_summary, fig_p_align, fig_acc_align, fig_calibration, fig_score_dist = evaluate(model, valid_loader, cfg.loss_type)
            val_loss = float(valid_summary["metrics"]["loss"]["value"])
            best_metric = val_loss
            best_epoch = epoch-1
            display_summaries("VALID", epoch-1, cfg.max_epochs, valid_summary, logger)
            summaries.append({
                "epoch": epoch-1, 
                "train_loss": None, 
                "valid_summary": valid_summary
            })
            os.makedirs(os.path.join(save_root, f"epoch_{epoch-1:03d}"), exist_ok=True)
            with open(os.path.join(save_root, f"epoch_{epoch-1:03d}", "summary.json"), "w") as f:
                json.dump(summaries, f, indent=4)
            fig_p_align.savefig(os.path.join(save_root, f"epoch_{epoch-1:03d}", "valid_probability_alignment_curve.png"))
            fig_acc_align.savefig(os.path.join(save_root, f"epoch_{epoch-1:03d}", "valid_acc_alignment_curve.png"))
            fig_calibration.savefig(os.path.join(save_root, f"epoch_{epoch-1:03d}", "valid_calibration_curve.png"))
            fig_score_dist.savefig(os.path.join(save_root, f"epoch_{epoch-1:03d}", "valid_score_distribution.png"))
            plt.clf()
            plt.close()
        
        model.train()
        running_loss = 0.0
        
        optimizer.zero_grad(set_to_none=True)

        for it, batch in enumerate(tqdm.tqdm(train_loader, total=len(train_loader), desc=f"epoch: {epoch}"), start=1):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if is_ampera_gpu_available() else torch.float32, enabled=True):
                if cfg.loss_type in "regression":
                    out = model(batch["person"], batch["garment"], batch["vton"])
                elif cfg.loss_type in ["ranking", "ranking+regression"]:
                    out_x = model(batch["person"], batch["garment"], batch["vton_x"])
                    out_y = model(batch["person"], batch["garment"], batch["vton_y"])

            if cfg.loss_type == "regression":
                loss = F.mse_loss(out.to(dtype=torch.float32), batch["reward"].to(device, dtype=torch.float32))
            elif cfg.loss_type == "ranking":
                logit = (out_x-out_y) / model.get_temperature()
                loss = F.binary_cross_entropy_with_logits(logit.to(dtype=torch.float32), batch["label"].to(device, dtype=torch.float32))
            elif cfg.loss_type == "ranking+regression":
                logit = (out_x-out_y) / model.get_temperature()
                ranking_loss = F.binary_cross_entropy_with_logits(logit.to(dtype=torch.float32), batch["label"].to(device, dtype=torch.float32))
                regression_loss = F.mse_loss(
                    torch.concat([out_x, out_y[torch.logical_not(batch["is_identical_pair"])]]).to(dtype=torch.float32),
                    torch.concat([batch["reward_x"], batch["reward_y"][torch.logical_not(batch["is_identical_pair"])]]).to(device, dtype=torch.float32)
                    )                    
                loss = loss_ratio * ranking_loss + (1.0 - loss_ratio) * regression_loss
    
            (loss / max(1, cfg.grad_accum_steps)).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if it % cfg.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            
            running_loss += loss.item()
            
            if global_step % 500 == 0 or global_step == 1:
                logger.info(f"[Epoch {epoch} Iter {it}/{len(train_loader)}] step={global_step} train_loss={(running_loss / it):.4f}")
            
        valid_summary, fig_p_align, fig_acc_align, fig_calibration, fig_score_dist = evaluate(model, valid_loader, cfg.loss_type)
        val_loss = float(valid_summary["metrics"]["loss"]["value"])        
        display_summaries("VALID", epoch, cfg.max_epochs, valid_summary, logger)

        summaries.append({
            "epoch": epoch, 
            "train_loss": running_loss / len(train_loader), 
            "valid_summary": valid_summary
            })

        if (epoch % max(1, cfg.save_every_epochs)) == 0:
            os.makedirs(os.path.join(save_root, f"epoch_{epoch:03d}"), exist_ok=True)
            with open(os.path.join(save_root, f"epoch_{epoch:03d}", "summary.json"), "w") as f:
                json.dump(summaries, f, indent=4)
            model.save_pretrained(os.path.join(save_root, f"epoch_{epoch:03d}"))
            fig_p_align.savefig(os.path.join(save_root, f"epoch_{epoch:03d}", "valid_probability_alignment_curve.png"))
            fig_acc_align.savefig(os.path.join(save_root, f"epoch_{epoch:03d}", "valid_acc_alignment_curve.png"))
            fig_calibration.savefig(os.path.join(save_root, f"epoch_{epoch:03d}", "valid_calibration_curve.png"))
            fig_score_dist.savefig(os.path.join(save_root, f"epoch_{epoch:03d}", "valid_score_distribution.png"))
            plt.clf()
            plt.close()

        # save (best & periodic)
        if val_loss < best_metric:
            best_metric = val_loss
            best_epoch = epoch
            model.save_pretrained(best_dir)
            logger.info(f"  -> saved best to {best_dir}")
            tolerance = 0
        else:
            tolerance += 1
                
        if cfg.early_stopping_patience <= tolerance:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    with open(os.path.join(save_root, "train_summary.json"), "w") as f:
        json.dump(summaries, f, indent=4) 
        
    del train_loader
    del valid_loader
    gc.collect()
    torch.cuda.empty_cache()
    
    model.to("cpu")
    
    logger.info("======== Start Test Evaluation ========")
    
    best_model = VTONScorer(VTONScorerConfig(**json.load(open(os.path.join(best_dir, "config.json")))))
    best_model.load_state_dict(load_file(os.path.join(best_dir, "model.safetensors"))) 
    best_model.to(device) 
    best_model.eval()     
    
    test_summary, fig_p_align, fig_acc_align, fig_calibration, fig_score_dist = evaluate(best_model, test_loader, cfg.loss_type)
    test_summary["best_epoch"] = best_epoch
    with open(os.path.join(save_root, "test_summary.json"), "w") as f:
        json.dump([test_summary], f, indent=4) 
        
    fig_p_align.savefig(os.path.join(best_dir, "test_probability_alignment_curve.png"))
    fig_acc_align.savefig(os.path.join(best_dir, "test_acc_alignment_curve.png"))
    fig_calibration.savefig(os.path.join(best_dir, "test_calibration_curve.png"))
    fig_score_dist.savefig(os.path.join(best_dir, "test_score_distribution.png"))
    plt.clf()
    plt.close()

    logger.info(f"Test summary (best epoch: {best_epoch}):")
    display_summaries("TEST", best_epoch, cfg.max_epochs, test_summary, logger)

    if test_data_unknown_models is not None:
        test_loader_unknown = DataLoader(
            test_data_unknown_models, batch_size=1, shuffle=False, num_workers=os.cpu_count()-6, pin_memory=True, pin_memory_device="cuda:0", persistent_workers=True, prefetch_factor=4
        )
        test_summary_unknown, fig_p_align, fig_acc_align, fig_calibration, fig_score_dist = evaluate(best_model, test_loader_unknown, cfg.loss_type)
        test_summary_unknown["best_epoch"] = best_epoch
        
        with open(os.path.join(save_root, "test_summary_unknown_models.json"), "w") as f:
            json.dump([test_summary_unknown], f, indent=4) 
        
        fig_p_align.savefig(os.path.join(best_dir, "test_unknown_models_probability_alignment_curve.png"))
        fig_acc_align.savefig(os.path.join(best_dir, "test_unknown_models_acc_alignment_curve.png"))
        fig_calibration.savefig(os.path.join(best_dir, "test_unknown_models_calibration_curve.png"))
        fig_score_dist.savefig(os.path.join(best_dir, "test_unknown_models_score_distribution.png"))
        plt.clf()
        plt.close()
        
        logger.info(f"Test summary (best epoch: {best_epoch}) for unknown models:")
        display_summaries("TEST_UNKNOWN_MODELS", best_epoch, cfg.max_epochs, test_summary_unknown, logger)
        
        test_loader_known = DataLoader(
            test_data_known_models, batch_size=1, shuffle=False, num_workers=os.cpu_count()-6, pin_memory=True, pin_memory_device="cuda:0", persistent_workers=True, prefetch_factor=4
        )
        test_summary_known, fig_p_align, fig_acc_align, fig_calibration, fig_score_dist = evaluate(best_model, test_loader_known, cfg.loss_type)
        test_summary_known["best_epoch"] = best_epoch
        
        with open(os.path.join(save_root, "test_summary_known_models.json"), "w") as f:
            json.dump([test_summary_known], f, indent=4) 
        
        fig_p_align.savefig(os.path.join(best_dir, "test_known_models_probability_alignment_curve.png"))
        fig_acc_align.savefig(os.path.join(best_dir, "test_known_models_acc_alignment_curve.png"))
        fig_calibration.savefig(os.path.join(best_dir, "test_known_models_calibration_curve.png"))
        fig_score_dist.savefig(os.path.join(best_dir, "test_known_models_score_distribution.png"))
        plt.clf()
        plt.close()
        
        logger.info(f"Test summary (best epoch: {best_epoch}) for known models:")
        display_summaries(" ", best_epoch, cfg.max_epochs, test_summary_known, logger)
        
        def pair_to_pairtype(p1, p2, known):
            if (p1 in known) and (p2 in known):
                return "KK"
            elif (p1 not in known) and (p2 not in known):
                return "UU"
            else:
                return "KU"

        target_model_pairs = [(p1, p2, pair_to_pairtype(p1, p2, sorted(list(known_vton_model_ids)))) for p1, p2 in itertools.combinations(cfg.test_target_vton_model_ids, 2)]
        
        pair_micro_and_type_macro = evaluate_pair_micro_and_type_macro(
            test_summary["raw"], 
            target_model_pairs
            )
        logger.info("Pair micro and type macro accuracy:")
        logger.info(
            f"type_macro(KK)={pair_micro_and_type_macro['type_macro']['KK']['macro_acc']:.4f}, "
            f"type_macro(KU)={pair_micro_and_type_macro['type_macro']['KU']['macro_acc']:.4f}, "
            f"type_macro(UU)={pair_micro_and_type_macro['type_macro']['UU']['macro_acc']:.4f}, "
            f"pair_macro={pair_micro_and_type_macro['pair_macro']['macro_acc']:.4f}, "
        )
        
        fig_pair_micro, _ = plot_pair_micro_heatmap_blocks(pair_micro_and_type_macro["pair_micro"], sorted(list(known_vton_model_ids)))
        fig_pair_micro.savefig(os.path.join(best_dir, "test_vton_model_pair_micro_acc_heatmap.png"))


def main():

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
        
    cfg = create_config()
    cfg["cmd"] = get_exec_cmd_as_str()
    cfg["start_time"] = get_now()
    cfg["train_job_id"] = str(uuid.uuid4())
    
    set_random_seed(cfg.seed)
    save_root = os.path.join("outputs", f"{cfg['model']['model_name'].replace('/', '+')}", cfg['start_time'])
    save_root = save_root+f"_{cfg.save_root_postfix}" if cfg.save_root_postfix is not None else save_root
    os.makedirs(save_root, exist_ok=True)    

    OmegaConf.save(cfg, os.path.join(save_root, "config.yaml"))
    
    logger = create_logger(name=__name__, file_name=os.path.join(save_root, "stdout.log"))
    logger.info(f"{OmegaConf.to_yaml(cfg)}")
    logger.info(f"save_root={save_root}")
    logger.info("Training VTON scorer...")
    
    train(cfg, save_root, logger)

if __name__=="__main__":
    main()
