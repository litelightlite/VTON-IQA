import os
import uuid
import tqdm
import glob
import json
import logging
import statistics
from typing import Dict, Any, List, Literal
import pandas as pd
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from omegaconf import OmegaConf, DictConfig
from scorer.configuration_vtoniqa import VTONScorerConfig
from scorer.modeling_vtoniqa import VTONScorer
from util.commons import set_random_seed, create_logger, create_config, get_exec_cmd_as_str, get_now, DRESSCODE_ROOT, VITON_HD_ROOT
from dataset import VTONIQADataset

HUMAN_TEMPERATURE = 0.65

torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


class QueryPathResolver:
    
    def __init__(self, dataset_type: Literal["dc", "vitonhd"]):   
        self.dataset_type = dataset_type
        self.fname_to_garment_type = {}
        if self.dataset_type in ["dc"]:
            garment_types = ["upper_body", "lower_body", "dresses"]
            for garment_type in garment_types:
                for p in glob.glob(os.path.join(DRESSCODE_ROOT, garment_type, "images", "*")):
                    self.fname_to_garment_type[p.split("/")[-1]]=garment_type

    def __call__(self, vton_path: str) -> Dict[str, Any]:

        if self.dataset_type == "dc":
            person_fname = "_".join(vton_path.split("/")[-1].split("_")[:2])+".jpg"
            person = os.path.join(DRESSCODE_ROOT, self.fname_to_garment_type[person_fname], "images", person_fname)
            garment_fname = "_".join(vton_path.split("/")[-1].split("_")[2:4])+".jpg"
            garment = os.path.join(DRESSCODE_ROOT, self.fname_to_garment_type[garment_fname], "images", garment_fname)
            garment_type = self.fname_to_garment_type[garment_fname]
        elif self.dataset_type == "vitonhd":
            person_fname = "_".join(vton_path.split("/")[-1].split("_")[:2])+".jpg"
            person = os.path.join(VITON_HD_ROOT, "test", "image", person_fname)
            garment_fname = "_".join(vton_path.split("/")[-1].split("_")[2:4])+".jpg"
            garment = os.path.join(VITON_HD_ROOT, "test", "cloth", garment_fname)
            garment_type = "upper_body"

        return {"person": person, "garment": garment, "garment_type": garment_type}


def build_vton_results_info(
    vton_images_root: str,
    dataset_type: Literal["dc", "vitonhd"],
    ) -> List[Dict[str, Any]]:
    
    resolver = QueryPathResolver(dataset_type)
    
    vton_results_info = []
    for vton_path in glob.glob(os.path.join(vton_images_root, "*")):
        query = resolver(vton_path)
        vton_results_info.append(query | {"vton": vton_path})
        
    return vton_results_info


def run(cfg: DictConfig, save_root: str, logger: logging.Logger):
    
    set_random_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vtoniqa = VTONScorer(VTONScorerConfig(**json.load(open(os.path.join(cfg.scorer_model_root, "config.json")))))
    vtoniqa.load_state_dict(load_file(os.path.join(cfg.scorer_model_root, "model.safetensors"))) 
    vtoniqa.to(device) 
    vtoniqa.eval()
    
    if cfg.dataset_type == "dc":
        dataset_root = DRESSCODE_ROOT
    elif cfg.dataset_type == "vitonhd":
        dataset_root = VITON_HD_ROOT
    
    image_size = (512, 512)
    vton_results_info = build_vton_results_info(cfg.vton_images_root, cfg.dataset_type)
    dataset = VTONIQADataset(vton_results_info, expand_2_square=True, align_image_size=True, image_size=image_size)
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=os.cpu_count()-6, pin_memory=True, pin_memory_device="cuda:0", persistent_workers=True, prefetch_factor=4
    )
    
    qa_results = []
    for batch in tqdm.tqdm(dataloader, total=len(dataloader), desc="Assessing VTON Image Quality"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            scores = vtoniqa(batch["person"].to(device), batch["garment"].to(device), batch["vton"].to(device))
        scores = scores.cpu().tolist()
        for i in range(len(scores)):
            qa_results.append({
                "person": batch["person_path"][i].replace(dataset_root+"/", ""), 
                "garment": batch["garment_path"][i].replace(dataset_root+"/", ""), 
                "vton": batch["vton_path"][i].replace(cfg.vton_images_root+"/", ""), 
                "garment_type": batch["garment_type"][i],
                "score": scores[i]
                })
    
    meta = {"scorer": cfg.scorer_model_root, "vton_images_root": cfg.vton_images_root}    
    
    summary_by_garment_category = pd.DataFrame(qa_results).groupby("garment_type")["score"].mean().to_dict()
    summary = {"overall": statistics.mean([r["score"] for r in qa_results])} | summary_by_garment_category
    
    logger.info(f"Summary of VTON Quality Assessment: {summary}")

    qa_results = {"meta": meta, "summary": summary, "results": qa_results}
    
    with open(os.path.join(save_root, "vtoniqa.json"), "w") as f:
        json.dump(qa_results, f, indent=4)

def main():

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
        
    cfg = create_config()
    cfg["cmd"] = get_exec_cmd_as_str()
    cfg["start_time"] = get_now()
    cfg["train_job_id"] = str(uuid.uuid4())
    
    set_random_seed(cfg.seed)
    save_root = os.path.join("outputs", "vtoniqa", cfg.vton_images_root.replace("/", "+"))
    save_root = save_root+f"_{cfg.save_root_postfix}" if cfg.save_root_postfix is not None else save_root
    os.makedirs(save_root, exist_ok=True)    

    OmegaConf.save(cfg, os.path.join(save_root, "config.yaml"))
    
    logger = create_logger(name=__name__, file_name=os.path.join(save_root, "stdout.log"))
    logger.info(f"{OmegaConf.to_yaml(cfg)}")
    logger.info(f"save_root={save_root}")
    logger.info("Quality Assessment of VTON Results...")
    
    run(cfg, save_root, logger)
    
if __name__=="__main__":
    main()
