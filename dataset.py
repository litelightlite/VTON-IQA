import os
import random
from typing import Literal, Optional, List, Dict, Any, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, get_worker_info
import torchvision
import torchvision.transforms.functional as tvf
from torchvision.io import read_image
from util.commons import DRESSCODE_ROOT, VITON_HD_ROOT, SYNTH_VTON_ROOT, VTONQBENCH_ROOT, VTON_MODEL_IDS, DATASET_TYPES
from util.curation import load_reward_dataset_as_df

def convert_label_id_to_score(label_id: Literal[0, 1, 2]) -> float:
    if label_id == 0:
        return 1.0
    elif label_id == 1:
        return 2.0
    elif label_id == 2:
        return 3.0
    else:
        return np.nan

def resolve_dataset_root(
    dataset_type: Literal["synth_vton", "dc", "vitonhd", "vton_evals_dc", "vton_evals_vitonhd"],
    image_type: Literal["person", "garment", "ref_person"] = "garment"
    ) -> str:

    if dataset_type in ["synth_vton"]:
        return SYNTH_VTON_ROOT
    elif dataset_type in ["dc"]:
        return DRESSCODE_ROOT
    elif dataset_type in ["vitonhd"]:
        return VITON_HD_ROOT
    
    if dataset_type in ["vton_evals_dc"]:
        if image_type=="person":
            return VTONQBENCH_ROOT
        elif image_type in ["garment", "ref_person"]:
            return DRESSCODE_ROOT

    if dataset_type in ["vton_evals_vitonhd"]:
        if image_type=="person":
            return VTONQBENCH_ROOT
        elif image_type in ["garment", "ref_person"]:
            return VITON_HD_ROOT
        
def seed_worker(worker_id: int):
    base_seed = torch.initial_seed()  # 64-bit
    random.seed(base_seed)
    np.random.seed(base_seed % (2**32))

    wi = get_worker_info()
    if wi is not None and hasattr(wi.dataset, "rng"):
        g = torch.Generator()
        g.manual_seed(base_seed)
        wi.dataset.rng = g

def z_score_standardization(df: pd.DataFrame) -> pd.DataFrame:
    df["reward"] = (df["reward"] - df["reward"].mean()) / df["reward"].std()
    return df

def min_max_standardize(df: pd.DataFrame) -> pd.DataFrame:
    df["reward"] = (df["reward"] - df["reward"].min()) / (df["reward"].max() - df["reward"].min())
    return df

def robust_standardize(df: pd.DataFrame) -> pd.DataFrame:
    median = df["reward"].median()
    mad = (df["reward"] - median).abs().median()
    df["reward"] = (df["reward"] - median) / (mad + 1e-8)
    return df


class BaseRewardDataset(Dataset):
    
    def __init__(
        self, 
        reward_dataset_path: Optional[str] = None,
        target_vton_model_ids: List[Literal[
            "fitdit", "any2any", "catvton_flux", "catvton", "idm", "oot", "vitonhd", 
            "hrviton", "sdviton", "qwen_edit", "nanobanana", "openai", "catdm", "ladi"
            ]] = VTON_MODEL_IDS,   
        target_dataset_types: List[Literal[
            "synth_vton", "dc", "vitonhd", "vton_evals_dc", "vton_evals_vitonhd"
            ]] = DATASET_TYPES,
        curation: bool = True,
        drop_dummies: bool = True,
        standardization_type: Optional[Literal["minmax", "standard", "robust"]] = None,
        with_metadata: bool = False,
        real_augmentation: bool = False
        ):
        self.reward_dataset_path = reward_dataset_path
        self.target_vton_model_ids = target_vton_model_ids
        self.target_dataset_types = target_dataset_types
        self.curation = curation
        self.drop_dummies = drop_dummies
        self.standardization_type = standardization_type
        self.with_metadata = with_metadata
        self.real_augmentation = real_augmentation
        if self.standardization_type == "standard":
            self.scaler = StandardScaler()
            self.scaler_lpips = StandardScaler()
            self.scaler_ssim = StandardScaler()
        elif self.standardization_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler_lpips = MinMaxScaler(feature_range=(-1, 1))
            self.scaler_ssim = MinMaxScaler(feature_range=(-1, 1))
        elif self.standardization_type == "robust":
            self.scaler = RobustScaler()
            self.scaler_lpips = RobustScaler()
            self.scaler_ssim = RobustScaler()
        elif self.standardization_type is None:
            self.scaler = None
        self.dataset = self._setup_dataset()
        self.n = len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        data_processed = dict(
            person=data["person"], garment=data["garment"], vton=data["vton"], 
            )
        
        if "reward" in data.keys():
            data_processed = data_processed | dict(reward=data["reward"], votes=torch.tensor(data["votes"]))        
        
        if self.with_metadata:
            data_processed = data_processed | dict(
            form_id=data["form_id"], problem_id=data["problem_id"], 
            dataset_type=data["dataset_type"], vton_model_id=data["vton_model_id"], garment_type=data["garment_type"],
            annotator_ids=data["annotator_ids"], answer_ids=data["answer_ids"], 
            ssim=data["ssim"], lpips=data["lpips"]
            )
        
        return data_processed
    
    def __len__(self):
        return self.n
    
    def _setup_dataset(self) -> List[Dict[str, Any]]:
        
        if self.reward_dataset_path is None:
            raw_data = load_reward_dataset_as_df(self.curation, self.drop_dummies)
        else:
            raw_data = pd.read_json(self.reward_dataset_path)
        
        raw_data = raw_data.to_dict(orient="records")
        
        dataset = []
        for record in raw_data:
            if self._is_target_record(record):
                person = os.path.join(resolve_dataset_root(record["dataset_type"], "person"), record["person_path"])
                garment = os.path.join(resolve_dataset_root(record["dataset_type"], "garment"), record["garment_path"])
                vton = os.path.join(VTONQBENCH_ROOT, record["vton_path"])
                dataset.append(dict(
                    form_id=record["form_id"], problem_id=record["problem_id"], 
                    dataset_type=record["dataset_type"], vton_model_id=record["vton_model_id"], garment_type=record["garment_type"],
                    person=person, garment=garment, vton=vton, 
                    reward=record["reward"], votes=record["votes"],
                    annotator_ids=record.get("annotator_ids"), answer_ids=record.get("answer_ids"),
                    ssim=record.get("ssim"), lpips=record.get("lpips")
                    ))
        
                if self.real_augmentation and record["dataset_type"] in ["vton_evals_dc", "vton_evals_vitonhd"]:
                    ref_person = os.path.join(resolve_dataset_root(record["dataset_type"], "ref_person"), record["ref_person_path"])
                    dataset.append(dict(
                        form_id=record["form_id"], problem_id=record["problem_id"], 
                        dataset_type=record["dataset_type"].split("_")[-1], vton_model_id=record["vton_model_id"], garment_type=record["garment_type"],
                        person=person, garment=garment, vton=ref_person, 
                        reward=3.0, votes=[0, 0, 0],
                        annotator_ids=[], answer_ids=[],
                        ))

        
        if self.scaler is not None:
            dataset = pd.DataFrame(dataset)
            dataset["reward"] = self.scaler.fit_transform(dataset[["reward"]])
            dataset["lpips"] = self.scaler_lpips.fit_transform(dataset[["lpips"]])
            dataset["ssim"] = self.scaler_ssim.fit_transform(dataset[["ssim"]])
            dataset = dataset.to_dict(orient="records")
        else:
            pass
        
        return dataset
        
    def _is_target_record(self, record: Dict[str, Any]) -> bool:
        c1 = True if record["dataset_type"] in self.target_dataset_types else False
        c2 = True if record["vton_model_id"] in self.target_vton_model_ids else False
        return c1 and c2
        
    def set_scaler(self, scaler: Union[StandardScaler, MinMaxScaler, RobustScaler]) -> None:
        self.scaler = scaler
        self.dataset = self._setup_dataset()
        
    def get_scaler(self) -> Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]]:
        return self.scaler
        
class BaseRewardComparisonDataset(Dataset):
    
    def __init__(
        self, 
        reward_dataset_path: Optional[str] = None,
        target_vton_model_ids: List[Literal[
            "fitdit", "any2any", "catvton_flux", "catvton", "idm", "oot", "vitonhd", 
            "hrviton", "sdviton", "qwen_edit", "nanobanana", "openai", "catdm", "ladi"
            ]] = VTON_MODEL_IDS,   
        target_dataset_types: List[Literal[
            "synth_vton", "dc", "vitonhd", "vton_evals_dc", "vton_evals_vitonhd"
            ]] = DATASET_TYPES,
        curation: bool = True,
        drop_dummies: bool = True,
        standardization_type: Optional[Literal["minmax", "standard", "robust"]] = None,
        with_metadata: bool = False,
        temperature: float = 0.9,
        real_augmentation: bool = False
        ):
        self.reward_dataset_path = reward_dataset_path
        self.target_vton_model_ids = target_vton_model_ids
        self.target_dataset_types = target_dataset_types
        self.curation = curation
        self.drop_dummies = drop_dummies
        self.standardization_type = standardization_type
        self.with_metadata = with_metadata
        self.real_augmentation = real_augmentation
        if self.standardization_type == "standard":
            self.scaler = StandardScaler()
        elif self.standardization_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif self.standardization_type == "robust":
            self.scaler = RobustScaler()
        elif self.standardization_type is None:
            self.scaler = None
        self.temperature = temperature
        self.rng = None  # worker_init_fn でセットされる想定
        self.dataset = self._setup_dataset()
        self.n = len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]

        if self.rng is None:
            seed = torch.initial_seed()
            self.rng = torch.Generator().manual_seed(seed)
            
        if 1 < len(data["reward"]):
            xid, yid = torch.randperm(len(data["reward"]), generator=self.rng)[:2].tolist()
            is_identical_pair = dict(is_identical_pair=False)
        else:
            xid, yid = 0, 0
            is_identical_pair = dict(is_identical_pair=True)
        
        data_processed_common = dict(person=data["person"], garment=data["garment"])
        
        data_processed_x = dict(
            vton_x=data["vton"][xid], reward_x=data["reward"][xid], votes_x=torch.tensor(data["votes"][xid]),
            )
        
        data_processed_y = dict(
            vton_y=data["vton"][yid], reward_y=data["reward"][yid], votes_y=torch.tensor(data["votes"][yid]),
            )
        
        label = dict(label=torch.sigmoid(torch.tensor(data_processed_x["reward_x"] - data_processed_y["reward_y"]) / self.temperature).item())
        
        data_processed = data_processed_common | data_processed_x | data_processed_y | label | is_identical_pair
        
        if self.with_metadata:
            data_processed = data_processed | dict(
            form_id_x=data["form_id"][xid], problem_id_x=data["problem_id"][xid], 
            dataset_type_x=data["dataset_type"][xid], vton_model_id_x=data["vton_model_id"][xid], 
            garment_type=data["garment_type"][xid],
            form_id_y=data["form_id"][yid], problem_id_y=data["problem_id"][yid], 
            dataset_type_y=data["dataset_type"][yid], vton_model_id_y=data["vton_model_id"][yid],
            annotator_ids_x=data["annotator_ids"][xid], annotator_ids_y=data["annotator_ids"][yid],
            answer_ids_x=data["answer_ids"][xid], answer_ids_y=data["answer_ids"][yid],
            )
        
        return data_processed
    
    def __len__(self):
        return self.n
    
    def _setup_dataset(self) -> List[Dict[str, Any]]:
        
        if self.reward_dataset_path is None:
            raw_data = load_reward_dataset_as_df(self.curation, self.drop_dummies)
        else:
            raw_data = pd.read_json(self.reward_dataset_path)
            
        raw_data = raw_data.to_dict(orient="records")
                
        dataset = []
        for record in raw_data:
            if self._is_target_record(record):
                person = os.path.join(resolve_dataset_root(record["dataset_type"], "person"), record["person_path"])
                garment = os.path.join(resolve_dataset_root(record["dataset_type"], "garment"), record["garment_path"])
                vton = os.path.join(VTONQBENCH_ROOT, record["vton_path"])
                
                dataset.append(dict(
                    form_id=record["form_id"], problem_id=record["problem_id"], 
                    dataset_type=record["dataset_type"], vton_model_id=record["vton_model_id"], garment_type=record["garment_type"],
                    person=person, garment=garment, vton=vton, 
                    reward=record["reward"], votes=record["votes"],
                    annotator_ids=record.get("annotator_ids"), answer_ids=record.get("answer_ids"),
                    ssim=record.get("ssim"), lpips=record.get("lpips")
                    ))
                
                if self.real_augmentation and record["dataset_type"] in ["vton_evals_dc", "vton_evals_vitonhd"]:
                    ref_person = os.path.join(resolve_dataset_root(record["dataset_type"], "ref_person"), record["ref_person_path"])
                    dataset.append(dict(
                        form_id=record["form_id"], problem_id=record["problem_id"], 
                        dataset_type=record["dataset_type"].split("_")[-1], vton_model_id=record["vton_model_id"], garment_type=record["garment_type"],
                        person=person, garment=garment, vton=ref_person, 
                        reward=3.0, votes=[0, 0, 0],
                        annotator_ids=[], answer_ids=[],
                        ))
        
        if self.scaler is not None:
            dataset = pd.DataFrame(dataset)
            dataset["reward"] = self.scaler.fit_transform(dataset[["reward"]])
            dataset = dataset.to_dict(orient="records")
        
        df = pd.DataFrame(dataset)
        dataset = df.groupby(["person", "garment"]).agg(list).reset_index().to_dict(orient="records")
        
        return dataset
        
    def _is_target_record(self, record: Dict[str, Any]) -> bool:
        c1 = True if record["dataset_type"] in self.target_dataset_types else False
        c2 = True if record["vton_model_id"] in self.target_vton_model_ids else False
        return c1 and c2

    def set_scaler(self, scaler: Union[StandardScaler, MinMaxScaler, RobustScaler]) -> None:
        self.scaler = scaler
        self.dataset = self._setup_dataset()
        
    def get_scaler(self) -> Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]]:
        return self.scaler


class BaseGroupedRewardDataset(Dataset):
    
    def __init__(
        self, 
        reward_dataset_path: Optional[str] = None,
        target_vton_model_ids: List[Literal[
            "fitdit", "any2any", "catvton_flux", "catvton", "idm", "oot", "vitonhd", 
            "hrviton", "sdviton", "qwen_edit", "nanobanana", "openai", "catdm", "ladi"
            ]] = VTON_MODEL_IDS,   
        target_dataset_types: List[Literal[
            "synth_vton", "dc", "vitonhd", "vton_evals_dc", "vton_evals_vitonhd"
            ]] = DATASET_TYPES,
        curation: bool = True,
        drop_dummies: bool = True,
        standardization_type: Optional[Literal["minmax", "standard", "robust"]] = None,
        with_metadata: bool = False,
        ):
        self.reward_dataset_path = reward_dataset_path
        self.target_vton_model_ids = target_vton_model_ids
        self.target_dataset_types = target_dataset_types
        self.curation = curation
        self.drop_dummies = drop_dummies
        self.standardization_type = standardization_type
        self.with_metadata = with_metadata
        if self.standardization_type == "standard":
            self.scaler = StandardScaler()
        elif self.standardization_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        elif self.standardization_type == "robust":
            self.scaler = RobustScaler()
        elif self.standardization_type is None:
            self.scaler = None
        self.dataset = self._setup_dataset()
        self.n = len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        
        data_processed = dict(
            person=data["person"], garment=data["garment"], vton=data["vton"], 
            reward=torch.tensor(data["reward"]), votes=torch.tensor(data["votes"]), group_id=data["group_id"]
            )
        
        if self.with_metadata:
            data_processed = data_processed | dict(
            form_id=data["form_id"], problem_id=data["problem_id"], 
            dataset_type=data["dataset_type"], vton_model_id=data["vton_model_id"], garment_type=data["garment_type"],
            annotator_ids=data["annotator_ids"], answer_ids=data["answer_ids"],
            )
            
        return data_processed
    
    def __len__(self):
        return self.n
    
    def _setup_dataset(self) -> List[Dict[str, Any]]:
        
        if self.reward_dataset_path is None:
            raw_data = load_reward_dataset_as_df(self.curation, self.drop_dummies)
        else:
            raw_data = pd.read_json(self.reward_dataset_path)
            
        raw_data = raw_data.to_dict(orient="records")
                
        dataset = []
        for group_id, record in enumerate(raw_data):
            if self._is_target_record(record):
                person = os.path.join(resolve_dataset_root(record["dataset_type"], "person"), record["person_path"])
                garment = os.path.join(resolve_dataset_root(record["dataset_type"], "garment"), record["garment_path"])
                vton = os.path.join(VTONQBENCH_ROOT, record["vton_path"])
                dataset.append(dict(
                    form_id=record["form_id"], problem_id=record["problem_id"], 
                    dataset_type=record["dataset_type"], vton_model_id=record["vton_model_id"], garment_type=record["garment_type"],
                    person=person, garment=garment, vton=vton, 
                    reward=record["reward"], votes=record["votes"],
                    annotator_ids=record.get("annotator_ids"), answer_ids=record.get("answer_ids"),
                    ))
        
        if self.scaler is not None:
            dataset = pd.DataFrame(dataset)
            dataset["reward"] = self.scaler.fit_transform(dataset[["reward"]])
            dataset = dataset.to_dict(orient="records")
        
        df = pd.DataFrame(dataset)
        dataset = df.groupby(["person", "garment"]).agg(list).reset_index().to_dict(orient="records")
        
        for group_id in range(len(dataset)):
            dataset[group_id]["group_id"] = group_id
        
        return dataset
        
    def _is_target_record(self, record: Dict[str, Any]) -> bool:
        c1 = True if record["dataset_type"] in self.target_dataset_types else False
        c2 = True if record["vton_model_id"] in self.target_vton_model_ids else False
        return c1 and c2

    def set_scaler(self, scaler: Union[StandardScaler, MinMaxScaler, RobustScaler]) -> None:
        self.scaler = scaler
        self.dataset = self._setup_dataset()
        
    def get_scaler(self) -> Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]]:
        return self.scaler


def pad_to_square_tensor(x: torch.Tensor) -> torch.Tensor:
    is_batched = (x.dim() == 4)
    if not is_batched:
        x = x.unsqueeze(0)  # -> BCHW

    _, _, h, w = x.shape
    if h == w:
        return x if is_batched else x.squeeze(0)

    size = max(h, w)
    pad_h_total = size - h
    pad_w_total = size - w
    pad_left  = pad_w_total // 2
    pad_right = pad_w_total - pad_left
    pad_top   = pad_h_total // 2
    pad_bottom= pad_h_total - pad_top

    x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
    return x_padded if is_batched else x_padded.squeeze(0)


def resize_to_match(image1: torch.Tensor, image2: torch.Tensor, mode="bilinear"):
    if image2.dim() == 3:
        _, h2, w2 = image2.shape
    elif image2.dim() == 4:
        _, _, h2, w2 = image2.shape
    else:
        raise ValueError("image2 must be CHW or BCHW tensor")

    if image1.dim() == 3:
        image1 = image1.unsqueeze(0)  # -> BCHW
        resized = F.interpolate(image1, size=(h2, w2), mode=mode, align_corners=False)
        return resized.squeeze(0)
    else:
        return F.interpolate(image1, size=(h2, w2), mode=mode, align_corners=False)
    
    
class RewardRegressionDataset(Dataset):
    
    def __init__(
        self, 
        expand_2_square: bool = True,
        align_image_size: bool = True,
        image_size: Optional[Tuple[int, int]] = None,
        reward_dataset_path: Optional[str] = None,
        target_vton_model_ids: List[Literal[
            "fitdit", "any2any", "catvton_flux", "catvton", "idm", "oot", "vitonhd", 
            "hrviton", "sdviton", "qwen_edit", "nanobanana", "openai", "catdm", "ladi"
            ]] = VTON_MODEL_IDS,    
        target_dataset_types: List[Literal[
            "synth_vton", "dc", "vitonhd", "vton_evals_dc", "vton_evals_vitonhd"
            ]] = DATASET_TYPES,
        curation: bool = True,
        drop_dummies: bool = True,
        standardization_type: Optional[Literal["standard", "minmax", "robust"]] = None, 
        with_metadata: bool = False,
        real_augmentation: bool = False,
        ):
        self.expand_2_square = expand_2_square
        self.align_image_size = align_image_size
        self.image_size = image_size
        self.dataset = BaseRewardDataset(reward_dataset_path, target_vton_model_ids, target_dataset_types, curation, drop_dummies, standardization_type, with_metadata, real_augmentation)
        self.n = len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        person = read_image(data["person"], mode=torchvision.io.ImageReadMode.RGB)
        garment = read_image(data["garment"], mode=torchvision.io.ImageReadMode.RGB)
        vton = read_image(data["vton"], mode=torchvision.io.ImageReadMode.RGB)
        if self.expand_2_square:
            person = pad_to_square_tensor(person)
            garment = pad_to_square_tensor(garment)
            vton = pad_to_square_tensor(vton)
        if self.align_image_size:
            vton = resize_to_match(vton, person, mode="bilinear")
        if self.image_size is not None:
            person = tvf.resize(person, size=self.image_size, interpolation=tvf.InterpolationMode.BICUBIC)
            garment = tvf.resize(garment, size=self.image_size, interpolation=tvf.InterpolationMode.BICUBIC)
            vton = tvf.resize(vton, size=self.image_size, interpolation=tvf.InterpolationMode.BICUBIC)

        processed_data = {k: v for k, v in data.items() if k not in ["person", "garment", "vton"]} | {"person": person, "garment": garment, "vton": vton}
      
        return processed_data
    
    def __len__(self):
        return self.n
    
    def set_scaler(self, scaler: Union[StandardScaler, MinMaxScaler, RobustScaler]) -> None:
        self.dataset.set_scaler(scaler)
        
    def get_scaler(self) -> Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]]:
        return self.dataset.get_scaler()
    
class RewardComparisonDataset(Dataset):
    
    def __init__(
        self, 
        expand_2_square: bool = True,
        align_image_size: bool = True,
        image_size: Optional[Tuple[int, int]] = None,
        reward_dataset_path: Optional[str] = None,
        target_vton_model_ids: List[Literal[
            "fitdit", "any2any", "catvton_flux", "catvton", "idm", "oot", "vitonhd", 
            "hrviton", "sdviton", "qwen_edit", "nanobanana", "openai", "catdm", "ladi"
            ]] = VTON_MODEL_IDS,   
        target_dataset_types: List[Literal[
            "synth_vton", "dc", "vitonhd", "vton_evals_dc", "vton_evals_vitonhd"
            ]] = DATASET_TYPES,
        curation: bool = True,
        drop_dummies: bool = True,
        standardization_type: Optional[Literal["standard", "minmax", "robust"]] = None, 
        with_metadata: bool = False,
        temperature: float = 0.65,
        real_augmentation: bool = False,
        ):
        self.expand_2_square = expand_2_square
        self.align_image_size = align_image_size
        self.image_size = image_size
        self.dataset = BaseRewardComparisonDataset(reward_dataset_path, target_vton_model_ids, target_dataset_types, curation, drop_dummies, standardization_type, with_metadata, temperature, real_augmentation)
        self.n = len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        person = read_image(data["person"], mode=torchvision.io.ImageReadMode.RGB)
        garment = read_image(data["garment"], mode=torchvision.io.ImageReadMode.RGB)
        vton_x = read_image(data["vton_x"], mode=torchvision.io.ImageReadMode.RGB)
        vton_y = read_image(data["vton_y"], mode=torchvision.io.ImageReadMode.RGB)
        if self.expand_2_square:
            person = pad_to_square_tensor(person)
            garment = pad_to_square_tensor(garment)
            vton_x = pad_to_square_tensor(vton_x)
            vton_y = pad_to_square_tensor(vton_y)
        if self.align_image_size:
            vton_x = resize_to_match(vton_x, person, mode="bilinear")
            vton_y = resize_to_match(vton_y, person, mode="bilinear")
        if self.image_size is not None:
            person = tvf.resize(person, size=self.image_size, interpolation=tvf.InterpolationMode.BICUBIC)
            garment = tvf.resize(garment, size=self.image_size, interpolation=tvf.InterpolationMode.BICUBIC)
            vton_x = tvf.resize(vton_x, size=self.image_size, interpolation=tvf.InterpolationMode.BICUBIC)
            vton_y = tvf.resize(vton_y, size=self.image_size, interpolation=tvf.InterpolationMode.BICUBIC)

        processed_data = {k: v for k, v in data.items() if k not in ["person", "garment", "vton_x", "vton_y"]} | {"person": person, "garment": garment, "vton_x": vton_x, "vton_y": vton_y}

        return processed_data
    
    def __len__(self):
        return self.n
    
    def set_scaler(self, scaler: Union[StandardScaler, MinMaxScaler, RobustScaler]) -> None:
        self.dataset.set_scaler(scaler)
        
    def get_scaler(self) -> Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]]:
        return self.dataset.get_scaler()


class GroupedRewardDataset(Dataset):
    
    def __init__(
        self, 
        expand_2_square: bool = True,
        align_image_size: bool = True,
        image_size: Optional[Tuple[int, int]] = None,
        reward_dataset_path: Optional[str] = None,
        target_vton_model_ids: List[Literal[
            "fitdit", "any2any", "catvton_flux", "catvton", "idm", "oot", "vitonhd", 
            "hrviton", "sdviton", "qwen_edit", "nanobanana", "openai", "catdm", "ladi"
            ]] = VTON_MODEL_IDS,   
        target_dataset_types: List[Literal[
            "synth_vton", "dc", "vitonhd", "vton_evals_dc", "vton_evals_vitonhd"
            ]] = DATASET_TYPES,
        curation: bool = True,
        drop_dummies: bool = True,
        standardization_type: Optional[Literal["standard", "minmax", "robust"]] = None, 
        with_metadata: bool = False,
        ):
        self.expand_2_square = expand_2_square
        self.align_image_size = align_image_size
        self.image_size = image_size
        self.dataset = BaseGroupedRewardDataset(reward_dataset_path, target_vton_model_ids, target_dataset_types, curation, drop_dummies, standardization_type, with_metadata)
        self.n = len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        person = read_image(data["person"], mode=torchvision.io.ImageReadMode.RGB)
        garment = read_image(data["garment"], mode=torchvision.io.ImageReadMode.RGB)
        vton = torch.stack([tvf.resize(read_image(path, mode=torchvision.io.ImageReadMode.RGB), size=person.shape[1:], interpolation=tvf.InterpolationMode.BICUBIC) for path in data["vton"]])

        if self.expand_2_square:
            person = pad_to_square_tensor(person)
            garment = pad_to_square_tensor(garment)
            vton = pad_to_square_tensor(vton)
        if self.align_image_size:
            vton = resize_to_match(vton, person, mode="bilinear")
        if self.image_size is not None:
            person = tvf.resize(person, size=self.image_size, interpolation=tvf.InterpolationMode.BICUBIC)
            garment = tvf.resize(garment, size=self.image_size, interpolation=tvf.InterpolationMode.BICUBIC)
            vton = tvf.resize(vton, size=self.image_size, interpolation=tvf.InterpolationMode.BICUBIC)
            
        person = person.unsqueeze(0).expand(vton.shape[0], -1, -1, -1)
        garment = garment.unsqueeze(0).expand(vton.shape[0], -1, -1, -1)
        group_id = torch.tensor(data["group_id"]).expand(vton.shape[0])
        processed_data = {k: v for k, v in data.items() if k not in ["person", "garment", "vton", "group_id"]} | {"person": person, "garment": garment, "vton": vton, "group_id": group_id}
        
        return processed_data
    
    def __len__(self):
        return self.n
    
    def set_scaler(self, scaler: Union[StandardScaler, MinMaxScaler, RobustScaler]) -> None:
        self.dataset.set_scaler(scaler)
        
    def get_scaler(self) -> Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]]:
        return self.dataset.get_scaler()


class VTONIQADataset(Dataset):
    
    def __init__(
        self,
        vton_results_info: List[Dict[str, Any]],
        expand_2_square: bool = True,
        align_image_size: bool = True,
        image_size: Optional[Tuple[int, int]] = None,
        ):
        self.vton_results_info = vton_results_info
        self.expand_2_square = expand_2_square
        self.align_image_size = align_image_size
        self.image_size = image_size
        self.n = len(self.vton_results_info)
    
    def __getitem__(self, index):
        
        person = read_image(self.vton_results_info[index]["person"], mode=torchvision.io.ImageReadMode.RGB) 
        garment = read_image(self.vton_results_info[index]["garment"], mode=torchvision.io.ImageReadMode.RGB)
        vton = read_image(self.vton_results_info[index]["vton"], mode=torchvision.io.ImageReadMode.RGB)
        
        if self.expand_2_square:
            person = pad_to_square_tensor(person)
            garment = pad_to_square_tensor(garment)
            vton = pad_to_square_tensor(vton)
        if self.align_image_size:
            vton = resize_to_match(vton, person, mode="bilinear")
        if self.image_size is not None:
            person = tvf.resize(person, size=self.image_size, interpolation=tvf.InterpolationMode.BICUBIC)
            garment = tvf.resize(garment, size=self.image_size, interpolation=tvf.InterpolationMode.BICUBIC)
            vton = tvf.resize(vton, size=self.image_size, interpolation=tvf.InterpolationMode.BICUBIC)
        
        processed_data = {
            "person": person, "garment": garment, "vton": vton, 
            "person_path": self.vton_results_info[index]["person"],
            "garment_path": self.vton_results_info[index]["garment"],
            "vton_path": self.vton_results_info[index]["vton"],
            "garment_type": self.vton_results_info[index]["garment_type"],
            }
        
        return processed_data
    
    def __len__(self):
        return self.n
    