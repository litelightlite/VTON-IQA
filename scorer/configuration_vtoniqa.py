from typing import Optional
from transformers import PretrainedConfig

class VTONScorerConfig(PretrainedConfig):
    
    model_type = "vton_scorer"

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        n_mid_s_attn: int = 12,
        n_mid_c_attn: int = 12,
        heads: int = 16,
        p_drop: float = 0.1,
        temperature_scale: Optional[float] = 0.65,
        learn_branch_weight: bool = True,
        **kwargs,
        ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.n_mid_s_attn = n_mid_s_attn
        self.n_mid_c_attn = n_mid_c_attn
        self.heads = heads
        self.p_drop = p_drop
        self.temperature_scale = temperature_scale
        self.learn_branch_weight = learn_branch_weight
