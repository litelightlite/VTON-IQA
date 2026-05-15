import itertools
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor, PreTrainedModel
from scorer.configuration_vtoniqa import VTONScorerConfig

def split_tokens(x: torch.Tensor, num_register_tokens: int):
    """
    x: [B, L, C] -> (cls: [B,1,C] or None, regs: [B,R,C], patches: [B,P,C])
    """
    B, _, C = x.shape
    idx = 0
    cls = x[:, 0:1, :]
    idx = 1
    R = max(0, int(num_register_tokens))
    regs = x[:, idx:idx+R, :] if R > 0 else x.new_zeros(B, 0, C)
    patches = x[:, idx+R:, :]
    return cls, regs, patches

def patch_only(tokens: torch.Tensor, num_register_tokens: int):
    _, _, patches = split_tokens(tokens, num_register_tokens)
    return patches

def patch_and_cls(tokens: torch.Tensor, num_register_tokens: int):
    cls, _, patches = split_tokens(tokens, num_register_tokens)
    patches_and_cls = torch.cat([cls, patches], dim=1)
    return patches_and_cls

class _VisionBackboneAdapter:
    
    def __init__(self, hf_model: nn.Module):
        
        self.model = hf_model
        self.model_type = self.model.config.model_type
        self._last_img_shape = None

        if self.model_type in ["dinov3_vit"]:
            self.blocks = list(self.model.layer)
            self._embed_module = self.model.embeddings
            self._pre_blocks = nn.Identity()
            self._post_blocks = nn.Sequential(
                self.model.norm,
            )
            self._pooler = nn.Identity()
            self.hidden_dim = self.model.config.hidden_size
            self.num_register_tokens = getattr(
                getattr(self.model, "config", None),
                "num_register_tokens",
                0,
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        
        self.num_layers = len(self.blocks)

    def set_image_shape(self, pixel_values: torch.Tensor):
        if pixel_values.dim() != 4 or pixel_values.shape[1] != 3:
            raise ValueError("Expected pixel_values of shape [B,3,H,W].")
        self._last_img_shape = tuple(pixel_values.shape)

    def _rope_pos(self, device: torch.device):
        if self._last_img_shape is None:
            raise ValueError("Call set_image_shape(...) before DINOv3 blocks.")
        B, C, H, W = self._last_img_shape
        dummy = torch.empty((B, C, H, W), device=device)
        return self.model.rope_embeddings(dummy)

    def run_block_attn(self, i: int, x: torch.Tensor):
        
        blk = self.blocks[i]

        residual = x
        hidden_states = blk.norm1(x)

        hidden_states, _ = blk.attention(
            hidden_states,
            attention_mask=None,
            position_embeddings=self._rope_pos(x.device),
            output_attentions=False
        )

        hidden_states = blk.layer_scale1(hidden_states)
        hidden_states = blk.drop_path(hidden_states) + residual

        return hidden_states

    def run_block_mlp(self, i: int, x: torch.Tensor):

        blk = self.blocks[i]
        residual = x
        hidden_states = blk.norm2(x)
        hidden_states = blk.mlp(hidden_states)
        hidden_states = blk.layer_scale2(hidden_states)
        hidden_states = blk.drop_path(hidden_states) + residual

        return hidden_states

    def embed_tokens(self, pixel_values: torch.Tensor):
        return self._embed_module(pixel_values)

    def run_pre_blocks(self, pixel_values: torch.Tensor):
        return self._pre_blocks(pixel_values)

    def run_post_blocks(self, pixel_values: torch.Tensor):
        return self._post_blocks(pixel_values)
    
    def run_pooler(self, pixel_values: torch.Tensor):
        return self._pooler(pixel_values)
    

class CrossAttnOnlyBlock(nn.Module):

    def __init__(self, hidden_dim, heads=16, p: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.ln_q = nn.LayerNorm(hidden_dim)
        self.ln_kv = nn.LayerNorm(hidden_dim)

        self.attn = nn.MultiheadAttention(
            hidden_dim, heads, dropout=p, batch_first=True
        )
        self.drop = nn.Dropout(p)

        self._init_as_noop()

    def forward(self, x, y):

        q = self.ln_q(x)
        kv = self.ln_kv(y)
        attn_out, _ = self.attn(
            q,
            kv,
            kv,
            need_weights=False,
            average_attn_weights=False,
        )
        x = x + self.drop(attn_out)
        return x
        
    def _init_as_noop(self):
        with torch.no_grad():
            self.attn.out_proj.weight.zero_()
            if self.attn.out_proj.bias is not None:
                self.attn.out_proj.bias.zero_()


class PairwiseCrossViewBackbone(nn.Module):

    def __init__(
        self, 
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m", 
        n_mid_c_attn: int = 12, 
        heads: int = 16, 
        p: float = 0.1,
    ):
        super().__init__()

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="eager",
            dtype=torch.bfloat16,
        )
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.adapter = _VisionBackboneAdapter(self.model)
        
        self.n_mid_c_attn = n_mid_c_attn
        self.mid_c_attn_layer_ids = sorted(
            [self.adapter.num_layers - i - 1 for i in range(n_mid_c_attn)]
        )
        
        self.hidden_dim = self.adapter.hidden_dim
        self.num_register_tokens = getattr(
            self.adapter, "num_register_tokens", 0,
        )

        self.mid_p2v_blocks = nn.ModuleDict({
            str(i): CrossAttnOnlyBlock(self.hidden_dim, heads, p)
            for i in self.mid_c_attn_layer_ids
        })
        self.mid_v2p_blocks = nn.ModuleDict({
            str(i): CrossAttnOnlyBlock(self.hidden_dim, heads, p)
            for i in self.mid_c_attn_layer_ids
        })
        self.mid_c2v_blocks = nn.ModuleDict({
            str(i): CrossAttnOnlyBlock(self.hidden_dim, heads, p)
            for i in self.mid_c_attn_layer_ids
        })
        self.mid_v2c_blocks = nn.ModuleDict({
            str(i): CrossAttnOnlyBlock(self.hidden_dim, heads, p)
            for i in self.mid_c_attn_layer_ids
        })

        self._init_cross_attn_blocks()

    def _embed(self, images):
        return self.adapter.embed_tokens(images)
    
    def _run_pre_blocks(self, x):
        return self.adapter.run_pre_blocks(x)
    
    def _run_post_blocks(self, x):
        return self.adapter.run_post_blocks(x)
    
    def _run_pooler(self, x):
        return self.adapter.run_pooler(x)

    def encode_images(self, person, cloth, vton):
        
        dev = next(self.parameters()).device

        p_in = self.processor(images=person, return_tensors="pt").pixel_values.to(dev)
        c_in = self.processor(images=cloth,  return_tensors="pt").pixel_values.to(dev)
        v_in = self.processor(images=vton,   return_tensors="pt").pixel_values.to(dev)

        self.adapter.set_image_shape(p_in)

        p = self._embed(p_in)
        c = self._embed(c_in)
        v = self._embed(v_in)
        
        p = self._run_pre_blocks(p)
        c = self._run_pre_blocks(c)
        v = self._run_pre_blocks(v)

        v_p = v.clone()
        v_c = v.clone()

        for i in range(self.adapter.num_layers):
            p = self.adapter.run_block_attn(i, p)
            c = self.adapter.run_block_attn(i, c)
            v_p = self.adapter.run_block_attn(i, v_p)
            v_c = self.adapter.run_block_attn(i, v_c)

            if (i in self.mid_c_attn_layer_ids) and (0 < self.n_mid_c_attn):
                p = self.mid_v2p_blocks[str(i)](p, v_p)
                v_p  = self.mid_p2v_blocks[str(i)](v_p, p)
                c  = self.mid_v2c_blocks[str(i)](c, v_c)
                v_c = self.mid_c2v_blocks[str(i)](v_c, c)
            p   = self.adapter.run_block_mlp(i, p)
            c   = self.adapter.run_block_mlp(i, c)
            v_p = self.adapter.run_block_mlp(i, v_p)
            v_c = self.adapter.run_block_mlp(i, v_c)
    
        p   = self._run_post_blocks(p)
        c   = self._run_post_blocks(c)
        v_p = self._run_post_blocks(v_p)
        v_c = self._run_post_blocks(v_c)

        p_pool, c_pool, v_p_pool, v_c_pool = p[:, 0, :], c[:, 0, :], v_p[:, 0, :], v_c[:, 0, :]
        
        return p_pool, c_pool, v_p_pool, v_c_pool

            
    def _init_cross_attn_blocks(self):
        with torch.no_grad():
            for blk in itertools.chain(
                self.mid_p2v_blocks.values(),
                self.mid_v2p_blocks.values(),
                self.mid_c2v_blocks.values(),
                self.mid_v2c_blocks.values(),
            ):
                if hasattr(blk, "reset_to_identity"):
                    blk.reset_to_identity()
                else:
                    if hasattr(blk, "proj"):
                        blk.proj.weight.zero_()
                        if blk.proj.bias is not None:
                            blk.proj.bias.zero_()
    

class VTONQualityScorer(nn.Module):
    
    def __init__(
        self,
        learn_branch_weight: bool = True, 
        num_register_tokens: int = 4,
        ):
        super().__init__()
        
        self.learn_branch_weight = learn_branch_weight
        self.num_register_tokens = int(num_register_tokens)        
        if self.learn_branch_weight:
            self.alpha = nn.Parameter(0.5 * torch.ones(()))
        else:
            self.register_buffer("alpha", 0.5 * torch.ones(()))
   
        self.a = nn.Parameter(3.00 * torch.ones(()))
        self.b = nn.Parameter(0.0 * torch.ones(()))
    
    def forward(self, p_pool, c_pool, v_p_pool, v_c_pool) -> torch.Tensor:
        
        s_pv_pool = F.cosine_similarity(p_pool, v_p_pool, dim=-1)
        s_cv_pool = F.cosine_similarity(c_pool, v_c_pool, dim=-1)

        s = self.alpha * s_pv_pool + (1 - self.alpha) * s_cv_pool
        
        s = torch.tanh(self.a * s + self.b)
        
        return s

class VTONScorer(PreTrainedModel):

    config_class = VTONScorerConfig
    base_model_prefix = "vton_scorer"

    def __init__(self, config: VTONScorerConfig):
        super().__init__(config)
        self.config = config
        
        self.backbone = PairwiseCrossViewBackbone(
            model_name=config.model_name,
            n_mid_c_attn=config.n_mid_c_attn,
            heads=config.heads,
            p=config.p_drop,
        )
        
        self.scorer = VTONQualityScorer(
            learn_branch_weight=config.learn_branch_weight,
            num_register_tokens=getattr(self.backbone, "num_register_tokens", 0),
        )
        
        self.temperature_scale = config.temperature_scale
        if config.temperature_scale is None:
            self.log_t = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("log_t", torch.log(self.temperature_scale * torch.ones(())))

    def forward(self, person, cloth, vton) -> Dict[str, torch.Tensor]:
        
        p_pool, c_pool, v_p_pool, v_c_pool = self.backbone.encode_images(person, cloth, vton)

        scores = self.scorer(p_pool, c_pool, v_p_pool, v_c_pool)
        
        return scores
    
    @torch.no_grad()
    def get_win_proba(
        self,
        person, cloth, vton_x, vton_y,
        return_dict: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        
        score_x = self.forward(
            person, cloth, vton_x,
            return_dict=return_dict,
            **kwargs,
        )
        score_y = self.forward(
            person, cloth, vton_y,
            return_dict=return_dict,
            **kwargs,
        )
        
        win_proba_x = torch.sigmoid((score_x - score_y) / self.get_temperature())
        
        return win_proba_x, 1.0 - win_proba_x
    
    def get_temperature(self):
        return torch.exp(self.log_t) + 1e-6
