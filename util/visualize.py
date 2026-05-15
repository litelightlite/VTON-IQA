import re
import os
import math
import itertools
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple, Literal, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.calibration import calibration_curve
from util.commons import VTONQBENCH_ROOT, DRESSCODE_ROOT, VITON_HD_ROOT

FIGSIZE_1COL = (3.3, 2.2)
FIGSIZE_2COL = (6.9, 3.0)

def setup_mpl():
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Computer Modern Roman"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

def sigmoid(x):
    return 1 / (1 + math.exp(-x))  

def plot_calibration_curve(
    test_results_raw: List[Dict[str, Any]],
    model_temperature: float,
    ece: float, 
    brier: float,
    n_bins = 10
    ):
    
    probas, targets = [], []
    for r in test_results_raw:
        if 1 < len(r["prediction"]): 
            for (i, si), (j, sj) in itertools.combinations(enumerate(r["prediction"]), 2):
                probas.append(sigmoid((si-sj) / model_temperature))
                targets.append(0 if r["reward"][i] < r["reward"][j] else 1)
                
    probas, targets = np.array(probas), np.array(targets)
        
    prob_true, prob_pred = calibration_curve(targets, probas, n_bins=n_bins, strategy='uniform')
    
    fig, ax = plt.subplots(figsize=FIGSIZE_2COL)
    ax.plot(prob_pred, prob_true, "o-", label=f"Model (ECE={ece:.3f}, Brier={brier:.3f})", color="C0")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration (y=x)")
    ax.set_xlabel(r'$\sigma(R(V_1)-R(V_2))$')
    ax.set_ylabel(r'#$V_{2}\prec V_{1}$')
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")  # 45°線が真正面になるよう比率を固定
    plt.tight_layout()
    
    return fig, ax

def plot_probability_alignment_curve(
    align_means: List[float], 
    palign_stdevs: List[float], 
    lbs: List[float],
    label: str,
    n_instances: Optional[List[int]] = None,
    ):
    
    fig, ax = plt.subplots()
    
    bin_size = round(float(lbs[1]-lbs[0]), 3)
    centers = [lb + bin_size/2 for lb in lbs]

    # --- main errorbar ---
    ax.errorbar(
        centers,
        align_means, 
        yerr=palign_stdevs,
        fmt='o-',
        linewidth=2,
        label=label,
        zorder=3
    )

    # calibrated line
    ax.plot(
        centers, centers,
        linestyle='--',
        color='red',
        linewidth=1.5,
        label="calibrated",
        zorder=2
    )

    #ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Human Probability of GT Winner wins")
    ax.set_ylabel("Model Probability of GT Winner wins")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # --- optional barplot (secondary axis) ---
    if n_instances is not None:
        ax2 = ax.twinx()

        ax2.bar(
            centers,
            n_instances,
            width=0.90 * bin_size,
            alpha=0.25,           # ← 薄くするポイント
            facecolor="none",
            edgecolor="gray",
            linewidth=1.5,
            label="#Instances",
            zorder=1
        )

        ax2.set_ylabel("#Instances")
        ax2.tick_params(axis='y', labelsize=8)
        ax2.set_ylim(0, max(n_instances) * 1.2)

    # --- legend ---
    handles, labels_ = ax.get_legend_handles_labels()
    if n_instances is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels_ += l2

    ax.legend(handles, labels_, frameon=False, loc="upper left")

    fig.tight_layout()
    return fig, ax

def plot_pairwise_acc_alignment_curve(
    accs: List[float], 
    lbs: List[float],
    label: str
    ):

    fig, ax = plt.subplots()
    ax.plot(
        lbs,
        accs, 
        label=label,
        marker="o"
    )
    ax.plot(lbs, lbs, marker="o", linestyle='--', color='red', label="baseline")
    ax.set_aspect('equal', adjustable='box')
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlabel("Human Probability of GT Winner wins")
    ax.set_ylabel("Model Pairwise Accuracy")
    fig.tight_layout()
    return fig, ax


def plot_score_distribution(test_results_raw: List[Dict[str, Any]]):
    
    r = list(itertools.chain.from_iterable([data["reward"] for data in test_results_raw]))
    p = list(itertools.chain.from_iterable([data["prediction"] for data in test_results_raw]))
    
    fig, ax = plt.subplots()
    ax.hist(r, bins=10, alpha=0.5, label="reward", histtype="step")
    ax.hist(p, bins=10, alpha=0.5, label="prediction", histtype="step")
    ax.legend()
    ax.set_xlabel("Score")
    fig.tight_layout()
    return fig, ax


def plot_pair_micro_heatmap_blocks(
    pair_micro: dict,
    known: list,
    annotate: bool = True,
    fmt: str = "{:.2f}",
    figsize=(12, 10),
    vmin: float = 0.5,
    vmax: float = 1.0,
    show_grid: bool = True,
    show_boundaries: bool = True,
    boundary_lw: float = 2.5,
    grid_lw: float = 0.5,
    alpha_block: float = 0.10,
):
    """
    Visualize pair_micro as an upper-triangular heatmap, with KK/KU/UU blocks emphasized.

    - Models ordered: known first, then unknown
    - Unknown labels colored red + bold
    - Upper triangle only (lower triangle + diagonal blank)
    - Block emphasis:
        KK: known-known block (top-left)
        KU: known-unknown block (top-right)
        UU: unknown-unknown block (bottom-right)
    """

    # Collect model IDs from keys
    all_models = sorted({m for pair in pair_micro.keys() for m in pair})
    known_set = set(known)

    # Order models: known first (keep given order), then unknown
    known_models = [m for m in known if m in all_models]
    unknown_models = [m for m in all_models if m not in known_set]
    models = known_models + unknown_models

    n = len(models)
    if n == 0:
        raise ValueError("No models found in pair_micro.")

    k = len(known_models)  # boundary index
    model_to_idx = {m: i for i, m in enumerate(models)}

    # Build matrix (store values in upper triangle)
    matrix = np.full((n, n), np.nan)
    for (a, b), stats in pair_micro.items():
        acc = stats.get("micro_acc", None)
        if acc is None:
            continue
        if a not in model_to_idx or b not in model_to_idx:
            continue

        i, j = model_to_idx[a], model_to_idx[b]
        if i < j:
            matrix[i, j] = acc
        else:
            matrix[j, i] = acc

    # Mask lower triangle + diagonal + NaNs
    lower_or_diag = np.tril(np.ones((n, n), dtype=bool), k=0)
    masked = np.ma.array(matrix, mask=lower_or_diag | np.isnan(matrix))

    fig, ax = plt.subplots(figsize=figsize)

    # Default colormap; masked cells are white
    cmap = plt.cm.get_cmap().copy()
    cmap.set_bad(color="white")

    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #cbar.ax.set_ylabel("micro_acc", rotation=90)

    # Axis ticks/labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticklabels(models)

    # Unknown labels red + bold
    for lab in ax.get_xticklabels():
        if lab.get_text() not in known_set:
            lab.set_color("red")
            lab.set_fontweight("bold")
    for lab in ax.get_yticklabels():
        if lab.get_text() not in known_set:
            lab.set_color("red")
            lab.set_fontweight("bold")

    # Emphasize KK/KU/UU blocks
    if k > 0 and k < n:
        # KK (known-known)
        ax.add_patch(
            plt.Rectangle(
                (-0.5, -0.5),
                k,
                k,
                fill=True,
                alpha=alpha_block,
                edgecolor="none",
            )
        )
        # KU (known-unknown)
        ax.add_patch(
            plt.Rectangle(
                (k - 0.5, -0.5),
                n - k,
                k,
                fill=True,
                alpha=alpha_block,
                edgecolor="none",
            )
        )
        # UU (unknown-unknown)
        ax.add_patch(
            plt.Rectangle(
                (k - 0.5, k - 0.5),
                n - k,
                n - k,
                fill=True,
                alpha=alpha_block,
                edgecolor="none",
            )
        )

        # Boundary lines
        if show_boundaries:
            ax.axhline(k - 0.5, linewidth=boundary_lw)
            ax.axvline(k - 0.5, linewidth=boundary_lw)
    # Grid lines
    if show_grid:
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", linewidth=grid_lw)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate cell values
    if annotate:
        for i in range(n):
            for j in range(n):
                if i < j and not np.isnan(matrix[i, j]):
                    ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center", fontsize=8)

    fig.tight_layout()
    
    return fig, ax

def plot_ranking_scatter(r1, r2, figsize=(8, 6), invert=True, ylabel="VTON-IQA Score (Half Dataset)"):

    label_color = {"known": "green", "unknown": "red", None: "black"}

    def parse_item(s: str):
        m = re.match(r"^(.*?)\((.*?)\)\s*$", s)
        if m:
            return m.group(1), m.group(2)
        return s, None

    # ==========================================================
    # Case 1: (name, score) tuple format → score scatter
    #   - marker: per-model unique (len(common) types)
    #   - color: aligned to r2 status (green/red)
    #   - add subtle y=x diagonal (dashed)
    # ==========================================================
    if isinstance(r1[0], tuple):

        score_a, status_a = {}, {}
        for name, score in r1:
            base, st = parse_item(name)
            score_a[base] = score
            status_a[base] = st

        score_b, status_b = {}, {}
        for name, score in r2:
            base, st = parse_item(name)
            score_b[base] = score
            status_b[base] = st

        common = sorted(set(score_a) & set(score_b))

        fig, ax = plt.subplots(figsize=figsize)

        # ---- marker pool (prepare len(common) types) ----
        marker_pool = [
            'o', 's', '^', 'D', 'v', 'P', 'X',
            '<', '>', '*', 'h', 'H', 'd', 'p'
        ]
        if len(common) > len(marker_pool):
            raise ValueError(
                f"Not enough marker types prepared: need {len(common)}, have {len(marker_pool)}"
            )

        legend_elements = []

        # ---- scatter ----
        for idx, base in enumerate(common):
            x = score_a[base]  # Full
            y = score_b[base]  # Half
            color = label_color.get(status_b.get(base), "black")
            marker = marker_pool[idx]

            ax.scatter(
                x, y,
                color=color,
                marker=marker,
                s=200,
                edgecolor='black',
                linewidth=1.0,
                zorder=3
            )

            # per-model legend entry (shape + color)
            legend_elements.append(
                Line2D(
                    [0], [0],
                    marker=marker,
                    color='black',
                    markerfacecolor=color,
                    markeredgecolor='black',
                    linestyle='None',
                    markersize=8,
                    label=base
                )
            )

        ax.set_xlabel("VTON-IQA Score (Full Dataset)", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)

        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

        # ---- y = x diagonal (subtle) ----
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        low = min(xmin, ymin)
        high = max(xmax, ymax)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.plot(
            [low, high],
            [low, high],
            linestyle='--',
            linewidth=1.5,
            color='0.3',
            alpha=0.7,
            zorder=1
        )

        # ---- model legend (outside right) ----
        ax.legend(
            handles=legend_elements,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=9,
            title="Models"
        )

        # ---- embedded color explanation (left-middle) ----
        ax.text(
            0.02, 0.60,
            "Green: Known Model",
            transform=ax.transAxes,
            fontsize=11,
            color="green",
            verticalalignment='center'
        )

        ax.text(
            0.02, 0.55,
            "Red: Unknown Model",
            transform=ax.transAxes,
            fontsize=11,
            color="red",
            verticalalignment='center'
        )

        fig.tight_layout()
        return fig, ax

    # ==========================================================
    # Case 2: original ranking format (string list)
    #   - marker: up/down/none
    #   - color: aligned to r2 status
    # ==========================================================
    rank_a, status_a = {}, {}
    for i, s in enumerate(r1):
        base, st = parse_item(s)
        rank_a[base] = i
        status_a[base] = st

    rank_b, status_b = {}, {}
    for i, s in enumerate(r2):
        base, st = parse_item(s)
        rank_b[base] = i
        status_b[base] = st

    common = sorted(set(rank_a) & set(rank_b), key=lambda x: rank_a[x])

    fig, ax = plt.subplots(figsize=figsize)

    for base in common:
        xi = rank_a[base]
        yi = rank_b[base]
        color = label_color.get(status_b.get(base), "black")

        if yi < xi:
            marker = r'$\mathbf{\uparrow}$'
        elif yi > xi:
            marker = r'$\mathbf{\downarrow}$'
        else:
            marker = 'o'

        ax.scatter(
            xi, yi,
            color=color,
            marker=marker,
            s=380,
            edgecolor='black',
            linewidth=1.2,
            zorder=3
        )

    ax.set_xticks(range(len(r1)))
    ax.set_yticks(range(len(r2)))

    a_by_rank = [parse_item(s)[0] for s in r1]
    b_by_rank = [parse_item(s)[0] for s in r2]

    ax.set_xticklabels(a_by_rank, rotation=45, ha="right")
    ax.set_yticklabels(b_by_rank)

    ax.set_xlabel("Rank (Full Dataset)", fontsize=13)
    ax.set_ylabel("Rank (Half Dataset)", fontsize=13)

    # x-axis labels all black
    for tick in ax.get_xticklabels():
        tick.set_color("black")

    # y-axis labels colored
    for tick in ax.get_yticklabels():
        name = tick.get_text()
        tick.set_color(label_color.get(status_b.get(name), "black"))

    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    fig.tight_layout()
    return fig, ax

def show_ordered_vton_imgs(
    person_path: str,
    garment_path: str,
    vton_paths: List[str],
    dataset_type: Literal["vton_evals_dc", "vton_evals_vitonhd"],
    key_scores: List[float],                    # 例: 人間評価(高いほど良いに統一済み想定)
    eval_scores: Dict[str, List[float]],        # metric -> scores (各vton_pathsと同じ順)
    border_threshold: float = 0.5,
    font_size: int = 48,
    unit_image_size: Tuple[int, int] = (512, 768),
    draw_score_mode: bool = True,               # key_scoresの値を描く
    draw_rank_mode: bool = True,                # metricごとの順位を描く
    show_metric_names: bool = True,             # True: "LPIPS:3"  False: "3" など
    ):
    """
    - key_scores と eval_scores は「higher is better」に統一済み（lower is betterは事前に符号反転されている）前提
    - 並び順は key_scores の高い順（GT順など）にする
    - 各画像に metric ごとの順位を埋め込む（1が最良）
    """

    def concat_horizontally(images):
        widths = [img.width for img in images]
        heights = [img.height for img in images]
        new_img = Image.new("RGB", (sum(widths), max(heights)))
        x_offset = 0
        for img in images:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width
        return new_img

    def add_red_border(img, border_width=8):
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for i in range(border_width):
            draw.rectangle([i, i, w - i - 1, h - i - 1], outline="red")
        return img

    def resolve_ref_person_path(
        garment_path: str,
        dataset_type: Literal["vitonhd_by_dc", "vitonhd", "dc", "synth_vton", "vton_evals_dc", "vton_evals_vitonhd"],
    ) -> str:
        if dataset_type in ["vitonhd_by_dc", "vitonhd", "vton_evals_vitonhd"]:
            return garment_path.replace("cloth", "image")
        elif dataset_type in ["dc", "synth_vton", "vton_evals_dc"]:
            return garment_path.replace("_1.jpg", "_0.jpg")
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    def load_font(font_size):
        candidates = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        for path in candidates:
            if os.path.exists(path):
                return ImageFont.truetype(path, font_size)
        return ImageFont.load_default()

    def draw_text_box(img, lines, font, anchor="tl", padding=12, margin=12):
        """
        lines: List[str] (複数行)
        anchor:
          - "tl": top-left
          - "tr": top-right
          - "bl": bottom-left
          - "br": bottom-right
        """
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # 行ごとのbboxを測る
        line_bboxes = [draw.textbbox((0, 0), t, font=font) for t in lines]
        line_ws = [(bb[2] - bb[0]) for bb in line_bboxes]
        line_hs = [(bb[3] - bb[1]) for bb in line_bboxes]

        text_w = max(line_ws) if line_ws else 0
        text_h = sum(line_hs) + max(0, (len(lines) - 1)) * int(font.size * 0.15)

        # 描画開始座標
        if anchor == "tl":
            x0 = margin
            y0 = margin
        elif anchor == "tr":
            x0 = w - margin - text_w
            y0 = margin
        elif anchor == "bl":
            x0 = margin
            y0 = h - margin - text_h
        elif anchor == "br":
            x0 = w - margin - text_w
            y0 = h - margin - text_h
        else:
            raise ValueError("anchor must be one of tl/tr/bl/br")

        # 背景（黒）
        bg = [x0 - padding, y0 - padding, x0 + text_w + padding, y0 + text_h + padding]
        draw.rectangle(bg, fill="black")

        # テキスト（白 + うっすらアウトライン）
        y = y0
        outline = 2
        for t, lh in zip(lines, line_hs):
            for ox in range(-outline, outline + 1):
                for oy in range(-outline, outline + 1):
                    draw.text((x0 + ox, y + oy), t, font=font, fill="black")
            draw.text((x0, y), t, font=font, fill="white")
            y += lh + int(font.size * 0.15)

        return img

    def draw_plain_text(img, text, font, anchor="tr", margin=12):
        """
        背景なし・黒文字のみで描画
        """
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        if anchor == "tr":
            x = w - margin - text_w
            y = margin
        elif anchor == "tl":
            x = margin
            y = margin
        elif anchor == "br":
            x = w - margin - text_w
            y = h - margin - text_h
        elif anchor == "bl":
            x = margin
            y = h - margin - text_h
        else:
            raise ValueError("anchor must be one of tl/tr/bl/br")

        draw.text((x, y), text, font=font, fill="black")
        return img

    def compute_ranks_desc(scores: List[float], eps: float = 1e-12) -> List[int]:
        """
        competition ranking (同値は同順位、次は飛ぶ)
        scores: higher is better
        return: ranks (1=best)
        """
        s = np.asarray(scores, dtype=np.float64)
        order = np.argsort(-s, kind="stable")  # desc
        ranks = np.empty_like(order, dtype=np.int64)

        rank = 1
        prev_val = None
        for i, idx in enumerate(order):
            val = s[idx]
            if prev_val is None:
                ranks[idx] = rank
                prev_val = val
            else:
                if np.isclose(val, prev_val, atol=eps, rtol=0.0):
                    ranks[idx] = rank
                else:
                    rank = i + 1  # competition: position-based
                    ranks[idx] = rank
                    prev_val = val
        return ranks.tolist()

    # --- load base images ---
    person = Image.open(os.path.join(VTONQBENCH_ROOT, person_path)).resize(unit_image_size, resample=Image.BILINEAR)

    if dataset_type == "vton_evals_dc":
        garment = Image.open(os.path.join(DRESSCODE_ROOT, garment_path)).resize(unit_image_size, resample=Image.BILINEAR)
        ref_person = Image.open(os.path.join(DRESSCODE_ROOT, resolve_ref_person_path(garment_path, dataset_type))).resize(unit_image_size, resample=Image.BILINEAR)
    elif dataset_type == "vton_evals_vitonhd":
        garment = Image.open(os.path.join(VITON_HD_ROOT, garment_path)).resize(unit_image_size, resample=Image.BILINEAR)
        ref_person = Image.open(os.path.join(VITON_HD_ROOT, resolve_ref_person_path(garment_path, dataset_type))).resize(unit_image_size, resample=Image.BILINEAR)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # --- sanity checks ---
    n = len(vton_paths)
    if len(key_scores) != n:
        raise ValueError(f"len(key_scores)={len(key_scores)} must match len(vton_paths)={n}")
    for m, xs in eval_scores.items():
        if len(xs) != n:
            raise ValueError(f"eval_scores['{m}'] length {len(xs)} must match len(vton_paths)={n}")

    # --- compute ranks per metric (1=best) ---
    metric_ranks: Dict[str, List[int]] = {m: compute_ranks_desc(xs) for m, xs in eval_scores.items()}

    # --- order by key_scores (higher is better) ---
    order_idx = np.argsort(np.asarray(key_scores), kind="stable")

    # --- draw ---
    font_score = load_font(font_size * 0.70)
    font_rank = load_font(max(20, int(font_size * 0.60)))
    
    vtons = []
    for r in order_idx:
        vton_path = vton_paths[r]
        key = float(key_scores[r])

        img = Image.open(os.path.join(VTONQBENCH_ROOT, vton_path)).resize(unit_image_size, resample=Image.BILINEAR)

        # key score (top-right)
        if draw_score_mode:
            img = draw_plain_text(
                img,
                f"{key:.3f}",
                font=font_score,
                anchor="tr",
                margin=12
            )

        # metric ranks (top-left)
        if draw_rank_mode and len(metric_ranks) > 0:
            lines = []
            for m in eval_scores.keys():  # dictの順序を保持（Python3.7+）
                rk = metric_ranks[m][r]
                lines.append(f"{m}:{rk}" if show_metric_names else f"{rk}")
            img = draw_text_box(
                img,
                lines,
                font=font_rank,
                anchor="tl",
                padding=10,
                margin=12
            )

        # border (based on key score threshold)
        if key >= border_threshold:
            img = add_red_border(img)

        vtons.append(img)

    cc = concat_horizontally([garment, person] + vtons + [ref_person])
    
    return cc
