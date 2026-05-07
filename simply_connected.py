import argparse
import bisect
import contextlib
import gc
import glob
import io
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
from PIL import Image
from torchvision.models import ConvNeXt_Tiny_Weights, DenseNet121_Weights, EfficientNet_B0_Weights, ResNet50_Weights, Swin_T_Weights, ViT_B_16_Weights, convnext_tiny, densenet121, efficientnet_b0, resnet50, swin_t, vit_b_16
ImageKey = Tuple[int, int]
Quad = Tuple[ImageKey, ImageKey, ImageKey, ImageKey]
VertexMap = Dict[ImageKey, torch.Tensor]
IMAGE_CHANNELS = 3
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_DIMENSION = IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH
IMAGE_RMS_DENOMINATOR = math.sqrt(float(IMAGE_DIMENSION))
IMAGENET_STANDARD_DEVIATION = (0.229, 0.224, 0.225)
ONE_GRAY_LEVEL_L2 = math.sqrt(IMAGE_HEIGHT * IMAGE_WIDTH * sum(((1.0 / (255.0 * value)) ** 2 for value in IMAGENET_STANDARD_DEVIATION)))
USE_CHANNELS_LAST = True

@dataclass
class RunPaths:
    quad_image_dir: Path
    results_dir: Path
    checkpoint_dir: Path

@dataclass
class FillConfig:
    max_deepfool_iter: int
    overshoot: float
    line_search_steps: int
    base_grid_side: int
    min_quad_side_rel: float
    stop_diameter_gray_rms: float
    sample_batch_size: int
    label_batch_size: int
    grid_point_block: int
    repair_batch_size: int
    checkpoint_every_level: bool
    keep_last_checkpoints: int
    checkpoint_interval: int
    size_metric_sample_quads: int
    size_filter_batch_size: int
    use_adaptive_grid_steps: bool
    min_grid_steps: int
    max_grid_steps: int
    max_runtime_s: Optional[float]
    use_boundary_param_cache: bool
    boundary_reference_grid: int

class Tee(io.StringIO):

    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def write(self, text):
        self.stream.write(text)
        self.stream.flush()
        return super().write(text)

    def flush(self):
        self.stream.flush()
        return super().flush()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quad-id', type=int, default=None)
    parser.add_argument('--model-name', default='resnet50')
    parser.add_argument('--repo-dir', default=None)
    parser.add_argument('--quad-image-dir', default=None)
    parser.add_argument('--results-dir', default=None)
    parser.add_argument('--checkpoint-dir', default=None)
    parser.add_argument('--expected-quads', type=int, default=1000)
    parser.add_argument('--filename-width', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deepfool-iter', type=int, default=50)
    parser.add_argument('--overshoot', type=float, default=0.02)
    parser.add_argument('--ls-steps', type=int, default=10)
    parser.add_argument('--base-grid-side', type=int, default=100)
    parser.add_argument('--min-quad-side-rel', type=float, default=1.0 / 10000.0)
    parser.add_argument('--stop-diameter-gray-rms', type=float, default=0.5)
    parser.add_argument('--sample-batch-size', type=int, default=128)
    parser.add_argument('--label-batch-size', type=int, default=256)
    parser.add_argument('--grid-point-block', type=int, default=32)
    parser.add_argument('--repair-batch-size', type=int, default=4)
    parser.add_argument('--checkpoint-every-level', action='store_true')
    parser.add_argument('--keep-last-checkpoints', type=int, default=1)
    parser.add_argument('--checkpoint-interval', type=int, default=2)
    parser.add_argument('--size-metric-sample-quads', type=int, default=256)
    parser.add_argument('--size-filter-batch-size', type=int, default=32)
    parser.add_argument('--use-adaptive-grid-steps', action='store_true', default=True)
    parser.add_argument('--no-adaptive-grid-steps', action='store_false', dest='use_adaptive_grid_steps')
    parser.add_argument('--min-grid-steps', type=int, default=2)
    parser.add_argument('--max-grid-steps', type=int, default=512)
    parser.add_argument('--use-channels-last', action='store_true', default=True)
    parser.add_argument('--no-channels-last', action='store_false', dest='use_channels_last')
    parser.add_argument('--use-tf32', action='store_true', default=True)
    parser.add_argument('--no-tf32', action='store_false', dest='use_tf32')
    parser.add_argument('--use-compile', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--max-wall-hours', type=float, default=22.75)
    parser.add_argument('--use-boundary-param-cache', action='store_true', default=True)
    parser.add_argument('--no-boundary-param-cache', action='store_false', dest='use_boundary_param_cache')
    parser.add_argument('--boundary-reference-grid', type=int, default=128)
    parser.add_argument('--cuda-wait-seconds', type=int, default=300)
    parser.add_argument('--cuda-retry-sleep', type=int, default=10)
    return parser.parse_args()

def resolve_paths(args) -> RunPaths:
    repo_dir = Path(args.repo_dir).resolve() if args.repo_dir else Path(__file__).resolve().parent
    model_name = str(args.model_name).strip().lower()
    quad_image_dir = Path(args.quad_image_dir).resolve() if args.quad_image_dir else repo_dir / 'imagenet_quads' / model_name
    results_dir = Path(args.results_dir).resolve() if args.results_dir else repo_dir / 'results' / model_name
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else repo_dir / 'checkpoints' / model_name
    return RunPaths(quad_image_dir=quad_image_dir, results_dir=results_dir, checkpoint_dir=checkpoint_dir)

def build_fill_config(args) -> FillConfig:
    max_runtime_s = None
    if args.max_wall_hours is not None and args.max_wall_hours > 0.0:
        max_runtime_s = float(args.max_wall_hours) * 3600.0
    return FillConfig(max_deepfool_iter=int(args.deepfool_iter), overshoot=float(args.overshoot), line_search_steps=int(args.ls_steps), base_grid_side=int(args.base_grid_side), min_quad_side_rel=float(args.min_quad_side_rel), stop_diameter_gray_rms=float(args.stop_diameter_gray_rms), sample_batch_size=int(args.sample_batch_size), label_batch_size=int(args.label_batch_size), grid_point_block=int(args.grid_point_block), repair_batch_size=int(args.repair_batch_size), checkpoint_every_level=bool(args.checkpoint_every_level), keep_last_checkpoints=int(args.keep_last_checkpoints), checkpoint_interval=int(args.checkpoint_interval), size_metric_sample_quads=int(args.size_metric_sample_quads), size_filter_batch_size=int(args.size_filter_batch_size), use_adaptive_grid_steps=bool(args.use_adaptive_grid_steps), min_grid_steps=int(args.min_grid_steps), max_grid_steps=int(args.max_grid_steps), max_runtime_s=max_runtime_s, use_boundary_param_cache=bool(args.use_boundary_param_cache), boundary_reference_grid=int(args.boundary_reference_grid))

def current_quad_id(args) -> int:
    if args.quad_id is not None:
        return int(args.quad_id)
    slurm_value = os.environ.get('SLURM_ARRAY_TASK_ID')
    if slurm_value is None:
        raise RuntimeError('Pass --quad-id or run as a SLURM array with SLURM_ARRAY_TASK_ID set')
    return int(slurm_value)

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def wait_for_device(cuda_wait_seconds: int, cuda_retry_sleep: int) -> torch.device:
    deadline = time.time() + float(cuda_wait_seconds)
    last_error = None
    while True:
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                device = torch.device('cuda:0')
                torch.cuda.set_device(device)
                _ = torch.empty(1, device=device)
                torch.cuda.synchronize(device)
                torch.backends.cudnn.benchmark = True
                return device
            last_error = 'torch.cuda.is_available() is False or device_count is 0'
        except Exception as exc:
            last_error = f'{type(exc).__name__}: {exc}'
        if time.time() >= deadline:
            visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
            raise RuntimeError(f'CUDA is not available after waiting {cuda_wait_seconds} seconds. CUDA_VISIBLE_DEVICES={visible_devices}. Last error: {last_error}')
        time.sleep(float(cuda_retry_sleep))

def setup_precision(device: torch.device, use_tf32: bool) -> None:
    if device.type == 'cuda' and use_tf32:
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

def cleanup_memory(device: Optional[torch.device]=None) -> None:
    gc.collect()
    if isinstance(device, torch.device) and device.type == 'cuda':
        torch.cuda.empty_cache()

def sync_model(model) -> None:
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def start_timer(model) -> float:
    sync_model(model)
    return time.perf_counter()

def end_timer(model, started_at: float) -> float:
    sync_model(model)
    return time.perf_counter() - started_at

def to_model_format(batch: torch.Tensor) -> torch.Tensor:
    if USE_CHANNELS_LAST and batch.device.type == 'cuda' and (batch.dim() == 4):
        return batch.contiguous(memory_format=torch.channels_last)
    return batch

def load_pil(path: Path) -> Image.Image:
    return Image.open(path).convert('RGB')

def load_model_and_preprocess(model_name: str, device: torch.device, use_channels_last: bool):
    key = str(model_name).strip().lower()
    model_registry = {'resnet50': (resnet50, ResNet50_Weights.DEFAULT), 'densenet121': (densenet121, DenseNet121_Weights.DEFAULT), 'efficientnet_b0': (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT), 'convnext_tiny': (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT), 'vit_b_16': (vit_b_16, ViT_B_16_Weights.DEFAULT), 'swin_t': (swin_t, Swin_T_Weights.DEFAULT)}
    if key not in model_registry:
        names = ', '.join(sorted(model_registry.keys()))
        raise RuntimeError(f'Unknown model_name={model_name}. Supported: {names}')
    constructor, weights = model_registry[key]
    preprocess = weights.transforms()
    model = constructor(weights=weights).to(device)
    if device.type == 'cuda' and use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return (model, preprocess)

def find_quad_image_path(image_dir: Path, quad_id: int, corner_name: str, filename_width: int) -> Path:
    candidates = [f'{quad_id}{corner_name}.jpeg', f'{quad_id}{corner_name}.jpg', f'{quad_id}{corner_name}.png']
    if filename_width > 0:
        candidates.extend([f'{quad_id:0{filename_width}d}{corner_name}.jpeg', f'{quad_id:0{filename_width}d}{corner_name}.jpg', f'{quad_id:0{filename_width}d}{corner_name}.png'])
    for name in candidates:
        path = image_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f'Could not find quad image for quad_id={quad_id}, corner={corner_name} in {image_dir}')

def load_quad_tensors(image_dir: Path, quad_id: int, filename_width: int, preprocess):
    image_paths = {corner: find_quad_image_path(image_dir, quad_id, corner, filename_width) for corner in ['a', 'b', 'c', 'd']}
    tensors = {}
    for corner, path in image_paths.items():
        tensors[corner] = preprocess(load_pil(path)).detach().cpu().clone()
    return (image_paths, tensors)

def tensor_to_cpu_vertex(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()

def row_l2(batch: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(batch.reshape(batch.shape[0], -1), dim=1)

def tensor_l2(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.reshape(-1)).item())

def l2_to_norm_rms(value: float) -> float:
    return float(value) / IMAGE_RMS_DENOMINATOR

def l2_to_gray_rms(value: float) -> float:
    return float(value) / ONE_GRAY_LEVEL_L2

def gray_rms_to_l2(value: float) -> float:
    return float(value) * ONE_GRAY_LEVEL_L2

def target_depth_for_l2(root_l2: float, target_l2: float) -> int:
    if root_l2 <= target_l2:
        return 0
    return int(math.ceil(math.log2(float(root_l2) / float(target_l2))))

def next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()

def previous_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << int(math.floor(math.log2(value)))

def triangle_area_l2(point1: torch.Tensor, point2: torch.Tensor, point3: torch.Tensor) -> float:
    side1 = (point2 - point1).reshape(-1).double()
    side2 = (point3 - point1).reshape(-1).double()
    dot11 = torch.dot(side1, side1)
    dot22 = torch.dot(side2, side2)
    dot12 = torch.dot(side1, side2)
    area_square = torch.clamp(dot11 * dot22 - dot12 * dot12, min=0.0)
    return float(0.5 * torch.sqrt(area_square).item())

def quad_area_l2(corner00: torch.Tensor, corner10: torch.Tensor, corner01: torch.Tensor, corner11: torch.Tensor) -> float:
    return triangle_area_l2(corner00, corner10, corner11) + triangle_area_l2(corner00, corner11, corner01)

def batched_triangle_area_l2(point1: torch.Tensor, point2: torch.Tensor, point3: torch.Tensor) -> torch.Tensor:
    side1 = (point2 - point1).reshape(point1.shape[0], -1).double()
    side2 = (point3 - point1).reshape(point1.shape[0], -1).double()
    dot11 = torch.sum(side1 * side1, dim=1)
    dot22 = torch.sum(side2 * side2, dim=1)
    dot12 = torch.sum(side1 * side2, dim=1)
    area_square = torch.clamp(dot11 * dot22 - dot12 * dot12, min=0.0)
    return 0.5 * torch.sqrt(area_square)

def batched_quad_area_sum(corner00: torch.Tensor, corner10: torch.Tensor, corner01: torch.Tensor, corner11: torch.Tensor, chunk_size: int=16) -> float:
    total = 0.0
    count = corner00.shape[0]
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        quad1 = corner00[start:end]
        quad2 = corner10[start:end]
        quad3 = corner01[start:end]
        quad4 = corner11[start:end]
        values = batched_triangle_area_l2(quad1, quad2, quad4) + batched_triangle_area_l2(quad1, quad4, quad3)
        total += float(torch.sum(values).item())
    return float(total)

def summarize_values(values: Sequence[float]):
    if len(values) == 0:
        return {'min': float('nan'), 'mean': float('nan'), 'max': float('nan'), 'sum': 0.0}
    return {'min': float(min(values)), 'mean': float(sum(values) / len(values)), 'max': float(max(values)), 'sum': float(sum(values))}

def sampled_indices(count: int, max_count: int) -> List[int]:
    if count <= max_count:
        return list(range(count))
    if max_count <= 1:
        return [0]
    step = float(count - 1) / float(max_count - 1)
    output = []
    seen = set()
    for index in range(max_count):
        sample_index = int(round(index * step))
        if sample_index not in seen:
            seen.add(sample_index)
            output.append(sample_index)
    return output

def quad_size(corner00: torch.Tensor, corner10: torch.Tensor, corner01: torch.Tensor, corner11: torch.Tensor):
    edge_bottom = tensor_l2(corner00 - corner10)
    edge_left = tensor_l2(corner00 - corner01)
    edge_right = tensor_l2(corner10 - corner11)
    edge_top = tensor_l2(corner01 - corner11)
    diag_main = tensor_l2(corner00 - corner11)
    diag_cross = tensor_l2(corner10 - corner01)
    edges = [edge_bottom, edge_left, edge_right, edge_top]
    diagonals = [diag_main, diag_cross]
    all_distances = edges + diagonals
    max_edge = max(edges)
    mean_edge = sum(edges) / len(edges)
    diameter = max(all_distances)
    area = quad_area_l2(corner00, corner10, corner01, corner11)
    return {'min_edge_l2': float(min(edges)), 'mean_edge_l2': float(mean_edge), 'max_edge_l2': float(max_edge), 'mean_diag_l2': float(sum(diagonals) / len(diagonals)), 'max_diag_l2': float(max(diagonals)), 'min_all_l2': float(min(all_distances)), 'mean_all_l2': float(sum(all_distances) / len(all_distances)), 'diameter_l2': float(diameter), 'diameter_norm_rms': l2_to_norm_rms(diameter), 'diameter_gray_rms': l2_to_gray_rms(diameter), 'max_edge_norm_rms': l2_to_norm_rms(max_edge), 'max_edge_gray_rms': l2_to_gray_rms(max_edge), 'mean_edge_norm_rms': l2_to_norm_rms(mean_edge), 'mean_edge_gray_rms': l2_to_gray_rms(mean_edge), 'area_l2_sq': float(area)}

def quad_size_summary(vertices: VertexMap, quads: Sequence[Quad], max_quads: int):
    chosen = sampled_indices(len(quads), max_quads)
    rows = []
    for index in chosen:
        key00, key10, key01, key11 = quads[index]
        rows.append(quad_size(vertices[key00], vertices[key10], vertices[key01], vertices[key11]))
    diameter = summarize_values([row['diameter_l2'] for row in rows])
    diameter_gray = summarize_values([row['diameter_gray_rms'] for row in rows])
    max_edge = summarize_values([row['max_edge_l2'] for row in rows])
    max_edge_gray = summarize_values([row['max_edge_gray_rms'] for row in rows])
    mean_edge = summarize_values([row['mean_edge_l2'] for row in rows])
    mean_edge_gray = summarize_values([row['mean_edge_gray_rms'] for row in rows])
    area = summarize_values([row['area_l2_sq'] for row in rows])
    area_total = float(area['sum'])
    if len(quads) > len(rows) and len(rows) > 0:
        area_total = float(area['mean'] * len(quads))
    return {'frontier': int(len(quads)), 'sample_n': int(len(rows)), 'mode': 'exact' if len(quads) <= max_quads else 'sampled', 'diameter_l2_min': diameter['min'], 'diameter_l2_mean': diameter['mean'], 'diameter_l2_max': diameter['max'], 'diameter_gray_min': diameter_gray['min'], 'diameter_gray_mean': diameter_gray['mean'], 'diameter_gray_max': diameter_gray['max'], 'max_edge_l2_min': max_edge['min'], 'max_edge_l2_mean': max_edge['mean'], 'max_edge_l2_max': max_edge['max'], 'max_edge_gray_min': max_edge_gray['min'], 'max_edge_gray_mean': max_edge_gray['mean'], 'max_edge_gray_max': max_edge_gray['max'], 'mean_edge_l2_min': mean_edge['min'], 'mean_edge_l2_mean': mean_edge['mean'], 'mean_edge_l2_max': mean_edge['max'], 'mean_edge_gray_min': mean_edge_gray['min'], 'mean_edge_gray_mean': mean_edge_gray['mean'], 'mean_edge_gray_max': mean_edge_gray['max'], 'area_l2_sq_min': area['min'], 'area_l2_sq_mean': area['mean'], 'area_l2_sq_max': area['max'], 'area_l2_sq_total_est': area_total}

def exact_quad_area_sum(vertices: VertexMap, quads: Sequence[Quad], batch_size: int=256) -> float:
    total = 0.0
    for start in range(0, len(quads), batch_size):
        chunk = quads[start:min(start + batch_size, len(quads))]
        for quad in chunk:
            total += quad_area_l2(vertices[quad[0]], vertices[quad[1]], vertices[quad[2]], vertices[quad[3]])
    return float(total)

def split_by_diameter(vertices: VertexMap, quads: Sequence[Quad], threshold_gray_rms: float, batch_size: int):
    if threshold_gray_rms is None or threshold_gray_rms <= 0.0:
        return ([], list(quads), {'checked': 0, 'accepted': 0, 'remaining': len(quads), 'min_gray': float('nan'), 'mean_gray': float('nan'), 'max_gray': float('nan'), 'accepted_area_l2_sq': 0.0})
    threshold_l2 = gray_rms_to_l2(threshold_gray_rms)
    accepted = []
    remaining = []
    diameter_gray_values = []
    with torch.no_grad():
        for start in range(0, len(quads), batch_size):
            chunk = quads[start:min(start + batch_size, len(quads))]
            quad1 = torch.stack([vertices[quad[0]] for quad in chunk], dim=0).contiguous()
            quad2 = torch.stack([vertices[quad[1]] for quad in chunk], dim=0).contiguous()
            quad3 = torch.stack([vertices[quad[2]] for quad in chunk], dim=0).contiguous()
            quad4 = torch.stack([vertices[quad[3]] for quad in chunk], dim=0).contiguous()
            dist1 = row_l2(quad1 - quad2)
            dist2 = row_l2(quad1 - quad3)
            dist3 = row_l2(quad2 - quad4)
            dist4 = row_l2(quad3 - quad4)
            dist5 = row_l2(quad1 - quad4)
            dist6 = row_l2(quad2 - quad3)
            diameter = torch.maximum(torch.maximum(torch.maximum(dist1, dist2), torch.maximum(dist3, dist4)), torch.maximum(dist5, dist6))
            keep_mask = diameter <= threshold_l2
            diameter_gray_values.extend((float(value) for value in (diameter / ONE_GRAY_LEVEL_L2).detach().cpu().tolist()))
            for quad, keep in zip(chunk, keep_mask.detach().cpu().tolist()):
                if keep:
                    accepted.append(quad)
                else:
                    remaining.append(quad)
    summary = summarize_values(diameter_gray_values)
    accepted_area = exact_quad_area_sum(vertices, accepted, batch_size=batch_size) if len(accepted) > 0 else 0.0
    return (accepted, remaining, {'checked': len(quads), 'accepted': len(accepted), 'remaining': len(remaining), 'min_gray': summary['min'], 'mean_gray': summary['mean'], 'max_gray': summary['max'], 'accepted_area_l2_sq': float(accepted_area)})

@torch.inference_mode()
def predict_labels_batch(model, batch: torch.Tensor) -> torch.Tensor:
    logits = model(to_model_format(batch))
    return torch.argmax(logits, dim=1)

@torch.inference_mode()
def predict_label(model, image: torch.Tensor) -> int:
    logits = model(to_model_format(image.unsqueeze(0))).squeeze(0)
    return int(torch.argmax(logits).item())

def new_repair_stats():
    return {'missing': 0, 'label_batches': 0, 'label_s': 0.0, 'repair_attempts': 0, 'repair_s': 0.0, 'repair_gpu_sum_s': 0.0, 'repair_iters_sum': 0, 'repair_iters_max': 0, 'repair_hit_max_iter': 0, 'repair_amp_used': 0, 'repair_amp_steps_sum': 0, 'repair_val_tol_steps': 0, 'repair_zero_grad_fallbacks': 0, 'repair_random_fallbacks': 0, 'repair_iter_le_5': 0, 'repair_iter_6_10': 0, 'repair_iter_11_25': 0, 'repair_iter_26_50': 0, 'repair_iter_51_100': 0, 'repair_iter_gt_100': 0}

def merge_repair_stats(total, partial):
    for key in total:
        if key not in partial:
            continue
        if key == 'repair_iters_max':
            total[key] = max(int(total[key]), int(partial[key]))
        elif key == 'missing':
            total[key] = int(total[key])
        elif isinstance(total[key], float):
            total[key] += float(partial[key])
        else:
            total[key] += int(partial[key])

def add_single_repair(total, stats):
    count = int(stats.get('deepfool_iters', 0))
    total['repair_iters_sum'] += count
    total['repair_iters_max'] = max(int(total['repair_iters_max']), count)
    if bool(stats.get('deepfool_hit_max_iter', False)):
        total['repair_hit_max_iter'] += 1
    if bool(stats.get('used_amp', False)):
        total['repair_amp_used'] += 1
    total['repair_amp_steps_sum'] += int(stats.get('amp_steps_used', 0))
    total['repair_val_tol_steps'] += int(stats.get('deepfool_val_tol_steps', 0))
    total['repair_zero_grad_fallbacks'] += int(stats.get('deepfool_zero_grad_fallbacks', 0))
    total['repair_random_fallbacks'] += int(stats.get('deepfool_random_fallbacks', 0))
    if count <= 5:
        total['repair_iter_le_5'] += 1
    elif count <= 10:
        total['repair_iter_6_10'] += 1
    elif count <= 25:
        total['repair_iter_11_25'] += 1
    elif count <= 50:
        total['repair_iter_26_50'] += 1
    elif count <= 100:
        total['repair_iter_51_100'] += 1
    else:
        total['repair_iter_gt_100'] += 1

def targeted_deepfool_l2_batch(model, batch: torch.Tensor, target_label: int, max_iter: int=50, overshoot: float=0.02, step_min: float=0.001, eps: float=1e-12, val_tol: float=1e-06):
    original = batch.detach()
    current = original.clone().detach()
    total_step = torch.zeros_like(original)
    batch_size = original.shape[0]
    device = original.device
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    hit_target = torch.zeros(batch_size, dtype=torch.bool, device=device)
    iteration_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    last_label = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    val_tol_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
    zero_grad_fallbacks = torch.zeros(batch_size, dtype=torch.long, device=device)
    random_fallbacks = torch.zeros(batch_size, dtype=torch.long, device=device)
    for iteration in range(1, max_iter + 1):
        active = ~done
        if not bool(active.any().item()):
            break
        current_req = current.detach().requires_grad_(True)
        logits = model(to_model_format(current_req))
        labels = torch.argmax(logits, dim=1)
        active_indices = active.nonzero(as_tuple=False).flatten()
        iteration_counts[active_indices] = int(iteration)
        last_label[active_indices] = labels[active_indices]
        hit_now = active & labels.eq(int(target_label))
        if bool(hit_now.any().item()):
            done[hit_now] = True
            hit_target[hit_now] = True
        needs_step = active & ~labels.eq(int(target_label))
        if not bool(needs_step.any().item()):
            current = current_req.detach()
            continue
        indices = needs_step.nonzero(as_tuple=False).flatten()
        current_labels = labels[indices]
        margin = logits[indices, current_labels] - logits[indices, int(target_label)]
        step_all = torch.zeros_like(original)
        handled = torch.zeros(indices.numel(), dtype=torch.bool, device=device)
        near_boundary = torch.abs(margin.detach()) < val_tol
        if bool(near_boundary.any().item()):
            near_positions = near_boundary.nonzero(as_tuple=False).flatten()
            near_indices = indices[near_positions]
            val_tol_steps[near_indices] += 1
            target_grad = torch.autograd.grad(logits[near_indices, int(target_label)].sum(), current_req, retain_graph=True, create_graph=False)[0].detach()
            direction = target_grad[near_indices]
            norm = row_l2(direction)
            can_step = norm > 0.0
            if bool(can_step.any().item()):
                step = (step_min / (norm[can_step] + eps)).view(-1, 1, 1, 1) * direction[can_step]
                step_all[near_indices[can_step]] = step
                handled[near_positions[can_step]] = True
        normal_positions = (~handled).nonzero(as_tuple=False).flatten()
        if normal_positions.numel() > 0:
            normal_indices = indices[normal_positions]
            normal_margin = margin[normal_positions]
            grad = torch.autograd.grad(normal_margin.sum(), current_req, retain_graph=False, create_graph=False)[0].detach()
            direction = grad[normal_indices].clone()
            norm = row_l2(direction)
            bad = ~(norm > 0.0)
            if bool(bad.any().item()):
                bad_indices = normal_indices[bad]
                zero_grad_fallbacks[bad_indices] += 1
                tmp = current.detach().clone().requires_grad_(True)
                tmp_logits = model(to_model_format(tmp))
                target_grad = torch.autograd.grad(tmp_logits[bad_indices, int(target_label)].sum(), tmp, retain_graph=False, create_graph=False)[0].detach()
                fallback_direction = target_grad[bad_indices]
                fallback_norm = row_l2(fallback_direction)
                can_use = fallback_norm > 0.0
                bad_positions = bad.nonzero(as_tuple=False).flatten()
                if bool(can_use.any().item()):
                    direction[bad_positions[can_use]] = fallback_direction[can_use]
                    norm[bad_positions[can_use]] = fallback_norm[can_use]
                still_bad = ~(norm > 0.0)
                if bool(still_bad.any().item()):
                    random_indices = normal_indices[still_bad]
                    random_fallbacks[random_indices] += 1
                    direction[still_bad] = torch.randn_like(direction[still_bad])
                    norm[still_bad] = row_l2(direction[still_bad])
            step = (-normal_margin.detach() / (norm * norm + eps)).view(-1, 1, 1, 1) * direction
            step_norm = row_l2(step)
            too_small = step_norm < step_min
            nonzero_small = too_small & (step_norm > 0.0)
            zero_small = too_small & ~(step_norm > 0.0)
            if bool(nonzero_small.any().item()):
                step[nonzero_small] = (step_min / (step_norm[nonzero_small] + eps)).view(-1, 1, 1, 1) * step[nonzero_small]
            if bool(zero_small.any().item()):
                step[zero_small] = (step_min / (norm[zero_small] + eps)).view(-1, 1, 1, 1) * direction[zero_small]
            step_all[normal_indices] = step
        total_step = total_step + step_all
        current = (original + (1.0 + overshoot) * total_step).detach()
    hit_max_iter = ~hit_target & (iteration_counts >= int(max_iter))
    stats = {'iters': iteration_counts.detach().cpu(), 'hit_target': hit_target.detach().cpu(), 'hit_max_iter': hit_max_iter.detach().cpu(), 'last_label': last_label.detach().cpu(), 'val_tol_steps': val_tol_steps.detach().cpu(), 'zero_grad_fallbacks': zero_grad_fallbacks.detach().cpu(), 'random_fallbacks': random_fallbacks.detach().cpu()}
    return (current.detach() - original.detach(), stats)

@torch.inference_mode()
def bisection_to_label(model, start_batch: torch.Tensor, target_batch: torch.Tensor, target_label: int, active: torch.Tensor, steps: int=10) -> torch.Tensor:
    output = target_batch.clone()
    if not bool(active.any().item()):
        return output
    indices = active.nonzero(as_tuple=False).flatten()
    start = start_batch[indices]
    target = target_batch[indices]
    low = torch.zeros(indices.numel(), device=start_batch.device)
    high = torch.ones(indices.numel(), device=start_batch.device)
    for _ in range(steps):
        mid = 0.5 * (low + high)
        mixed = (1.0 - mid).view(-1, 1, 1, 1) * start + mid.view(-1, 1, 1, 1) * target
        labels = predict_labels_batch(model, mixed)
        good = labels.eq(int(target_label))
        high = torch.where(good, mid, high)
        low = torch.where(good, low, mid)
    output[indices] = (1.0 - high).view(-1, 1, 1, 1) * start + high.view(-1, 1, 1, 1) * target
    return output

def repair_batch_to_label(model, batch: torch.Tensor, target_label: int, max_iter: int, overshoot: float, line_search_steps: int, amp_steps: int=12):
    stats = new_repair_stats()
    batch = batch.detach()
    device = batch.device
    initial_labels = predict_labels_batch(model, batch)
    ok = initial_labels.eq(int(target_label))
    output = batch.clone()
    todo = ~ok
    if not bool(todo.any().item()):
        return (output, ok.detach().cpu().tolist(), stats)
    todo_indices = todo.nonzero(as_tuple=False).flatten()
    work_batch = batch[todo_indices]
    repair_step, deepfool_stats = targeted_deepfool_l2_batch(model, work_batch, target_label, max_iter=max_iter, overshoot=overshoot)
    count = work_batch.shape[0]
    stats['repair_attempts'] += int(count)
    target_batch = work_batch + repair_step
    target_labels = predict_labels_batch(model, target_batch)
    hit = target_labels.eq(int(target_label))
    used_amp = torch.zeros(count, dtype=torch.bool, device=device)
    amp_count = torch.zeros(count, dtype=torch.long, device=device)
    alpha = torch.ones(count, device=device)
    needs_amp = ~hit
    for _ in range(amp_steps):
        if not bool(needs_amp.any().item()):
            break
        active_indices = needs_amp.nonzero(as_tuple=False).flatten()
        used_amp[active_indices] = True
        amp_count[active_indices] += 1
        alpha[active_indices] *= 2.0
        trial = work_batch[active_indices] + alpha[active_indices].view(-1, 1, 1, 1) * repair_step[active_indices]
        trial_labels = predict_labels_batch(model, trial)
        got = trial_labels.eq(int(target_label))
        if bool(got.any().item()):
            got_indices = active_indices[got]
            target_batch[got_indices] = trial[got]
            hit[got_indices] = True
            needs_amp[got_indices] = False
    minimal_batch = bisection_to_label(model, work_batch, target_batch, target_label, active=hit, steps=line_search_steps)
    final_labels = predict_labels_batch(model, minimal_batch)
    final_ok = hit & final_labels.eq(int(target_label))
    if bool(final_ok.any().item()):
        output[todo_indices[final_ok]] = minimal_batch[final_ok]
        ok[todo_indices[final_ok]] = True
    for index in range(count):
        add_single_repair(stats, {'deepfool_iters': int(deepfool_stats['iters'][index].item()), 'deepfool_hit_target': bool(deepfool_stats['hit_target'][index].item()), 'deepfool_hit_max_iter': bool(deepfool_stats['hit_max_iter'][index].item()), 'deepfool_val_tol_steps': int(deepfool_stats['val_tol_steps'][index].item()), 'deepfool_zero_grad_fallbacks': int(deepfool_stats['zero_grad_fallbacks'][index].item()), 'deepfool_random_fallbacks': int(deepfool_stats['random_fallbacks'][index].item()), 'used_amp': bool(used_amp[index].item()), 'amp_steps_used': int(amp_count[index].item())})
    return (output, ok.detach().cpu().tolist(), stats)

def repair_vertex_keys(model, vertices: VertexMap, verified_vertices: set, keys: Sequence[ImageKey], target_label: int, config: FillConfig, stop_on_failure: bool=False):
    stats = new_repair_stats()
    if len(keys) == 0:
        return (True, 0, stats)
    device = next(model.parameters()).device
    started_at = time.perf_counter()
    success = True
    for start in range(0, len(keys), config.repair_batch_size):
        batch_keys = list(keys[start:min(start + config.repair_batch_size, len(keys))])
        cpu_batch = torch.stack([vertices[key] for key in batch_keys], dim=0).contiguous()
        gpu_batch = cpu_batch.to(device, non_blocking=False)
        repair_started = start_timer(model)
        repaired_batch, ok_list, batch_stats = repair_batch_to_label(model, gpu_batch, target_label, max_iter=config.max_deepfool_iter, overshoot=config.overshoot, line_search_steps=config.line_search_steps)
        batch_stats['repair_gpu_sum_s'] += float(end_timer(model, repair_started))
        merge_repair_stats(stats, batch_stats)
        repaired_cpu = repaired_batch.detach().cpu()
        for index, key in enumerate(batch_keys):
            if ok_list[index]:
                vertices[key] = repaired_cpu[index].clone()
                verified_vertices.add(key)
            else:
                success = False
        cleanup_memory(device)
        if not success and stop_on_failure:
            break
    stats['repair_s'] = time.perf_counter() - started_at
    return (success, int(stats['repair_attempts']), stats)

def repair_wrong_vertices(model, vertices: VertexMap, verified_vertices: set, wrong_keys: Sequence[ImageKey], target_label: int, config: FillConfig):
    stats = new_repair_stats()
    stats['missing'] = len(wrong_keys)
    if len(wrong_keys) == 0:
        return (True, 0, stats)
    ok, attempts, local_stats = repair_vertex_keys(model, vertices, verified_vertices, wrong_keys, target_label, config, stop_on_failure=True)
    local_stats['missing'] = len(wrong_keys)
    return (ok, attempts, local_stats)

def ordered_unique(keys: Iterable[ImageKey]) -> List[ImageKey]:
    seen = set()
    output = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            output.append(key)
    return output

def ordered_quad_keys(quads: Sequence[Quad]) -> List[ImageKey]:
    return ordered_unique((key for quad in quads for key in quad))

def ensure_vertices_verified(model, vertices: VertexMap, verified_vertices: set, keys: Sequence[ImageKey], target_label: int, config: FillConfig):
    unique_keys = ordered_unique(list(keys))
    missing = [key for key in unique_keys if key not in verified_vertices]
    stats = new_repair_stats()
    stats['missing'] = len(missing)
    if len(missing) == 0:
        return (True, 0, stats)
    device = next(model.parameters()).device
    wrong_keys = []
    for start in range(0, len(missing), config.label_batch_size):
        batch_keys = missing[start:min(start + config.label_batch_size, len(missing))]
        label_started = start_timer(model)
        cpu_batch = torch.stack([vertices[key] for key in batch_keys], dim=0).contiguous()
        gpu_batch = cpu_batch.to(device, non_blocking=False)
        labels = predict_labels_batch(model, gpu_batch).detach().cpu().tolist()
        stats['label_s'] += end_timer(model, label_started)
        stats['label_batches'] += 1
        for key, label in zip(batch_keys, labels):
            if int(label) == int(target_label):
                verified_vertices.add(key)
            else:
                wrong_keys.append(key)
    original_missing = int(stats['missing'])
    ok, attempts, repair_stats = repair_wrong_vertices(model, vertices, verified_vertices, wrong_keys, target_label, config)
    merge_repair_stats(stats, repair_stats)
    stats['missing'] = original_missing
    return (ok, attempts, stats)

def coordinate_depth(min_quad_side_rel: float) -> int:
    return int(math.ceil(math.log2(1.0 / min_quad_side_rel)))

def midpoint_key(key1: ImageKey, key2: ImageKey) -> ImageKey:
    return ((key1[0] + key2[0]) // 2, (key1[1] + key2[1]) // 2)

def is_boundary_key(key: ImageKey, coord_denom: int) -> bool:
    return key[0] == 0 or key[0] == coord_denom or key[1] == 0 or (key[1] == coord_denom)

def boundary_edge_name(key: ImageKey, coord_denom: int) -> str:
    index1, index2 = key
    if index2 == 0:
        return 'a_to_b'
    if index1 == coord_denom:
        return 'b_to_d'
    if index2 == coord_denom:
        return 'c_to_d'
    if index1 == 0:
        return 'a_to_c'
    return 'interior'

def boundary_point_from_parameter(key: ImageKey, coord_denom: int, corner00: torch.Tensor, corner10: torch.Tensor, corner01: torch.Tensor, corner11: torch.Tensor) -> torch.Tensor:
    index1, index2 = key
    if index2 == 0:
        value = float(index1) / float(coord_denom)
        return (1.0 - value) * corner00 + value * corner10
    if index1 == coord_denom:
        value = float(index2) / float(coord_denom)
        return (1.0 - value) * corner10 + value * corner11
    if index2 == coord_denom:
        value = float(index1) / float(coord_denom)
        return (1.0 - value) * corner01 + value * corner11
    if index1 == 0:
        value = float(index2) / float(coord_denom)
        return (1.0 - value) * corner00 + value * corner01
    raise RuntimeError(f'Boundary point requested for interior key {key}')

def create_midpoint(vertices: VertexMap, new_key: ImageKey, key1: ImageKey, key2: ImageKey, coord_denom: int, corner00: torch.Tensor, corner10: torch.Tensor, corner01: torch.Tensor, corner11: torch.Tensor, counts, use_boundary_param_cache: bool) -> bool:
    if new_key in vertices:
        return False
    if use_boundary_param_cache and is_boundary_key(new_key, coord_denom):
        vertices[new_key] = boundary_point_from_parameter(new_key, coord_denom, corner00, corner10, corner01, corner11)
        edge_name = boundary_edge_name(new_key, coord_denom)
        counts[edge_name] = counts.get(edge_name, 0) + 1
    else:
        vertices[new_key] = 0.5 * (vertices[key1] + vertices[key2])
    return True

def create_center(vertices: VertexMap, center_key: ImageKey, key00: ImageKey, key10: ImageKey, key01: ImageKey, key11: ImageKey) -> bool:
    if center_key in vertices:
        return False
    vertices[center_key] = 0.25 * (vertices[key00] + vertices[key10] + vertices[key01] + vertices[key11])
    return True

def subdivision_keys(quad: Quad):
    key00, key10, key01, key11 = quad
    key_bottom = midpoint_key(key00, key10)
    key_left = midpoint_key(key00, key01)
    key_right = midpoint_key(key10, key11)
    key_top = midpoint_key(key01, key11)
    key_center = midpoint_key(key00, key11)
    children = [(key00, key_bottom, key_left, key_center), (key_bottom, key10, key_center, key_right), (key_left, key_center, key01, key_top), (key_center, key_right, key_top, key11)]
    return ([key_bottom, key_left, key_right, key_top, key_center], children)

def subdivide_failing_quads(model, vertices: VertexMap, verified_vertices: set, failing_quads: Sequence[Quad], target_label: int, config: FillConfig, coord_denom: int, corner00: torch.Tensor, corner10: torch.Tensor, corner01: torch.Tensor, corner11: torch.Tensor):
    stats = new_repair_stats()
    created_count = 0
    boundary_counts = {'a_to_b': 0, 'b_to_d': 0, 'c_to_d': 0, 'a_to_c': 0}
    next_frontier = []
    candidate_keys = []
    for quad in failing_quads:
        key00, key10, key01, key11 = quad
        new_keys, children = subdivision_keys(quad)
        key_bottom, key_left, key_right, key_top, key_center = new_keys
        if create_midpoint(vertices, key_bottom, key00, key10, coord_denom, corner00, corner10, corner01, corner11, boundary_counts, config.use_boundary_param_cache):
            created_count += 1
        if create_midpoint(vertices, key_left, key00, key01, coord_denom, corner00, corner10, corner01, corner11, boundary_counts, config.use_boundary_param_cache):
            created_count += 1
        if create_midpoint(vertices, key_right, key10, key11, coord_denom, corner00, corner10, corner01, corner11, boundary_counts, config.use_boundary_param_cache):
            created_count += 1
        if create_midpoint(vertices, key_top, key01, key11, coord_denom, corner00, corner10, corner01, corner11, boundary_counts, config.use_boundary_param_cache):
            created_count += 1
        if create_center(vertices, key_center, key00, key10, key01, key11):
            created_count += 1
        candidate_keys.extend(new_keys)
        next_frontier.extend(children)
    candidate_keys = ordered_unique(candidate_keys)
    ok, attempts, check_stats = ensure_vertices_verified(model, vertices, verified_vertices, candidate_keys, target_label, config)
    stats.update(check_stats)
    stats['failed_quads'] = len(failing_quads)
    stats['created_vertices'] = created_count
    stats['candidate_vertices'] = len(candidate_keys)
    stats['boundary_created_a_to_b'] = int(boundary_counts['a_to_b'])
    stats['boundary_created_b_to_d'] = int(boundary_counts['b_to_d'])
    stats['boundary_created_c_to_d'] = int(boundary_counts['c_to_d'])
    stats['boundary_created_a_to_c'] = int(boundary_counts['a_to_c'])
    stats['boundary_created_total'] = int(sum(boundary_counts.values()))
    if not ok:
        return (None, False, attempts, stats)
    return (next_frontier, True, attempts, stats)

def bilinear_points(corner00: torch.Tensor, corner10: torch.Tensor, corner01: torch.Tensor, corner11: torch.Tensor, grid_u: torch.Tensor, grid_v: torch.Tensor) -> torch.Tensor:
    weight00 = (1.0 - grid_u) * (1.0 - grid_v)
    weight10 = grid_u * (1.0 - grid_v)
    weight01 = (1.0 - grid_u) * grid_v
    weight11 = grid_u * grid_v
    points = corner00.unsqueeze(1) * weight00.view(1, -1, 1, 1, 1) + corner10.unsqueeze(1) * weight10.view(1, -1, 1, 1, 1) + corner01.unsqueeze(1) * weight01.view(1, -1, 1, 1, 1) + corner11.unsqueeze(1) * weight11.view(1, -1, 1, 1, 1)
    return points.reshape(-1, corner00.shape[1], corner00.shape[2], corner00.shape[3])

def check_quads_on_grid(model, vertices: VertexMap, frontier: Sequence[Quad], target_label: int, grid_steps: int, sample_batch_size: int, grid_point_block: int):
    grid_steps = max(1, int(grid_steps))
    side = grid_steps + 1
    total_points = side * side
    info = {'grid_steps': int(grid_steps), 'side': int(side), 'total_points_per_quad': int(total_points), 'frontier': int(len(frontier)), 'point_blocks': 0, 'model_batches': 0, 'samples': 0, 'model_s': 0.0, 'cpu_s': 0.0, 'stack_s': 0.0, 'failed_early': 0}
    if len(frontier) == 0:
        return ([], [], 0, None, info)
    active = [True] * len(frontier)
    first_fail = None
    checked_samples = 0
    device = next(model.parameters()).device
    dtype = next(iter(vertices.values())).dtype
    for point_start in range(0, total_points, grid_point_block):
        cpu_started = time.perf_counter()
        active_indices = [index for index, is_active in enumerate(active) if is_active]
        info['cpu_s'] += time.perf_counter() - cpu_started
        if len(active_indices) == 0:
            break
        point_end = min(point_start + grid_point_block, total_points)
        point_ids = torch.arange(point_start, point_end, device=device)
        grid_i = point_ids % side
        grid_j = torch.div(point_ids, side, rounding_mode='floor')
        grid_u = grid_i.to(dtype=dtype) / float(grid_steps)
        grid_v = grid_j.to(dtype=dtype) / float(grid_steps)
        point_count = int(point_end - point_start)
        quad_chunk_size = max(1, sample_batch_size // max(1, point_count))
        info['point_blocks'] += 1
        for offset in range(0, len(active_indices), quad_chunk_size):
            quad_indices = active_indices[offset:min(offset + quad_chunk_size, len(active_indices))]
            stack_started = time.perf_counter()
            corner00 = torch.stack([vertices[frontier[index][0]] for index in quad_indices], dim=0).to(device, non_blocking=False)
            corner10 = torch.stack([vertices[frontier[index][1]] for index in quad_indices], dim=0).to(device, non_blocking=False)
            corner01 = torch.stack([vertices[frontier[index][2]] for index in quad_indices], dim=0).to(device, non_blocking=False)
            corner11 = torch.stack([vertices[frontier[index][3]] for index in quad_indices], dim=0).to(device, non_blocking=False)
            info['stack_s'] += time.perf_counter() - stack_started
            model_started = start_timer(model)
            samples = bilinear_points(corner00, corner10, corner01, corner11, grid_u, grid_v)
            labels = predict_labels_batch(model, samples).view(len(quad_indices), point_count)
            labels_cpu = labels.detach().cpu()
            info['model_s'] += end_timer(model, model_started)
            info['model_batches'] += 1
            info['samples'] += int(labels.numel())
            checked_samples += int(labels.numel())
            cpu_started = time.perf_counter()
            for row_index, quad_index in enumerate(quad_indices):
                row = labels_cpu[row_index]
                bad = torch.nonzero(row != target_label, as_tuple=False)
                if bad.numel() > 0:
                    active[quad_index] = False
                    info['failed_early'] += 1
                    if first_fail is None:
                        local_point = int(bad[0, 0].item())
                        flat_index = int(point_start + local_point)
                        fail_i = flat_index % side
                        fail_j = flat_index // side
                        first_fail = {'quad_frontier_index': int(quad_index), 'flat_index': int(flat_index), 'u_index': int(fail_i), 'v_index': int(fail_j), 'u': float(fail_i / grid_steps), 'v': float(fail_j / grid_steps), 'label': int(row[local_point].item())}
            info['cpu_s'] += time.perf_counter() - cpu_started
    passing = [frontier[index] for index, is_active in enumerate(active) if is_active]
    failing = [frontier[index] for index, is_active in enumerate(active) if not is_active]
    return (passing, failing, checked_samples, first_fail, info)

def get_boundary_curve(vertices: VertexMap, coord_denom: int, edge: str):
    if edge == 'bottom':
        keys = sorted([key for key in vertices if key[1] == 0], key=lambda key: key[0])
        params = [float(key[0]) / float(coord_denom) for key in keys]
    elif edge == 'right':
        keys = sorted([key for key in vertices if key[0] == coord_denom], key=lambda key: key[1])
        params = [float(key[1]) / float(coord_denom) for key in keys]
    elif edge == 'top':
        keys = sorted([key for key in vertices if key[1] == coord_denom], key=lambda key: key[0])
        params = [float(key[0]) / float(coord_denom) for key in keys]
    elif edge == 'left':
        keys = sorted([key for key in vertices if key[0] == 0], key=lambda key: key[1])
        params = [float(key[1]) / float(coord_denom) for key in keys]
    else:
        raise RuntimeError(f'Unknown boundary edge {edge}')
    if len(keys) < 2:
        raise RuntimeError(f'Boundary edge {edge} has fewer than two vertices')
    return (params, [vertices[key] for key in keys])

def interpolate_curve(params: Sequence[float], values: Sequence[torch.Tensor], value: float) -> torch.Tensor:
    if value <= params[0]:
        return values[0]
    if value >= params[-1]:
        return values[-1]
    index = bisect.bisect_right(params, value) - 1
    index = max(0, min(index, len(params) - 2))
    value0 = params[index]
    value1 = params[index + 1]
    if value1 <= value0:
        return values[index]
    alpha = (float(value) - float(value0)) / (float(value1) - float(value0))
    return (1.0 - alpha) * values[index] + alpha * values[index + 1]

def sample_curve(params: Sequence[float], values: Sequence[torch.Tensor], grid_steps: int) -> torch.Tensor:
    samples = [interpolate_curve(params, values, float(index) / float(grid_steps)) for index in range(grid_steps + 1)]
    return torch.stack(samples, dim=0).contiguous()

def coons_row(bottom, top, left_value, right_value, corner00, corner10, corner01, corner11, u_grid, v_value):
    grid_u = u_grid.view(-1, 1, 1, 1)
    grid_v = float(v_value)
    corner_patch = (1.0 - grid_u) * (1.0 - grid_v) * corner00.unsqueeze(0) + grid_u * (1.0 - grid_v) * corner10.unsqueeze(0) + (1.0 - grid_u) * grid_v * corner01.unsqueeze(0) + grid_u * grid_v * corner11.unsqueeze(0)
    row = (1.0 - grid_v) * bottom + grid_v * top + (1.0 - grid_u) * left_value.unsqueeze(0) + grid_u * right_value.unsqueeze(0) - corner_patch
    return row.contiguous()

def coons_reference_area(vertices: VertexMap, coord_denom: int, corner00: torch.Tensor, corner10: torch.Tensor, corner01: torch.Tensor, corner11: torch.Tensor, grid_steps: int=128, chunk_size: int=16) -> float:
    grid_steps = int(grid_steps)
    if grid_steps <= 0:
        return float('nan')
    bottom_params, bottom_values = get_boundary_curve(vertices, coord_denom, 'bottom')
    right_params, right_values = get_boundary_curve(vertices, coord_denom, 'right')
    top_params, top_values = get_boundary_curve(vertices, coord_denom, 'top')
    left_params, left_values = get_boundary_curve(vertices, coord_denom, 'left')
    bottom = sample_curve(bottom_params, bottom_values, grid_steps)
    top = sample_curve(top_params, top_values, grid_steps)
    u_grid = torch.linspace(0.0, 1.0, grid_steps + 1, dtype=bottom.dtype)
    previous_row = coons_row(bottom, top, interpolate_curve(left_params, left_values, 0.0), interpolate_curve(right_params, right_values, 0.0), corner00, corner10, corner01, corner11, u_grid, 0.0)
    total = 0.0
    for row_index in range(grid_steps):
        v_value = float(row_index + 1) / float(grid_steps)
        current_row = coons_row(bottom, top, interpolate_curve(left_params, left_values, v_value), interpolate_curve(right_params, right_values, v_value), corner00, corner10, corner01, corner11, u_grid, v_value)
        total += batched_quad_area_sum(previous_row[:-1], previous_row[1:], current_row[:-1], current_row[1:], chunk_size=chunk_size)
        previous_row = current_row
    return float(total)

def choose_grid_steps(depth: int, base_grid_side: int, stop_diameter_gray_rms: float, size_info, use_adaptive_grid_steps: bool, min_grid_steps: int, max_grid_steps: int):
    base_steps = max(int(min_grid_steps), int(math.ceil(float(base_grid_side) / float(2 ** depth))))
    adaptive_steps = int(min_grid_steps)
    if use_adaptive_grid_steps and stop_diameter_gray_rms is not None and (stop_diameter_gray_rms > 0.0):
        adaptive_steps = int(math.ceil(float(size_info['diameter_gray_max']) / float(stop_diameter_gray_rms)))
        adaptive_steps = max(int(min_grid_steps), adaptive_steps)
    raw_steps = max(base_steps, adaptive_steps)
    grid_steps = next_power_of_two(raw_steps)
    capped = False
    if max_grid_steps is not None and max_grid_steps > 0 and (grid_steps > int(max_grid_steps)):
        grid_steps = previous_power_of_two(int(max_grid_steps))
        capped = True
    grid_steps = max(int(min_grid_steps), int(grid_steps))
    return (int(grid_steps), int(base_steps), int(adaptive_steps), bool(capped))

def new_run_stats(target_label: Optional[int], config: FillConfig, coord_depth: int, coord_denom: int):
    return {'y': target_label, 'initial_quad_ok': None, 'levels_processed': 0, 'quads_checked': 0, 'quads_ok': 0, 'quads_failed': 0, 'quads_created': 0, 'quads_size_accepted': 0, 'repair_attempts': 0, 'repair_failed': 0, 'leaf_quads': 0, 'resolution_leaf_quads': 0, 'grid_points_checked': 0, 'time_total_s': 0.0, 'time_vertex_s': 0.0, 'time_grid_s': 0.0, 'time_subdivide_s': 0.0, 'time_repair_s': 0.0, 'time_repair_gpu_sum_s': 0.0, 'time_grid_model_s': 0.0, 'time_grid_cpu_s': 0.0, 'time_grid_stack_s': 0.0, 'time_size_filter_s': 0.0, 'repair_iters_sum': 0, 'repair_iters_max': 0, 'repair_hit_max_iter': 0, 'repair_amp_used': 0, 'reason': None, 'corner_labels': None, 'first_fail': None, 'failed_depth': None, 'failed_quad_side_rel': None, 'min_quad_side_rel': float(config.min_quad_side_rel), 'stop_diameter_gray_rms': float(config.stop_diameter_gray_rms), 'effective_quad_side_rel': float(1.0 / coord_denom), 'effective_quad_area_rel': float((1.0 / coord_denom) ** 2), 'coord_depth': int(coord_depth), 'coord_denom': int(coord_denom), 'max_depth_observed': 0, 'vertex_count': 4, 'vertex_ok_count': 4, 'frontier_remaining': 0, 'root_diameter_l2': None, 'root_max_edge_l2': None, 'root_mean_edge_l2': None, 'root_diameter_gray_rms': None, 'original_straight_quad_area_l2_sq': None, 'constructed_area_l2_sq': 0.0, 'boundary_reference_area_l2_sq': None, 'boundary_reference_grid': int(config.boundary_reference_grid), 'boundary_reference_s': 0.0, 'use_boundary_param_cache': bool(config.use_boundary_param_cache), 'boundary_created_total': 0, 'boundary_created_a_to_b': 0, 'boundary_created_b_to_d': 0, 'boundary_created_c_to_d': 0, 'boundary_created_a_to_c': 0, 'level_records': []}

def add_level_record(stats, record) -> None:
    stats['level_records'].append(record)

def update_repair_totals(stats, repair_stats, attempts: int=0) -> None:
    stats['repair_attempts'] += int(attempts)
    stats['time_repair_s'] += float(repair_stats.get('repair_s', 0.0))
    stats['time_repair_gpu_sum_s'] += float(repair_stats.get('repair_gpu_sum_s', 0.0))
    stats['repair_iters_sum'] += int(repair_stats.get('repair_iters_sum', 0))
    stats['repair_iters_max'] = max(int(stats['repair_iters_max']), int(repair_stats.get('repair_iters_max', 0)))
    stats['repair_hit_max_iter'] += int(repair_stats.get('repair_hit_max_iter', 0))
    stats['repair_amp_used'] += int(repair_stats.get('repair_amp_used', 0))

def update_boundary_totals(stats, subdivision_stats) -> None:
    for key in ['boundary_created_total', 'boundary_created_a_to_b', 'boundary_created_b_to_d', 'boundary_created_c_to_d', 'boundary_created_a_to_c']:
        stats[key] += int(subdivision_stats.get(key, 0))

def print_area_table(stats) -> None:
    records = stats.get('level_records', [])
    if len(records) == 0:
        return
    print()
    print('Area coverage table:')
    print('depth | frontier | size_pass | grid_pass | res_pass | fail | next | accepted_area_% | cumulative_area_% | remaining_area_% | geom_area_l2_sq | geom_area_ratio')
    cumulative = 0.0
    geometric_cumulative = 0.0
    root_area = float(stats.get('original_straight_quad_area_l2_sq') or 0.0)
    for record in records:
        depth = int(record['depth'])
        denom = float(4 ** depth)
        accepted_here = int(record.get('size_pass', 0)) + int(record.get('grid_pass', 0)) + int(record.get('resolution_pass', 0))
        next_count = int(record.get('next', 0))
        accepted_area = accepted_here / denom
        cumulative += accepted_area
        remaining_area = next_count / float(4 ** (depth + 1)) if next_count > 0 else 0.0
        geom_here = float(record.get('geom_area_l2_sq', 0.0))
        geometric_cumulative += geom_here
        geom_ratio = geometric_cumulative / root_area if root_area > 0.0 else float('nan')
        print(f"{depth:5d} | {int(record.get('frontier', 0)):8d} | {int(record.get('size_pass', 0)):9d} | {int(record.get('grid_pass', 0)):9d} | {int(record.get('resolution_pass', 0)):8d} | {int(record.get('fail', 0)):4d} | {next_count:4d} | {100.0 * accepted_area:15.6f} | {100.0 * cumulative:17.6f} | {100.0 * remaining_area:16.6f} | {geom_here:15.6g} | {geom_ratio:15.6g}")
    boundary_area = float(stats.get('boundary_reference_area_l2_sq') or 0.0)
    print()
    print(f'Estimated constructed surface area L2^2: {geometric_cumulative:.6g}')
    if root_area > 0.0:
        print(f'Estimated constructed/original straight-corner area ratio: {geometric_cumulative / root_area:.6g}x')
    if boundary_area > 0.0:
        print(f'Boundary-matched Coons reference area L2^2: {boundary_area:.6g}')
        print(f'Estimated constructed/boundary-reference area ratio: {geometric_cumulative / boundary_area:.6g}x')
    print()

def save_checkpoint(checkpoint_dir: Path, checkpoint_prefix: str, depth: int, vertices: VertexMap, verified_vertices: set, accepted: Sequence[Tuple[Quad, int]], frontier: Sequence[Quad], stats, keep_last: int):
    if checkpoint_dir is None:
        return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f'{checkpoint_prefix}_depth_{depth:02d}.pt'
    tmp_path = Path(str(path) + '.tmp')
    payload = {'depth': int(depth), 'vertices': vertices, 'vertex_ok': list(verified_vertices), 'accepted': accepted, 'frontier': frontier, 'stats': stats}
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    files = sorted(glob.glob(str(checkpoint_dir / f'{checkpoint_prefix}_depth_*.pt')))
    if keep_last is not None and keep_last > 0 and (len(files) > keep_last):
        for old_path in files[:-keep_last]:
            try:
                os.remove(old_path)
            except OSError:
                pass

def maybe_save_checkpoint(paths: RunPaths, checkpoint_prefix: str, depth: int, vertices: VertexMap, verified_vertices: set, accepted, frontier, stats, config: FillConfig):
    if config.checkpoint_every_level and depth % config.checkpoint_interval == 0:
        stats['vertex_count'] = len(vertices)
        stats['vertex_ok_count'] = len(verified_vertices)
        stats['frontier_remaining'] = len(frontier)
        save_checkpoint(paths.checkpoint_dir, checkpoint_prefix, depth, vertices, verified_vertices, accepted, frontier, stats, config.keep_last_checkpoints)

def fill_quad_surface(model, corner00: torch.Tensor, corner10: torch.Tensor, corner01: torch.Tensor, corner11: torch.Tensor, config: FillConfig, paths: RunPaths, checkpoint_prefix: str, target_label: Optional[int]=None):
    coord_depth = coordinate_depth(config.min_quad_side_rel)
    coord_denom = 2 ** coord_depth
    stats = new_run_stats(target_label, config, coord_depth, coord_denom)
    sync_model(model)
    total_started = time.perf_counter()
    device = next(model.parameters()).device
    corner00_cpu = tensor_to_cpu_vertex(corner00)
    corner10_cpu = tensor_to_cpu_vertex(corner10)
    corner01_cpu = tensor_to_cpu_vertex(corner01)
    corner11_cpu = tensor_to_cpu_vertex(corner11)
    root_size = quad_size(corner00_cpu, corner10_cpu, corner01_cpu, corner11_cpu)
    stats['root_diameter_l2'] = float(root_size['diameter_l2'])
    stats['root_max_edge_l2'] = float(root_size['max_edge_l2'])
    stats['root_mean_edge_l2'] = float(root_size['mean_edge_l2'])
    stats['root_diameter_gray_rms'] = float(root_size['diameter_gray_rms'])
    stats['original_straight_quad_area_l2_sq'] = float(root_size['area_l2_sq'])
    gpu_corners = [corner00_cpu.to(device), corner10_cpu.to(device), corner01_cpu.to(device), corner11_cpu.to(device)]
    if target_label is None:
        target_label = predict_label(model, gpu_corners[0])
    stats['y'] = int(target_label)
    corner_labels = tuple((predict_label(model, corner) for corner in gpu_corners))
    stop_depth_diam = target_depth_for_l2(root_size['diameter_l2'], gray_rms_to_l2(config.stop_diameter_gray_rms))
    stop_depth_edge = target_depth_for_l2(root_size['max_edge_l2'], gray_rms_to_l2(config.stop_diameter_gray_rms))
    stop_depth_one_gray = target_depth_for_l2(root_size['diameter_l2'], ONE_GRAY_LEVEL_L2)
    stop_depth_half_gray = target_depth_for_l2(root_size['diameter_l2'], 0.5 * ONE_GRAY_LEVEL_L2)
    if any((label != target_label for label in corner_labels)):
        stats['reason'] = 'corners_not_target_label'
        stats['corner_labels'] = tuple((int(label) for label in corner_labels))
        sync_model(model)
        stats['time_total_s'] = time.perf_counter() - total_started
        return (None, None, stats)
    key00 = (0, 0)
    key10 = (coord_denom, 0)
    key01 = (0, coord_denom)
    key11 = (coord_denom, coord_denom)
    vertices = {key00: corner00_cpu, key10: corner10_cpu, key01: corner01_cpu, key11: corner11_cpu}
    verified_vertices = {key00, key10, key01, key11}
    accepted = []
    frontier = [(key00, key10, key01, key11)]
    for depth in range(coord_depth + 1):
        if config.max_runtime_s is not None and time.perf_counter() - total_started > config.max_runtime_s:
            stats['reason'] = 'python_wall_time_limit_reached'
            break
        level_started = time.perf_counter()
        if len(frontier) == 0:
            stats['reason'] = 'success'
            break
        stats['max_depth_observed'] = max(int(stats['max_depth_observed']), int(depth))
        stats['frontier_remaining'] = len(frontier)
        frontier_before = len(frontier)
        accepted_before = len(accepted)
        vertices_before = len(vertices)
        size_started = time.perf_counter()
        size_info = quad_size_summary(vertices, frontier, config.size_metric_sample_quads)
        size_s = time.perf_counter() - size_started
        keys_started = time.perf_counter()
        frontier_keys = ordered_quad_keys(frontier)
        keys_s = time.perf_counter() - keys_started
        vertex_started = start_timer(model)
        ok_vertices, vertex_attempts, vertex_stats = ensure_vertices_verified(model, vertices, verified_vertices, frontier_keys, target_label, config)
        vertex_s = end_timer(model, vertex_started)
        stats['time_vertex_s'] += vertex_s
        update_repair_totals(stats, vertex_stats, vertex_attempts)
        if not ok_vertices:
            stats['repair_failed'] += 1
            stats['reason'] = 'frontier_vertex_repair_failed'
            stats['failed_depth'] = int(depth)
            stats['vertex_count'] = len(vertices)
            stats['vertex_ok_count'] = len(verified_vertices)
            sync_model(model)
            stats['time_total_s'] = time.perf_counter() - total_started
            return (None, vertices, stats)
        if depth >= coord_depth:
            accepted.extend(((quad, depth) for quad in frontier))
            stats['leaf_quads'] += len(frontier)
            stats['resolution_leaf_quads'] += len(frontier)
            geom_area = exact_quad_area_sum(vertices, frontier, batch_size=config.size_filter_batch_size)
            stats['constructed_area_l2_sq'] += float(geom_area)
            add_level_record(stats, {'depth': depth, 'frontier': frontier_before, 'size_pass': 0, 'grid_pass': 0, 'resolution_pass': len(frontier), 'fail': 0, 'next': 0, 'geom_area_l2_sq': float(geom_area)})
            frontier = []
            stats['reason'] = 'success'
            level_s = time.perf_counter() - level_started
            maybe_save_checkpoint(paths, checkpoint_prefix, depth, vertices, verified_vertices, accepted, frontier, stats, config)
            break
        size_filter_started = time.perf_counter()
        size_accepted, grid_frontier, size_filter_info = split_by_diameter(vertices, frontier, config.stop_diameter_gray_rms, config.size_filter_batch_size)
        size_filter_s = time.perf_counter() - size_filter_started
        stats['time_size_filter_s'] += float(size_filter_s)
        if len(size_accepted) > 0:
            accepted.extend(((quad, depth) for quad in size_accepted))
            stats['leaf_quads'] += len(size_accepted)
            stats['quads_size_accepted'] += len(size_accepted)
            stats['constructed_area_l2_sq'] += float(size_filter_info['accepted_area_l2_sq'])
        if len(grid_frontier) == 0:
            if stats['initial_quad_ok'] is None:
                stats['initial_quad_ok'] = depth == 0 and len(size_accepted) == frontier_before
            frontier = []
            stats['reason'] = 'success'
            add_level_record(stats, {'depth': depth, 'frontier': frontier_before, 'size_pass': len(size_accepted), 'grid_pass': 0, 'resolution_pass': 0, 'fail': 0, 'next': 0, 'geom_area_l2_sq': float(size_filter_info['accepted_area_l2_sq'])})
            level_s = time.perf_counter() - level_started
            maybe_save_checkpoint(paths, checkpoint_prefix, depth, vertices, verified_vertices, accepted, frontier, stats, config)
            break
        grid_size_info = quad_size_summary(vertices, grid_frontier, config.size_metric_sample_quads)
        grid_steps, base_steps, adaptive_steps, grid_capped = choose_grid_steps(depth, config.base_grid_side, config.stop_diameter_gray_rms, grid_size_info, config.use_adaptive_grid_steps, config.min_grid_steps, config.max_grid_steps)
        grid_started = start_timer(model)
        passing, failing, checked, first_fail, grid_info = check_quads_on_grid(model, vertices, grid_frontier, target_label, grid_steps, config.sample_batch_size, config.grid_point_block)
        grid_s = end_timer(model, grid_started)
        stats['time_grid_s'] += grid_s
        stats['time_grid_model_s'] += float(grid_info['model_s'])
        stats['time_grid_cpu_s'] += float(grid_info['cpu_s'])
        stats['time_grid_stack_s'] += float(grid_info['stack_s'])
        stats['quads_checked'] += len(grid_frontier)
        stats['grid_points_checked'] += int(checked)
        if stats['initial_quad_ok'] is None:
            stats['initial_quad_ok'] = depth == 0 and len(failing) == 0 and (len(size_accepted) + len(passing) == frontier_before)
        stats['quads_ok'] += len(passing)
        stats['quads_failed'] += len(failing)
        accepted.extend(((quad, depth) for quad in passing))
        stats['leaf_quads'] += len(passing)
        grid_pass_area = exact_quad_area_sum(vertices, passing, batch_size=config.size_filter_batch_size) if len(passing) > 0 else 0.0
        stats['constructed_area_l2_sq'] += float(grid_pass_area)
        if first_fail is not None and stats['first_fail'] is None:
            stats['first_fail'] = first_fail
        if len(failing) == 0:
            frontier = []
            stats['reason'] = 'success'
            add_level_record(stats, {'depth': depth, 'frontier': frontier_before, 'size_pass': len(size_accepted), 'grid_pass': len(passing), 'resolution_pass': 0, 'fail': 0, 'next': 0, 'geom_area_l2_sq': float(size_filter_info['accepted_area_l2_sq'] + grid_pass_area)})
            level_s = time.perf_counter() - level_started
            maybe_save_checkpoint(paths, checkpoint_prefix, depth, vertices, verified_vertices, accepted, frontier, stats, config)
            break
        subdivision_started = start_timer(model)
        next_frontier, repair_ok, repair_attempts, subdivision_stats = subdivide_failing_quads(model, vertices, verified_vertices, failing, target_label, config, coord_denom, corner00_cpu, corner10_cpu, corner01_cpu, corner11_cpu)
        subdivision_s = end_timer(model, subdivision_started)
        stats['time_subdivide_s'] += subdivision_s
        update_repair_totals(stats, subdivision_stats, repair_attempts)
        update_boundary_totals(stats, subdivision_stats)
        if not repair_ok:
            stats['repair_failed'] += 1
            stats['reason'] = 'subdivision_vertex_repair_failed'
            stats['failed_depth'] = int(depth)
            stats['failed_quad_side_rel'] = float(1.0 / float(2 ** depth))
            stats['vertex_count'] = len(vertices)
            stats['vertex_ok_count'] = len(verified_vertices)
            sync_model(model)
            stats['time_total_s'] = time.perf_counter() - total_started
            return (None, vertices, stats)
        stats['quads_created'] += 4 * len(failing)
        stats['levels_processed'] += 1
        frontier = next_frontier
        add_level_record(stats, {'depth': depth, 'frontier': frontier_before, 'size_pass': len(size_accepted), 'grid_pass': len(passing), 'resolution_pass': 0, 'fail': len(failing), 'next': len(frontier), 'geom_area_l2_sq': float(size_filter_info['accepted_area_l2_sq'] + grid_pass_area)})
        level_s = time.perf_counter() - level_started
        repair_attempt_count = max(1, int(subdivision_stats['repair_attempts']))
        repair_wall_avg = float(subdivision_stats['repair_s']) / repair_attempt_count
        repair_gpu_avg = float(subdivision_stats['repair_gpu_sum_s']) / repair_attempt_count
        repair_avg_iter = float(subdivision_stats['repair_iters_sum']) / repair_attempt_count
        amp_avg_steps = float(subdivision_stats['repair_amp_steps_sum']) / max(1, int(subdivision_stats['repair_amp_used']))
        cleanup_memory(device)
        maybe_save_checkpoint(paths, checkpoint_prefix, depth, vertices, verified_vertices, accepted, frontier, stats, config)
    if len(frontier) > 0 and stats['reason'] is None:
        stats['reason'] = 'unprocessed_frontier_after_coord_depth'
    stats['vertex_count'] = len(vertices)
    stats['vertex_ok_count'] = len(verified_vertices)
    stats['frontier_remaining'] = len(frontier)
    if stats['reason'] == 'success' and int(config.boundary_reference_grid) > 0:
        reference_started = time.perf_counter()
        try:
            reference_area = coons_reference_area(vertices, coord_denom, corner00_cpu, corner10_cpu, corner01_cpu, corner11_cpu, grid_steps=int(config.boundary_reference_grid), chunk_size=16)
            stats['boundary_reference_area_l2_sq'] = float(reference_area)
            stats['boundary_reference_s'] = time.perf_counter() - reference_started
            constructed_area = float(stats.get('constructed_area_l2_sq') or 0.0)
            ratio = constructed_area / reference_area if reference_area > 0.0 else float('nan')
        except Exception as exc:
            stats['boundary_reference_area_l2_sq'] = None
            stats['boundary_reference_s'] = time.perf_counter() - reference_started
    sync_model(model)
    stats['time_total_s'] = time.perf_counter() - total_started
    avg_iter = float(stats['repair_iters_sum']) / max(1, int(stats['repair_attempts']))
    print_area_table(stats)
    if stats['reason'] == 'success':
        return (accepted, vertices, stats)
    return (None, vertices, stats)

def write_final_result(results_dir: Path, quad_id: int, text: str) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    final_path = results_dir / f'quad{quad_id}.txt'
    tmp_path = results_dir / f'quad{quad_id}.tmp'
    with tmp_path.open('w') as handle:
        handle.write(text)
    os.replace(tmp_path, final_path)
    return final_path

def print_run_header(args, paths: RunPaths, quad_id: int, config: FillConfig) -> None:
    print(f'Quad ID: {quad_id}', flush=True)
    print(f'Model name: {args.model_name}', flush=True)
    print(f'Quad image dir: {paths.quad_image_dir}', flush=True)
    print(f'Results dir: {paths.results_dir}', flush=True)
    print(f'Checkpoint dir: {paths.checkpoint_dir}', flush=True)
    print(f'Expected quads: {args.expected_quads}', flush=True)
    print(f'Stop diameter gray RMS: {config.stop_diameter_gray_rms:.8g}', flush=True)
    print(f'DeepFool iter: {config.max_deepfool_iter}', flush=True)
    print(f'Line search steps: {config.line_search_steps}', flush=True)
    print(f'Sample batch size: {config.sample_batch_size}', flush=True)
    print(f'Label batch size: {config.label_batch_size}', flush=True)
    print(f'Grid point block: {config.grid_point_block}', flush=True)
    print(f'Repair batch size: {config.repair_batch_size}', flush=True)
    print(f'Boundary reference grid: {config.boundary_reference_grid}', flush=True)
    print(f'One gray-level normalized L2: {float(ONE_GRAY_LEVEL_L2):.6g}', flush=True)
    print()

def print_final_summary(quad_id: int, stats, ok: bool) -> None:
    print(f"Quad {quad_id}: label={stats['y']}", flush=True)
    print(f"Root quad accepted without subdivision: {('OK' if stats['initial_quad_ok'] else 'FAILED')}", flush=True)
    print(f"Levels processed: {int(stats['levels_processed'])}", flush=True)
    print(f"Quads checked by grid: {int(stats['quads_checked'])}", flush=True)
    print(f"Quads accepted by size threshold: {int(stats['quads_size_accepted'])}", flush=True)
    print(f"Leaf quads: {int(stats['leaf_quads'])}", flush=True)
    print(f"Resolution leaf quads: {int(stats['resolution_leaf_quads'])}", flush=True)
    print(f"Quads created: {int(stats['quads_created'])}", flush=True)
    print(f"Grid samples evaluated: {int(stats['grid_points_checked'])}", flush=True)
    print(f"Repair attempts: {int(stats['repair_attempts'])}", flush=True)
    print(f"Repair failed: {int(stats['repair_failed'])}", flush=True)
    print(f"Repair average DeepFool iterations: {float(stats['repair_iters_sum']) / max(1, int(stats['repair_attempts'])):.2f}", flush=True)
    print(f"Repair max DeepFool iterations: {int(stats['repair_iters_max'])}", flush=True)
    print(f"Repair hit max_iter count: {int(stats['repair_hit_max_iter'])}", flush=True)
    print(f"Repair amplification used count: {int(stats['repair_amp_used'])}", flush=True)
    print(f"Vertices in shared registry: {int(stats['vertex_count'])}", flush=True)
    print(f"Verified vertices: {int(stats['vertex_ok_count'])}", flush=True)
    print(f"Max depth observed: {int(stats['max_depth_observed'])}", flush=True)
    print(f"Coordinate depth: {int(stats['coord_depth'])}", flush=True)
    print(f"Requested side threshold: {float(stats['min_quad_side_rel']):.8g}", flush=True)
    print(f"Stop diameter gray RMS: {float(stats['stop_diameter_gray_rms']):.8g}", flush=True)
    print(f"Effective side threshold: {float(stats['effective_quad_side_rel']):.8g}", flush=True)
    print(f"Effective area threshold: {float(stats['effective_quad_area_rel']):.8g}", flush=True)
    print(f"Root diameter L2: {float(stats['root_diameter_l2']):.6g}", flush=True)
    print(f"Root diameter gray RMS: {float(stats['root_diameter_gray_rms']):.6g}", flush=True)
    print(f"Root max edge L2: {float(stats['root_max_edge_l2']):.6g}", flush=True)
    print(f"Root mean edge L2: {float(stats['root_mean_edge_l2']):.6g}", flush=True)
    print(f"Original straight-edge quad area L2^2: {float(stats['original_straight_quad_area_l2_sq']):.6g}", flush=True)
    print(f"Estimated constructed surface area L2^2: {float(stats['constructed_area_l2_sq']):.6g}", flush=True)
    if float(stats['original_straight_quad_area_l2_sq']) > 0.0:
        print(f"Estimated constructed/original straight-corner area ratio: {float(stats['constructed_area_l2_sq']) / float(stats['original_straight_quad_area_l2_sq']):.6g}x", flush=True)
    if stats.get('boundary_reference_area_l2_sq') is not None:
        print(f"Boundary-matched Coons reference grid: {int(stats['boundary_reference_grid'])}", flush=True)
        print(f"Boundary-matched Coons reference area L2^2: {float(stats['boundary_reference_area_l2_sq']):.6g}", flush=True)
        if float(stats['boundary_reference_area_l2_sq']) > 0.0:
            print(f"Estimated constructed/boundary-reference area ratio: {float(stats['constructed_area_l2_sq']) / float(stats['boundary_reference_area_l2_sq']):.6g}x", flush=True)
        print(f"Boundary reference area seconds: {float(stats['boundary_reference_s']):.3f}", flush=True)
    print(f'One gray-level normalized L2: {float(ONE_GRAY_LEVEL_L2):.6g}', flush=True)
    print(f"Elapsed seconds: {float(stats['time_total_s']):.3f}", flush=True)
    print(f"Timing grid seconds: {float(stats['time_grid_s']):.3f}", flush=True)
    print(f"Timing subdivide seconds: {float(stats['time_subdivide_s']):.3f}", flush=True)
    print(f"Timing size filter seconds: {float(stats['time_size_filter_s']):.3f}", flush=True)
    print(f"Timing repair wall seconds: {float(stats['time_repair_s']):.3f}", flush=True)
    if ok:
        print(f"Final result: SUCCESS. Found a shared-vertex level-wise label-preserving filling of the loop a-b-d-c-a for label={int(stats['y'])}.", flush=True)
    else:
        reason = stats['reason'] if stats['reason'] is not None else 'unknown'
        print(f'Final result: FAILURE. Reason: {reason}.', flush=True)
        if stats.get('failed_depth') is not None:
            print(f"Failed depth: {int(stats['failed_depth'])}", flush=True)
        if stats.get('failed_quad_side_rel') is not None:
            print(f"Failed quad relative side: {float(stats['failed_quad_side_rel']):.8g}", flush=True)
        if stats['first_fail'] is not None:
            first_fail = stats['first_fail']
            print(f"First sampled failure: u={float(first_fail['u']):.6g}, v={float(first_fail['v']):.6g}, label={int(first_fail['label'])}", flush=True)
    print()
    print('COMPLETE', flush=True)

def main() -> None:
    global USE_CHANNELS_LAST
    args = parse_args()
    quad_id = current_quad_id(args)
    paths = resolve_paths(args)
    config = build_fill_config(args)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    final_path = paths.results_dir / f'quad{quad_id}.txt'
    if final_path.exists() and (not args.overwrite):
        print(f'Skipping quad {quad_id}: result already exists at {final_path}', flush=True)
        return
    if quad_id < 1 or quad_id > int(args.expected_quads):
        raise RuntimeError(f'quad_id={quad_id} is outside expected range 1..{args.expected_quads}')
    USE_CHANNELS_LAST = bool(args.use_channels_last)
    set_seed(int(args.seed) + quad_id)
    tee = Tee(sys.stdout)
    had_exception = False
    with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
        print_run_header(args, paths, quad_id, config)
        try:
            device = wait_for_device(int(args.cuda_wait_seconds), int(args.cuda_retry_sleep))
            setup_precision(device, bool(args.use_tf32))
            model, preprocess = load_model_and_preprocess(args.model_name, device, USE_CHANNELS_LAST)
            if args.use_compile:
                model = torch.compile(model, mode='reduce-overhead')
            _image_paths, tensors = load_quad_tensors(paths.quad_image_dir, quad_id, int(args.filename_width), preprocess)
            accepted, vertices, stats = fill_quad_surface(model, tensors['a'], tensors['b'], tensors['c'], tensors['d'], config=config, paths=paths, checkpoint_prefix=f'quad_{quad_id:04d}', target_label=None)
            ok = accepted is not None
            print_final_summary(quad_id, stats, ok)
        except Exception:
            had_exception = True
            print()
            print('EXCEPTION', flush=True)
            traceback.print_exc()
            print()
            print('COMPLETE_WITH_EXCEPTION', flush=True)
    text = tee.getvalue()
    if had_exception:
        print(f'Not saving final result file because quad {quad_id} ended with an exception.', flush=True)
        raise SystemExit(1)
    saved_path = write_final_result(paths.results_dir, quad_id, text)
    print(f'Saved final result file: {saved_path}', flush=True)
if __name__ == '__main__':
    main()
