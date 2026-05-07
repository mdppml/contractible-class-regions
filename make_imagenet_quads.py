import argparse
import csv
import io
import json
import os
import random
import shutil
import time
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image, ImageOps
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    Swin_T_Weights,
    ViT_B_16_Weights,
    convnext_tiny,
    densenet121,
    efficientnet_b0,
    resnet50,
    swin_t,
    vit_b_16,
)

MODELS = {
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
    "densenet121": (densenet121, DenseNet121_Weights.DEFAULT),
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.DEFAULT),
    "swin_t": (swin_t, Swin_T_Weights.DEFAULT),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS.keys()))
    parser.add_argument("--dataset-name", default="ILSVRC/imagenet-1k")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-labels", type=int, default=1000)
    parser.add_argument("--images-per-label", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--token", default=None)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--filename-width", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=100)
    parser.add_argument("--jpeg-subsampling", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--verify-batch-size", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-channels-last", action="store_true")
    parser.add_argument("--use-tf32", action="store_true", default=True)
    parser.add_argument("--no-tf32", action="store_false", dest="use_tf32")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    return parser.parse_args()


def output_folder(args):
    root = Path(args.output_dir).resolve() if args.output_dir else Path.cwd() / "imagenet_quads"
    return root / args.model


def get_token(args):
    value = args.token
    if value is None or len(value.strip()) == 0:
        value = os.environ.get(args.token_env)
    if value is None or len(value.strip()) == 0:
        return None
    return value.strip()


def image_name(quad_id, corner, width):
    if width > 0:
        return f"{quad_id:0{width}d}{corner}.jpeg"
    return f"{quad_id}{corner}.jpeg"


def read_image(value):
    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return ImageOps.exif_transpose(Image.open(io.BytesIO(value["bytes"]))).convert("RGB")
        if value.get("path") is not None:
            return ImageOps.exif_transpose(Image.open(value["path"])).convert("RGB")
    raise RuntimeError(f"Unsupported image object: {type(value)}")


def read_image_path(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def pick_device(device_name):
    if device_name is not None and len(device_name.strip()) > 0:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_precision(device, use_tf32):
    if device.type == "cuda" and use_tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True


def load_model(model_name, device, use_channels_last):
    build_model, weights = MODELS[model_name]
    model = build_model(weights=weights).to(device)
    if device.type == "cuda" and use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model, weights.transforms()


def model_input(batch, use_channels_last):
    if use_channels_last and batch.device.type == "cuda" and batch.dim() == 4:
        return batch.contiguous(memory_format=torch.channels_last)
    return batch


@torch.inference_mode()
def predict_batch(model, batch, use_channels_last):
    logits = model(model_input(batch, use_channels_last))
    return torch.argmax(logits, dim=1)


def predict_images(model, preprocess, device, use_channels_last, images):
    tensors = [preprocess(image).detach().cpu() for image in images]
    batch = torch.stack(tensors, dim=0).to(device, non_blocking=False)
    labels = predict_batch(model, batch, use_channels_last).detach().cpu().tolist()
    return [int(label) for label in labels]


def predict_files(model, preprocess, device, use_channels_last, paths, batch_size):
    labels = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        images = [read_image_path(path) for path in batch_paths]
        labels.extend(predict_images(model, preprocess, device, use_channels_last, images))
    return labels


def save_image(path, image, quality, subsampling):
    image.save(path, format="JPEG", quality=int(quality), subsampling=int(subsampling), optimize=False)


def load_imagenet_stream(args, token):
    return load_dataset(args.dataset_name, split=args.split, streaming=True, token=token)


def is_complete(out_dir, args):
    run_file = out_dir / "run.json"
    quad_file = out_dir / "quads.json"
    if not run_file.exists() or not quad_file.exists():
        return False
    try:
        with run_file.open("r") as handle:
            info = json.load(handle)
        if info.get("model") != args.model:
            return False
        if int(info.get("final_unique_labels", -1)) != int(args.num_labels):
            return False
        if int(info.get("final_quads", -1)) != int(args.num_labels):
            return False
        if not bool(info.get("final_disk_verified", False)):
            return False
        for quad_id in range(1, int(args.num_labels) + 1):
            for corner in ["a", "b", "c", "d"]:
                if not (out_dir / image_name(quad_id, corner, args.filename_width)).exists():
                    return False
        return True
    except Exception:
        return False


def make_output_dir(out_dir, args):
    if out_dir.exists():
        if is_complete(out_dir, args) and not args.overwrite:
            print(f"Output already complete: {out_dir}", flush=True)
            return False
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise RuntimeError(f"Output directory exists but is incomplete. Use --overwrite to rebuild: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    return True


def make_staging_dir(out_dir):
    staging = out_dir / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def collect_images(args, token, model, preprocess, device, staging):
    wanted_labels = set(range(int(args.num_labels)))
    found = {label: [] for label in wanted_labels}
    stream = load_imagenet_stream(args, token)

    images = []
    indices = []
    rows_seen = 0
    predicted = 0
    checked_after_save = 0
    kept_after_save = 0
    start_time = time.perf_counter()

    def labels_done():
        return sum(1 for label in wanted_labels if len(found[label]) >= int(args.images_per_label))

    def all_done():
        return labels_done() == int(args.num_labels)

    def flush_batch():
        nonlocal images, indices, predicted, checked_after_save, kept_after_save

        if len(images) == 0:
            return

        labels = predict_images(model, preprocess, device, args.use_channels_last, images)
        predicted += len(labels)

        for image, source_index, label in zip(images, indices, labels):
            label = int(label)
            source_index = int(source_index)

            if label not in wanted_labels:
                continue

            if len(found[label]) >= int(args.images_per_label):
                continue

            tmp_path = staging / f"label_{label:04d}_candidate_{len(found[label]):02d}_src_{source_index}.jpeg"
            save_image(tmp_path, image, args.jpeg_quality, args.jpeg_subsampling)

            saved_label = predict_files(
                model,
                preprocess,
                device,
                args.use_channels_last,
                [tmp_path],
                batch_size=1,
            )[0]

            checked_after_save += 1

            if int(saved_label) == label:
                found[label].append(
                    {
                        "source_index": source_index,
                        "tmp_path": str(tmp_path),
                        "label": label,
                        "reload_prediction": int(saved_label),
                    }
                )
                kept_after_save += 1
            else:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        images = []
        indices = []

    for row_index, row in enumerate(stream):
        if args.max_records > 0 and row_index >= int(args.max_records):
            break

        rows_seen += 1
        images.append(read_image(row[args.image_column]))
        indices.append(int(row_index))

        if len(images) >= int(args.batch_size):
            flush_batch()
            elapsed = time.perf_counter() - start_time
            print(
                f"model={args.model} rows_seen={rows_seen} predicted={predicted} "
                f"disk_checked={checked_after_save} disk_kept={kept_after_save} "
                f"completed_labels={labels_done()}/{args.num_labels} elapsed_s={elapsed:.1f}",
                flush=True,
            )
            if all_done():
                break

    flush_batch()

    missing = {
        int(label): int(args.images_per_label) - len(found[label])
        for label in sorted(wanted_labels)
        if len(found[label]) < int(args.images_per_label)
    }

    if len(missing) > 0:
        missing_file = staging / "missing_labels.json"
        with missing_file.open("w") as handle:
            json.dump(missing, handle, indent=2)
        raise RuntimeError(f"Could not collect enough images for {len(missing)} labels. See {missing_file}")

    return found, rows_seen, predicted, checked_after_save, kept_after_save


def write_quads(out_dir, found, args):
    quads = []

    for label in range(int(args.num_labels)):
        quad_id = int(label) + 1
        chosen = found[label][:int(args.images_per_label)]

        if len(chosen) != 4:
            raise RuntimeError(f"Expected 4 images for label={label}, got {len(chosen)}")

        filenames = []

        for corner, item in zip(["a", "b", "c", "d"], chosen):
            src = Path(item["tmp_path"])
            filename = image_name(quad_id, corner, args.filename_width)
            dst = out_dir / filename
            save_image(dst, read_image_path(src), args.jpeg_quality, args.jpeg_subsampling)
            filenames.append(filename)

        quads.append(
            {
                "quad_id": quad_id,
                "label": int(label),
                "indices": [int(item["source_index"]) for item in chosen],
                "filenames": filenames,
                "source": "disk_stable_model_prediction",
                "model": args.model,
                "predictions": [int(label), int(label), int(label), int(label)],
            }
        )

    return quads


def verify_quads(out_dir, quads, model, preprocess, device, args):
    paths = []
    targets = []

    for quad in sorted(quads, key=lambda item: int(item["quad_id"])):
        label = int(quad["label"])
        for corner in ["a", "b", "c", "d"]:
            paths.append(out_dir / image_name(int(quad["quad_id"]), corner, args.filename_width))
            targets.append(label)

    labels = predict_files(model, preprocess, device, args.use_channels_last, paths, batch_size=int(args.verify_batch_size))
    bad = []

    for path, label, target in zip(paths, labels, targets):
        if int(label) != int(target):
            bad.append({"path": str(path), "prediction": int(label), "target": int(target)})

    if len(bad) > 0:
        bad_file = out_dir / "final_verify_bad.json"
        with bad_file.open("w") as handle:
            json.dump(bad, handle, indent=2)
        raise RuntimeError(f"Final disk verification failed for {len(bad)} images. See {bad_file}")

    print(f"Final disk verification passed for model={args.model}", flush=True)


def write_metadata(out_dir, quads, args, rows_seen, predicted, checked_after_save, kept_after_save):
    metadata_file = out_dir / "metadata.csv"
    label_file = out_dir / "label_summary.csv"
    quads_file = out_dir / "quads.json"
    run_file = out_dir / "run.json"

    with metadata_file.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "quad_id", "position", "filename", "source_index", "label", "source"],
        )
        writer.writeheader()
        for quad in sorted(quads, key=lambda item: int(item["quad_id"])):
            for corner, filename, source_index in zip(["a", "b", "c", "d"], quad["filenames"], quad["indices"]):
                writer.writerow(
                    {
                        "model": args.model,
                        "quad_id": int(quad["quad_id"]),
                        "position": corner,
                        "filename": filename,
                        "source_index": int(source_index),
                        "label": int(quad["label"]),
                        "source": quad["source"],
                    }
                )

    with label_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "label", "quad_id", "filenames", "source_indices"])
        writer.writeheader()
        for quad in sorted(quads, key=lambda item: int(item["label"])):
            writer.writerow(
                {
                    "model": args.model,
                    "label": int(quad["label"]),
                    "quad_id": int(quad["quad_id"]),
                    "filenames": " ".join(quad["filenames"]),
                    "source_indices": " ".join(str(int(value)) for value in quad["indices"]),
                }
            )

    with quads_file.open("w") as handle:
        json.dump(quads, handle, indent=2)

    run_info = {
        "model": args.model,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "output_dir": str(out_dir),
        "num_labels": int(args.num_labels),
        "images_per_label": int(args.images_per_label),
        "final_quads": len(quads),
        "final_unique_labels": len(set(int(quad["label"]) for quad in quads)),
        "rows_scanned": int(rows_seen),
        "model_predicted_images": int(predicted),
        "stable_candidates_tested": int(checked_after_save),
        "stable_candidates_kept": int(kept_after_save),
        "jpeg_quality": int(args.jpeg_quality),
        "jpeg_subsampling": int(args.jpeg_subsampling),
        "filename_width": int(args.filename_width),
        "batch_size": int(args.batch_size),
        "verify_batch_size": int(args.verify_batch_size),
        "final_disk_verified": True,
        "created_at_unix": time.time(),
    }

    with run_file.open("w") as handle:
        json.dump(run_info, handle, indent=2)

    print(f"Metadata written: {metadata_file}", flush=True)
    print(f"Label summary written: {label_file}", flush=True)
    print(f"Quads written: {quads_file}", flush=True)
    print(f"Run info written: {run_file}", flush=True)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    token = get_token(args)
    out_dir = output_folder(args)

    if not make_output_dir(out_dir, args):
        return

    staging = make_staging_dir(out_dir)

    device = pick_device(args.device)
    setup_precision(device, args.use_tf32)

    print(f"Model: {args.model}", flush=True)
    print(f"Dataset: {args.dataset_name}", flush=True)
    print(f"Split: {args.split}", flush=True)
    print(f"Output dir: {out_dir}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Num labels: {args.num_labels}", flush=True)
    print(f"Images per label: {args.images_per_label}", flush=True)
    print(f"JPEG quality: {args.jpeg_quality}", flush=True)
    print(f"JPEG subsampling: {args.jpeg_subsampling}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)

    model, preprocess = load_model(args.model, device, args.use_channels_last)

    found, rows_seen, predicted, checked_after_save, kept_after_save = collect_images(
        args,
        token,
        model,
        preprocess,
        device,
        staging,
    )

    quads = write_quads(out_dir, found, args)

    if len(quads) != int(args.num_labels):
        raise RuntimeError(f"Expected {args.num_labels} quads, got {len(quads)}")

    labels = [int(quad["label"]) for quad in quads]

    if set(labels) != set(range(int(args.num_labels))):
        raise RuntimeError("Final labels are not exactly 0..num_labels-1")

    verify_quads(out_dir, quads, model, preprocess, device, args)

    write_metadata(
        out_dir,
        quads,
        args,
        rows_seen=rows_seen,
        predicted=predicted,
        checked_after_save=checked_after_save,
        kept_after_save=kept_after_save,
    )

    if staging.exists() and not args.keep_staging:
        shutil.rmtree(staging)

    print("Done", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Output dir: {out_dir}", flush=True)
    print(f"Final quads: {len(quads)}", flush=True)
    print(f"Final unique labels: {len(set(labels))}", flush=True)
    print(f"Rows scanned: {rows_seen}", flush=True)
    print(f"Stable candidates tested: {checked_after_save}", flush=True)
    print(f"Stable candidates kept: {kept_after_save}", flush=True)


if __name__ == "__main__":
    main()
