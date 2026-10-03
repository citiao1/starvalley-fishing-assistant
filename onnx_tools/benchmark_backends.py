from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import cv2

from fishing_assistant.onnx_detector import OnnxBiteDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX detections.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument(
        "--provider",
        choices=("cuda", "directml", "cpu"),
        default="cuda",
    )
    args = parser.parse_args()

    from ultralytics import YOLO

    torchlib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torchlib.exists():
        os.environ["PATH"] = f"{torchlib}{os.pathsep}{os.environ.get('PATH', '')}"
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(torchlib))

    pytorch = YOLO(str(args.weights))
    import torch

    torch_device = "0" if torch.cuda.is_available() else "cpu"
    onnx = OnnxBiteDetector(
        args.onnx,
        confidence=0.05,
        image_size=args.imgsz,
        use_cuda=torch_device != "cpu",
        dll_search_paths=(torchlib,),
        execution_provider=args.provider,
    )
    print(onnx.load())

    paths = sorted(args.images.glob("*.jpg"))[: args.limit]
    if not paths:
        raise FileNotFoundError(f"No JPG images found in {args.images}")

    pytorch.predict(
        cv2.imread(str(paths[0])),
        imgsz=args.imgsz,
        conf=0.05,
        max_det=5,
        device=torch_device,
        half=torch_device != "cpu",
        verbose=False,
    )
    onnx.detect(
        OnnxBiteDetector.resize_for_inference(
            cv2.imread(str(paths[0])),
            max_width=args.imgsz,
        )[0]
    )

    both = 0
    only_pytorch = 0
    only_onnx = 0
    ious: list[float] = []
    confidence_deltas: list[float] = []
    pytorch_ms: list[float] = []
    onnx_ms: list[float] = []

    for path in paths:
        original = cv2.imread(str(path))
        resized, scale_x, scale_y = OnnxBiteDetector.resize_for_inference(
            original,
            max_width=args.imgsz,
        )

        started = time.perf_counter()
        result = pytorch.predict(
            original,
            imgsz=args.imgsz,
            conf=0.05,
            max_det=5,
            device=torch_device,
            half=torch_device != "cpu",
            verbose=False,
        )[0]
        pytorch_ms.append((time.perf_counter() - started) * 1000)
        pytorch_boxes = []
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls[0].item()) == 0:
                    xyxy = [float(value) for value in box.xyxy[0].tolist()]
                    pytorch_boxes.append((float(box.conf[0].item()), xyxy))

        started = time.perf_counter()
        onnx_boxes, _ = onnx.detect(resized)
        onnx_ms.append((time.perf_counter() - started) * 1000)
        onnx_boxes = [
            (
                confidence,
                [
                    x1 * scale_x,
                    y1 * scale_y,
                    x2 * scale_x,
                    y2 * scale_y,
                ],
            )
            for x1, y1, x2, y2, confidence in onnx_boxes
        ]

        if pytorch_boxes and onnx_boxes:
            both += 1
            ious.append(_iou(pytorch_boxes[0][1], onnx_boxes[0][1]))
            confidence_deltas.append(
                abs(pytorch_boxes[0][0] - onnx_boxes[0][0])
            )
        elif pytorch_boxes:
            only_pytorch += 1
        elif onnx_boxes:
            only_onnx += 1

    print(f"images={len(paths)} both={both}")
    print(f"only_pytorch={only_pytorch} only_onnx={only_onnx}")
    print(f"mean_iou={sum(ious) / len(ious) if ious else 0:.6f}")
    print(
        "max_confidence_delta="
        f"{max(confidence_deltas) if confidence_deltas else 0:.6f}"
    )
    print(f"pytorch_ms={sum(pytorch_ms) / len(pytorch_ms):.2f}")
    print(f"onnx_ms={sum(onnx_ms) / len(onnx_ms):.2f}")
    print(f"pytorch_p50_ms={_percentile(pytorch_ms, 50):.2f}")
    print(f"pytorch_p95_ms={_percentile(pytorch_ms, 95):.2f}")
    print(f"onnx_p50_ms={_percentile(onnx_ms, 50):.2f}")
    print(f"onnx_p95_ms={_percentile(onnx_ms, 95):.2f}")
    print(f"onnx_device={onnx.device} input={onnx.input_shape}")
    print(f"onnx_sha256={_sha256(args.onnx)}")
    return 0


def _iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile // 100
    return ordered[index]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
