from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the fishing model to dynamic ONNX.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    staged_weights = args.output_dir / args.weights.name
    staged_weights.write_bytes(args.weights.read_bytes())
    model = YOLO(str(staged_weights))
    output = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=1,
        dynamic=True,
        simplify=False,
        nms=False,
        opset=17,
        device="cpu",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
