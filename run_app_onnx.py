import os

os.environ.setdefault("FISHING_ASSISTANT_BACKEND", "onnx")
os.environ.setdefault("FISHING_ASSISTANT_ONNX_PROVIDER", "auto")

from run_app import run


if __name__ == "__main__":
    raise SystemExit(run())
