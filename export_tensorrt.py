import os
import subprocess
import torch
from ultralytics import YOLO


def export_jetson_engines(
    yolo_weights: str = "yolov8n.pt",
    ap10k_onnx_path: str = os.path.expanduser(
        "~/.cache/rtmlib/hub/checkpoints/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.onnx"
    ),
    output_dir: str = "engines",
):
    """
    Compiles YOLOv8n and AP-10K RTMPose ONNX into FP16 TensorRT engines (.engine)
    on an NVIDIA Jetson Orin Nano (must be executed on target CUDA hardware).
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected! TensorRT .engine files are hardware-specific "
            "and must be compiled directly on the NVIDIA Jetson Orin Nano."
        )

    os.makedirs(output_dir, exist_ok=True)

    # 1. Export YOLOv8n Detector to FP16 TensorRT (.engine)
    print(f"[1/2] Exporting {yolo_weights} to FP16 TensorRT engine...")
    detector = YOLO(yolo_weights)
    exported_yolo = detector.export(format="engine", half=True, imgsz=640, device=0)
    target_yolo_engine = os.path.join(output_dir, "yolov8n_fp16.engine")
    if os.path.exists(exported_yolo) and exported_yolo != target_yolo_engine:
        os.replace(exported_yolo, target_yolo_engine)
    print(f"Saved: {target_yolo_engine}")

    # 2. Export AP-10K RTMPose ONNX to FP16 TensorRT via NVIDIA trtexec
    if not os.path.exists(ap10k_onnx_path):
        raise FileNotFoundError(
            f"AP-10K ONNX file not found at {ap10k_onnx_path}. "
            "Run PawsVisionPipeline once first to download the checkpoint."
        )

    target_pose_engine = os.path.join(output_dir, "rtmpose_ap10k_fp16.engine")
    print(f"[2/2] Compiling {ap10k_onnx_path} via trtexec (FP16)...")
    trtexec_cmd = [
        "/usr/src/tensorrt/bin/trtexec",
        f"--onnx={ap10k_onnx_path}",
        f"--saveEngine={target_pose_engine}",
        "--fp16",
        "--minShapes=input:1x3x256x256",
        "--optShapes=input:1x3x256x256",
        "--maxShapes=input:2x3x256x256",
    ]
    subprocess.run(trtexec_cmd, check=True)
    print(f"Saved: {target_pose_engine}")
    print("\nTensorRT FP16 export complete!")


if __name__ == "__main__":
    export_jetson_engines()