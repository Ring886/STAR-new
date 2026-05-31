import argparse
import base64
import os

import cv2
import dlib
import numpy as np
import torch

from demo import Alignment


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def normalize_to_uint8(arr):
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    max_value = arr.max()
    if max_value > 1e-6:
        arr = arr / max_value
    return (arr * 255).astype(np.uint8)


def resize_heatmap_for_export(heatmap, size):
    heatmap = heatmap.astype(np.float32)
    heatmap = heatmap - heatmap.min()
    max_value = heatmap.max()
    if max_value > 1e-6:
        heatmap = heatmap / max_value
    resized = cv2.resize(heatmap, (size, size), interpolation=cv2.INTER_CUBIC)
    resized = np.clip(resized, 0.0, 1.0)
    return (resized * 255).astype(np.uint8)


def write_png_svg(svg_path, png_path, width, height):
    with open(png_path, "rb") as png_file:
        encoded = base64.b64encode(png_file.read()).decode("ascii")

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <image width="{width}" height="{height}" href="data:image/png;base64,{encoded}"/>
</svg>
"""
    with open(svg_path, "w", encoding="utf-8") as svg_file:
        svg_file.write(svg)


def get_face_geometry(image, predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    dets = detector(image, 1)
    if len(dets) == 0:
        raise RuntimeError("No face found in input image.")

    face = predictor(image, dets[0])
    shape = np.array([(face.part(i).x, face.part(i).y) for i in range(68)])
    x1, x2 = shape[:, 0].min(), shape[:, 0].max()
    y1, y2 = shape[:, 1].min(), shape[:, 1].max()
    scale = min(x2 - x1, y2 - y1) / 200 * 1.05
    center_w = (x2 + x1) / 2
    center_h = (y2 + y1) / 2
    return float(scale), float(center_w), float(center_h)


def main():
    parser = argparse.ArgumentParser(description="Export one predicted keypoint heatmap.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_definition", default="300W", choices=["300W", "COFW", "WFLW"])
    parser.add_argument("--device_ids", default="0")
    parser.add_argument("--predictor_path", default="model/shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--my_image_path", default="my_images/2.png")
    parser.add_argument("--point_index", type=int, default=36, help="0-based landmark index")
    parser.add_argument("--stack_index", type=int, default=-1, help="Hourglass stack index; -1 means final stack")
    parser.add_argument("--out_dir", default="heatmap_exports")
    parser.add_argument("--export_size", type=int, default=1024, help="side length for high-resolution heatmap exports")
    args = parser.parse_args()

    model_path = resolve_path(args.model_path)
    predictor_path = resolve_path(args.predictor_path)
    image_path = resolve_path(args.my_image_path)
    out_dir = resolve_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)

    align_args = argparse.Namespace(config_name="alignment", data_definition=args.data_definition)
    device_ids = list(map(int, args.device_ids.split(",")))
    alignment = Alignment(align_args, model_path, dl_framework="pytorch", device_ids=device_ids)

    scale, center_w, center_h = get_face_geometry(image, predictor_path)
    input_tensor, matrix = alignment.preprocess(image, scale, center_w, center_h)

    with torch.no_grad():
        outputs = alignment.alignment(input_tensor)

    _, fusionmaps, landmarks = outputs
    heatmaps = fusionmaps[args.stack_index][0]
    if args.point_index < 0 or args.point_index >= heatmaps.shape[0]:
        raise ValueError(f"point_index must be in [0, {heatmaps.shape[0] - 1}], got {args.point_index}")

    heatmap = heatmaps[args.point_index].detach().cpu().numpy()
    heatmap_u8 = normalize_to_uint8(heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_large = resize_heatmap_for_export(heatmap, args.export_size)
    heatmap_large_color = cv2.applyColorMap(heatmap_large, cv2.COLORMAP_JET)
    heatmap_large_rgba = cv2.cvtColor(heatmap_large_color, cv2.COLOR_BGR2BGRA)
    heatmap_large_rgba[:, :, 3] = heatmap_large

    aligned_face = alignment.transformPerspective.process(image, matrix)
    heatmap_256 = cv2.resize(heatmap_u8, (aligned_face.shape[1], aligned_face.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap_256_color = cv2.applyColorMap(heatmap_256, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(aligned_face, 0.55, heatmap_256_color, 0.45, 0)

    landmark_crop = alignment.denorm_points(landmarks).detach().cpu().numpy()[0]
    point = landmark_crop[args.point_index]
    cv2.circle(overlay, (int(round(point[0])), int(round(point[1]))), 3, (0, 255, 0), -1, cv2.LINE_AA)

    prefix = f"{args.data_definition}_point{args.point_index}_stack{args.stack_index}"
    gray_path = os.path.join(out_dir, f"{prefix}_gray.png")
    color_path = os.path.join(out_dir, f"{prefix}_color.png")
    overlay_path = os.path.join(out_dir, f"{prefix}_overlay.png")
    crop_path = os.path.join(out_dir, f"{prefix}_aligned_face.png")
    large_gray_path = os.path.join(out_dir, f"{prefix}_gray_{args.export_size}.png")
    large_color_path = os.path.join(out_dir, f"{prefix}_color_{args.export_size}.png")
    large_transparent_path = os.path.join(out_dir, f"{prefix}_transparent_{args.export_size}.png")
    large_svg_path = os.path.join(out_dir, f"{prefix}_color_{args.export_size}.svg")

    cv2.imwrite(gray_path, heatmap_u8)
    cv2.imwrite(color_path, heatmap_color)
    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(crop_path, aligned_face)
    cv2.imwrite(large_gray_path, heatmap_large)
    cv2.imwrite(large_color_path, heatmap_large_color)
    cv2.imwrite(large_transparent_path, heatmap_large_rgba)
    write_png_svg(large_svg_path, large_color_path, args.export_size, args.export_size)

    print("Saved:")
    print(gray_path)
    print(color_path)
    print(large_gray_path)
    print(large_color_path)
    print(large_transparent_path)
    print(large_svg_path)
    print(overlay_path)
    print(crop_path)
    print("heatmap shape:", tuple(heatmaps.shape))
    print("selected point in aligned 256x256 image:", point.tolist())


if __name__ == "__main__":
    main()
