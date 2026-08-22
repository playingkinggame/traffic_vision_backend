"""
Converts a Roboflow COCO-format export (train/_annotations.coco.json etc,
images sitting directly in train/valid/test) into the YOLO layout
ultralytics expects:

    <out>/train/images/*.jpg
    <out>/train/labels/*.txt
    <out>/valid/images/*.jpg
    <out>/valid/labels/*.txt
    <out>/test/images/*.jpg
    <out>/test/labels/*.txt
    <out>/data.yaml

Usage:
    python convert_coco_to_yolo.py --src "C:/Users/yathin/Downloads/Vehicles-coco.v2i.multiclass" --out "C:/Users/yathin/Downloads/vehicles-yolo"

Then train with:
    python train.py --data "C:/Users/yathin/Downloads/vehicles-yolo/data.yaml" --epochs 100
"""
import argparse
import json
import shutil
from pathlib import Path


def convert_split(src_split: Path, out_split: Path):
    ann_path = src_split / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"  skip {src_split.name}: no _annotations.coco.json found")
        return None

    with ann_path.open() as f:
        coco = json.load(f)

    # Map category_id -> yolo class index, in the order Roboflow lists them.
    # Roboflow's COCO export sometimes includes a supercategory placeholder
    # as id 0 - keep everything as-is and just report the mapping so you can
    # eyeball it against car/bus/bike/truck.
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories)}
    class_names = [c["name"] for c in categories]

    images_by_id = {img["id"]: img for img in coco["images"]}

    # group annotations per image
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    img_out_dir = out_split / "images"
    lbl_out_dir = out_split / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for img_id, img in images_by_id.items():
        file_name = img["file_name"]
        src_img_path = src_split / file_name
        if not src_img_path.exists():
            continue

        shutil.copy2(src_img_path, img_out_dir / file_name)

        w, h = img["width"], img["height"]
        lines = []
        for ann in anns_by_image.get(img_id, []):
            x, y, bw, bh = ann["bbox"]  # COCO: top-left x,y + width,height, absolute px
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            cls_idx = cat_id_to_idx[ann["category_id"]]
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path = lbl_out_dir / (Path(file_name).stem + ".txt")
        label_path.write_text("\n".join(lines))
        converted += 1

    print(f"  {src_split.name}: converted {converted} images")
    return class_names


def main():
    parser = argparse.ArgumentParser(description="Convert Roboflow COCO export to YOLO format")
    parser.add_argument("--src", required=True, help="Path to the Vehicles-coco.v2i.multiclass folder")
    parser.add_argument("--out", required=True, help="Where to write the converted YOLO dataset")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    class_names = None
    for split in ["train", "valid", "test"]:
        src_split = src / split
        if not src_split.exists():
            print(f"  skip {split}: folder not found")
            continue
        print(f"Converting {split}...")
        names = convert_split(src_split, out / split)
        if names:
            class_names = names  # same across splits, last non-empty wins

    if class_names is None:
        raise SystemExit(
            "No _annotations.coco.json found in any split - check --src points "
            "at the Vehicles-coco.v2i.multiclass folder itself."
        )

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )

    print(f"\nDone. Class mapping: {list(enumerate(class_names))}")
    print(f"Wrote {data_yaml}")
    print(f"\nNow train with:\n  python train.py --data \"{data_yaml}\" --epochs 100")


if __name__ == "__main__":
    main()
