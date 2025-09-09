import os
import re

# ---------- helpers ----------

def _norm(name: str) -> str:
    # normalize for matching dataset folder names to human-readable names
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")

def _reorder_to_dataset_mapping(canonical_names, dataset):
    """
    Reorder canonical class names to the dataset's internal label order.
    Works with torchvision-style datasets that expose .classes and .class_to_idx.
    """
    # Build lookup from normalized canonical name -> display name
    canon_map = {_norm(n): n for n in canonical_names}
    ordered = []
    for raw in getattr(dataset, "classes", []):
        key = _norm(raw)
        # If we don't find a perfect match, fall back to a friendly version of the raw folder name
        ordered.append(canon_map.get(key, raw.replace("_", " ").lower()))
    return ordered

# ---------- canonical lists ----------

VISDA12 = [
    # VisDA-2017 classification categories
    "aeroplane", "bicycle", "bus", "car", "horse", "knife",
    "motorcycle", "person", "plant", "skateboard", "train", "truck",
]

OFFICE31_31 = [
    # Alphabetical; union of common / source-private / target-private sets (31 total)
    "back_pack", "bike", "bike_helmet", "bookcase", "bottle",
    "calculator", "desk_chair", "desk_lamp", "desktop_computer", "file_cabinet",
    "headphones", "keyboard", "laptop_computer", "letter_tray", "mobile_phone",
    "monitor", "mouse", "mug", "paper_notebook", "pen",
    "phone", "printer", "projector", "punchers", "ring_binder",
    "ruler", "scissors", "speaker", "stapler", "tape_dispenser",
    "trash_can",
]

OFFICEHOME_65 = [
    # Canonical 65 Office-Home classes
    "alarm clock","backpack","batteries","bed","bike","bottle","bucket","calculator","calendar","candles",
    "chair","clipboards","computer","couch","curtains","desk lamp","drill","eraser","exit sign","fan",
    "file cabinet","flipflops","flower","folder","fork","glasses","hammer","helmet","kettle","keyboard",
    "knives","lamp shade","laptop","marker","monitor","mop","mouse","mug","notebook","oven",
    "pan","paper clip","pen","pencil","postit notes","printer","push pin","radio","refrigerator","ruler",
    "scissors","screwdriver","shelf","sink","sneakers","soda","speaker","spoon","table","telephone",
    "toothbrush","toys","trash can","tv","webcam",
]

def load_domainnet_345_from_txt(path: str):
    """
    Load 345 DomainNet class names from a text file (1 class per line).
    Keep names as-is for CLIP prompts; you can lowercase or tweak later if desired.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"Provide domainnet_classes.txt (345 lines). "
            f"Tip: use the official class list (e.g., TFDS/HF DomainNet) and save locally."
        )
    with open(path, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if len(names) != 345:
        raise ValueError(f"Expected 345 classes, got {len(names)} at {path}")
    return names

# ---------- public API ----------

def template_for_domain(domain: str) -> str:
    d = (domain or "").strip().lower()
    if "sketch" in d:
        return "a black-and-white sketch of a {}"
    if "clipart" in d or "clip_art" in d:
        return "a clipart drawing of a {}"
    # fallback
    return "a photo of a {}"

def derive_domainnet_classes_from_dataset(dataset):
    """
    Derive DomainNet class names in the exact label order used by the dataset.
    Works for torchvision-like datasets exposing .classes or .class_to_idx.
    """
    # Preferred: dataset.classes is an ordered list matching label ids
    if hasattr(dataset, "classes") and dataset.classes:
        return list(dataset.classes)
    # Fallback: build from class_to_idx by sorting on the index
    if hasattr(dataset, "class_to_idx") and dataset.class_to_idx:
        return [name for name, idx in sorted(dataset.class_to_idx.items(), key=lambda kv: kv[1])]
    raise ValueError("Dataset does not expose .classes or .class_to_idx for DomainNet.")


def get_clip_class_names(
    source, *, dataset_for_order=None, domainnet_txt_path=None
):
    """
    Returns class_names ordered to match dataset_for_order (if provided).
    - source: your DatasetsType.* or equivalent string
    - dataset_for_order: a dataset object (e.g., src_tr[0]) exposing .classes and .class_to_idx
      so we can reorder canonical names into the exact label order used at runtime.
    - domainnet_txt_path: path to a 345-line file for DomainNet classes
    """
    # Choose canonical
    if str(source).lower().endswith("visda") or str(source).lower() == "visda":
        canonical = VISDA12
    elif "office31" in str(source).lower() or str(source).lower() == "office31":
        canonical = OFFICE31_31
    elif "officehome" in str(source).lower() or str(source).lower() == "officehome":
        canonical = OFFICEHOME_65
    elif "domainnet" in str(source).lower() or str(source).lower() == "domainnet":
        if dataset_for_order is not None:
            # No external file required – get the correct order from the split itself (clipart/sketch)
            canonical = derive_domainnet_classes_from_dataset(dataset_for_order)
        elif domainnet_txt_path:
            # Optional: still allow loading from a 345-line file if provided
            canonical = load_domainnet_345_from_txt(domainnet_txt_path)
        else:
            raise FileNotFoundError(
                "DomainNet class names unavailable: pass dataset_for_order (preferred) "
                "or domainnet_txt_path (optional 345-line file)."
            )
    else:
        raise ValueError(f"Unknown dataset source for CLIP labels: {source}")

    # Reorder to match dataset label mapping if we have a reference dataset
    if dataset_for_order is not None and hasattr(dataset_for_order, "class_to_idx"):
        return _reorder_to_dataset_mapping(canonical, dataset_for_order)
    return canonical

def build_clip_prompts(class_names, template="a photo of a {}"):
    # Minimal prompt builder (you can swap template per domain if you like)
    return [template.format(name) for name in class_names]
