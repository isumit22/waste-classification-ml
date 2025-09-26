import os
from PIL import Image
from pycocotools.coco import COCO

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TACO_ANN = os.path.join(BASE_DIR, "TACO/data/annotations.json")
TACO_IMG = os.path.join(BASE_DIR, "TACO/data/images")

OUT_RECYCLE = os.path.join(BASE_DIR, "dataset/recycle")
OUT_NON = os.path.join(BASE_DIR, "dataset/non_recycle")

os.makedirs(OUT_RECYCLE, exist_ok=True)
os.makedirs(OUT_NON, exist_ok=True)

# Mapping TACO → Your labels
recycle_classes = {
    'aluminium_foil', 'carton', 'glass_bottle',
    'metal_can', 'plastic_bag', 'plastic_bottle',
    'plastic_container', 'plastic_cup', 'plastic_lid',
    'plastic_utensils', 'paper', 'cardboard'
}
non_recycle_classes = {
    'styrofoam_piece', 'food_wrapper', 'cigarette',
    'diaper', 'chip_bag', 'toothbrush',
    'sponge', 'other_plastics', 'other_rubber'
}

# Load COCO
coco = COCO(TACO_ANN)

for ann in coco.loadAnns(coco.getAnnIds()):
    cat = coco.loadCats(ann['category_id'])[0]['name']
    if cat in recycle_classes:
        target_folder = OUT_RECYCLE
    elif cat in non_recycle_classes:
        target_folder = OUT_NON
    else:
        continue  # skip unmapped

    img_info = coco.loadImgs(ann['image_id'])[0]
    img_path = os.path.join(TACO_IMG, img_info['file_name'])

    if not os.path.exists(img_path):
        continue

    try:
        with Image.open(img_path) as img:
            x, y, w, h = ann['bbox']
            cropped = img.crop((x, y, x + w, y + h))
            save_name = f"{img_info['id']}_{ann['id']}.jpg"
            cropped.save(os.path.join(target_folder, save_name))
    except Exception as e:
        print(f"Error: {e}")
