import os
import shutil
import random
import glob

# --- CONFIGURATION ---
# Base paths - update these to matching your absolute paths on the Ubuntu server
CLEANED_DATA_ROOT = "/home/ubuntu/cleanedData"
LABELS_SRC_DIR = "/home/ubuntu/yolo/labels"
OUTPUT_DATASET_DIR = "/home/ubuntu/dataset"

# Split ratio (0.8 = 80% train, 20% val)
TRAIN_RATIO = 0.8

def prepare_yolo_dataset():
    # 1. Create directory structure
    subdirs = [
        "images/train", "images/val",
        "labels/train", "labels/val"
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(OUTPUT_DATASET_DIR, subdir), exist_ok=True)

    # 2. Find all images recursively in cleanedData
    # Matches .jpg, .png, .jpeg
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(glob.glob(os.path.join(CLEANED_DATA_ROOT, "**", ext), recursive=True))

    print(f"Found {len(image_paths)} total images.")

    # 3. Match images with labels
    matched_pairs = []
    missing_labels = 0

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        img_id = os.path.splitext(img_name)[0]
        label_path = os.path.join(LABELS_SRC_DIR, f"{img_id}.txt")

        # In your Option B, healthy prints might have no label file, 
        # or an empty label file. Let's decide how to handle missing files:
        if os.path.exists(label_path):
            matched_pairs.append((img_path, label_path))
        else:
            # For Option B, if label doesn't exist, we create an empty one to make it background
            matched_pairs.append((img_path, None))
            missing_labels += 1

    print(f"Matched {len(matched_pairs)} pairs. (Incl. {missing_labels} background images with no labels).")

    # 4. Shuffle and Split
    random.shuffle(matched_pairs)
    split_idx = int(len(matched_pairs) * TRAIN_RATIO)
    train_pairs = matched_pairs[:split_idx]
    val_pairs = matched_pairs[split_idx:]

    # 5. Move/Copy files
    def process_split(pairs, split_name):
        print(f"Processing {split_name} split ({len(pairs)} files)...")
        for img_src, label_src in pairs:
            img_name = os.path.basename(img_src)
            img_id = os.path.splitext(img_name)[0]
            
            # Destination paths
            img_dst = os.path.join(OUTPUT_DATASET_DIR, "images", split_name, img_name)
            label_dst = os.path.join(OUTPUT_DATASET_DIR, "labels", split_name, f"{img_id}.txt")

            # Copy image
            shutil.copy2(img_src, img_dst)

            # Copy label or create empty one
            if label_src and os.path.exists(label_src):
                shutil.copy2(label_src, label_dst)
            else:
                with open(label_dst, 'w') as f:
                    pass # Create empty file for background

    process_split(train_pairs, "train")
    process_split(val_pairs, "val")

    print("\n--- Done! ---")
    print(f"Dataset ready at: {OUTPUT_DATASET_DIR}")
    print("Don't forget to update your 'path' inside data.yaml to point to this new folder!")

if __name__ == "__main__":
    prepare_yolo_dataset()
