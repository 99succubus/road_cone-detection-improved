import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import mysql.connector
import random
from PIL import Image
import yaml
import logging
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import DATA_CONFIG, TRAIN_CONFIG, DB_CONFIG, CLASS_MAPPING, MODEL_CONFIG
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.warning(f"Invalid image file {file_path}: {str(e)}")
        return False

def normalize_bbox(xmin, ymin, xmax, ymax, img_width, img_height):
    x_center = (xmin + xmax) / (2 * img_width)
    y_center = (ymin + ymax) / (2 * img_height)
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return int(CLASS_MAPPING['road_cone']), x_center, y_center, width, height

def apply_augmentations(image, bboxes, target_size=MODEL_CONFIG['img_size']):
    if not bboxes:
        transform = A.Compose([
            A.LongestMaxSize(max_size=target_size),
            A.PadIfNeeded(
                min_height=target_size,
                min_width=target_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(114, 114, 114)
            ),
            A.RandomCrop(width=target_size, height=target_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
            A.GaussNoise(p=0.2),
            A.MotionBlur(p=0.2),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0),
            ToTensorV2(),
        ])
        transformed = transform(image=image)
        return transformed['image'], []

    if not isinstance(bboxes[0], (list, tuple)):
        bboxes = [bboxes]

    transform = A.Compose([
        A.LongestMaxSize(max_size=target_size),
        A.PadIfNeeded(
            min_height=target_size,
            min_width=target_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        ),
        A.RandomCrop(width=target_size, height=target_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
        A.GaussNoise(p=0.2),
        A.MotionBlur(p=0.2),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    class_labels = [bbox[0] for bbox in bboxes]
    bbox_list = [bbox[1:] for bbox in bboxes]

    try:
        transformed = transform(image=image, bboxes=bbox_list, class_labels=class_labels)
    except ValueError as e:
        logger.error(f"Error in applying augmentations: {str(e)}")
        logger.error(f"Image shape: {image.shape}, Bboxes: {bboxes}")
        return image, bboxes

    augmented_bboxes = [(label, *box) for label, box in zip(transformed['class_labels'], transformed['bboxes'])]
    return transformed['image'], augmented_bboxes

def load_data_from_mysql(host, user, password, database, val_size=TRAIN_CONFIG['val_size']):
    logger.info("Connecting to MySQL database...")
    try:
        conn = mysql.connector.connect(host=host, user=user, password=password, database=database)
        cursor = conn.cursor()

        cursor.execute("SELECT id, file_path, detected_cone, cone_number FROM road_cone_image")
        all_images = cursor.fetchall()

        # 确保目标目录存在
        ensure_dir(DATA_CONFIG['road_cone_dir'])
        ensure_dir(DATA_CONFIG['labels_dir'])

        data = []
        for image_id, file_path, detected_cone, cone_number in all_images:
            full_path = os.path.join(DATA_CONFIG['road_cone_dir'], os.path.basename(file_path))
            if not os.path.exists(full_path):
                # 如果目标目录中找不到图片,尝试从原始位置复制
                logger.info(f"Image not found in target directory: {full_path}")
                logger.info(f"Attempting to copy from original location: {file_path}")
                try:
                    shutil.copy2(file_path, full_path)
                    logger.info(f"Successfully copied image to: {full_path}")
                except Exception as e:
                    logger.error(f"Failed to copy image: {str(e)}")
                    continue

            if os.path.exists(full_path) and is_valid_image(full_path):
                image = cv2.imread(full_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                cursor.execute("""
                    SELECT class_name, xmin, ymin, xmax, ymax 
                    FROM road_cone_annotations 
                    WHERE image_id = %s
                """, (image_id,))
                annotations = cursor.fetchall()

                normalized_annotations = []
                for ann in annotations:
                    class_name, xmin, ymin, xmax, ymax = ann
                    norm_bbox = normalize_bbox(xmin, ymin, xmax, ymax, image.shape[1], image.shape[0])
                    normalized_annotations.append(norm_bbox)

                try:
                    augmented_image, augmented_bboxes = apply_augmentations(image, normalized_annotations)
                    data.append((full_path, augmented_bboxes, detected_cone, cone_number))
                except Exception as e:
                    logger.error(f"Error applying augmentations to {full_path}: {str(e)}")
                    logger.error(f"Skipping this image.")
                    continue
            else:
                logger.warning(f"Invalid or non-existent image file: {full_path}")

        cursor.close()
        conn.close()

        cone_data = [d for d in data if d[2] > 0]
        non_cone_data = [d for d in data if d[2] == 0]

        random.shuffle(cone_data)
        random.shuffle(non_cone_data)

        val_size_cone = int(len(cone_data) * val_size)
        val_size_non_cone = int(len(non_cone_data) * val_size)

        train_data = cone_data[val_size_cone:] + non_cone_data[val_size_non_cone:]
        val_data = cone_data[:val_size_cone] + non_cone_data[:val_size_non_cone]

        random.shuffle(train_data)
        random.shuffle(val_data)

        logger.info(f"Preprocessed and loaded {len(train_data)} training images and {len(val_data)} validation images")

        return {
            'train': train_data,
            'val': val_data
        }
    except Exception as e:
        logger.error(f"Error preprocessing and loading data: {str(e)}", exc_info=True)
        raise

def create_dataset_files(train_data, val_data):
    train_file = DATA_CONFIG['train_file']
    val_file = DATA_CONFIG['val_file']

    def write_to_file(file_path, image_list):
        cone_count = 0
        no_cone_count = 0
        with open(file_path, 'w') as f:
            for image_path, _, detected_cone, _ in image_list:
                f.write(f"{image_path}\n")
                if detected_cone > 0:
                    cone_count += 1
                else:
                    no_cone_count += 1
        return cone_count, no_cone_count

    train_cone_count, train_no_cone_count = write_to_file(train_file, train_data)
    val_cone_count, val_no_cone_count = write_to_file(val_file, val_data)

    logger.info(f"Created train file at {train_file}")
    logger.info(f"Training set: {train_cone_count} with cones, {train_no_cone_count} without cones")
    logger.info(f"Created validation file at {val_file}")
    logger.info(f"Validation set: {val_cone_count} with cones, {val_no_cone_count} without cones")

    return {
        'train': {'cone': train_cone_count, 'no_cone': train_no_cone_count},
        'val': {'cone': val_cone_count, 'no_cone': val_no_cone_count}
    }

def create_annotation_files(data, output_dir):
    ensure_dir(output_dir)
    for image_path, annotations, detected_cone, cone_number in data:
        base_name = os.path.basename(image_path)
        label_path = os.path.join(output_dir, os.path.splitext(base_name)[0] + '.txt')

        with open(label_path, 'w') as f:
            if detected_cone == 0 or not annotations:
                logger.info(f"Created empty label file for no_cone image: {image_path}")
            else:
                for ann in annotations:
                    class_id, x_center, y_center, width, height = ann
                    f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        logger.debug(f"Created label file for {image_path}")

def create_dataset_yaml(output_path):
    data_yaml = {
        'train': DATA_CONFIG['train_file'],
        'val': DATA_CONFIG['val_file'],
        'nc': len(CLASS_MAPPING),
        'names': list(CLASS_MAPPING.keys())
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Created dataset YAML at {output_path}")

    return output_path

def validate_dataset(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    for split in ['train', 'val']:
        split_file = data[split]
        if not os.path.exists(split_file):
            logger.error(f"Split file not found: {split_file}")
            continue

        with open(split_file, 'r') as f:
            image_paths = f.read().splitlines()

        for img_path in image_paths:
            if not os.path.exists(img_path):
                logger.warning(f"Image file not found: {img_path}")

            label_path = os.path.join(DATA_CONFIG['labels_dir'],
                                      os.path.splitext(os.path.basename(img_path))[0] + '.txt')
            if not os.path.exists(label_path):
                logger.warning(f"Missing label file for {img_path}")
            else:
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        logger.info(f"Empty label file (likely no road cone): {label_path}")
                    else:
                        lines = content.split('\n')
                        for line in lines:
                            parts = line.split()
                            if len(parts) != 5 or not all(part.replace('.', '', 1).isdigit() for part in parts[1:]):
                                logger.error(f"Invalid label format in {label_path}: {line}")

    logger.info("Dataset validation completed.")

def preprocess_data():
    logger.info("Starting data preprocessing...")
    data = load_data_from_mysql(**DB_CONFIG)
    create_annotation_files(data['train'] + data['val'], DATA_CONFIG['labels_dir'])
    stats = create_dataset_files(data['train'], data['val'])
    yaml_file = create_dataset_yaml(DATA_CONFIG['yaml_path'])
    validate_dataset(yaml_file)
    logger.info("Data preprocessing completed.")
    return yaml_file, data

if __name__ == "__main__":
    yaml_file, data = preprocess_data()
    logger.info("Dataset statistics:")
    logger.info(f"Training set: {len(data['train'])} images")
    logger.info(f"Validation set: {len(data['val'])} images")