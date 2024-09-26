import cv2
import os
import pandas as pd
import torch
from yolo import RoadConeDetector, get_latest_model_path
from config import MODEL_CONFIG, EVALUATION_CONFIG, OUTPUT_CONFIG
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve
import seaborn as sns
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_annotations(xlsx_path):
    try:
        df = pd.read_excel(xlsx_path, engine='openpyxl')
        logger.info("Successfully loaded annotations from Excel file")
        logger.info(f"Annotations:\n{df}")
        return dict(zip(df['image_name'], df['number']))
    except Exception as e:
        logger.error(f"An error occurred while loading the Excel file: {e}")
        return None


def test_road_cones(image_folder, annotation_file):
    try:
        device = torch.device(MODEL_CONFIG['device'])
        model = RoadConeDetector(
            yaml_file=None,
            device=device,
            conf_thres=MODEL_CONFIG['conf_thres'],
            iou_thres=MODEL_CONFIG['iou_thres'],
            pretrained=MODEL_CONFIG['pretrained']
        )

        annotations = load_annotations(annotation_file)
        if annotations is None:
            logger.error("Failed to load annotations. Exiting.")
            return

        total_images = 0
        correct_count = 0
        true_counts = []
        predicted_counts = []
        confidences = []

        # 创建输出目录
        output_dir = os.path.join(OUTPUT_CONFIG['runs_dir'], 'test_results')
        os.makedirs(output_dir, exist_ok=True)

        # 创建一个列表来存储所有图片的路径
        all_image_paths = []

        for image_name, true_count in annotations.items():
            image_path = os.path.join(image_folder, image_name)
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue

            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Failed to read image: {image_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = model.detect(image_path)
            detected_count = len(results[0].boxes)

            # 为每张图片初始化一个置信度列表
            image_confidences = []

            # 在图像上绘制检测框和置信度
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                image_confidences.append(conf)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), MODEL_CONFIG['detection_color'], MODEL_CONFIG['detection_thickness'])
                cv2.putText(img, f"{conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, MODEL_CONFIG['detection_color'], 2)

            # 如果没有检测到任何物体，添加一个0的置信度
            if not image_confidences:
                image_confidences.append(0)

            # 使用这张图片的最高置信度（如果有检测到物体）或0（如果没有检测到物体）
            confidences.append(max(image_confidences))

            # 在图片上添加真实数量和预测数量的文字
            cv2.putText(img, f"True: {true_count}, Pred: {detected_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 保存带有检测结果的图片
            output_path = os.path.join(output_dir, f"detected_{image_name}")
            cv2.imwrite(output_path, img)
            all_image_paths.append(output_path)

            if detected_count == true_count:
                correct_count += 1

            logger.info(f"Image: {image_name}, Detected cones: {detected_count}, True cones: {true_count}")

            total_images += 1
            true_counts.append(true_count)
            predicted_counts.append(detected_count)

        accuracy = correct_count / total_images if total_images > 0 else 0

        logger.info(f"Total images: {total_images}")
        logger.info(f"Correctly detected: {correct_count}")
        logger.info(f"Accuracy: {accuracy:.4f}")

        # 检查 confidences 长度
        if len(confidences) != total_images:
            logger.error(
                f"Number of confidence scores ({len(confidences)}) doesn't match total images ({total_images})")
            return  # 如果不匹配，提前退出函数

        # Visualize results
        visualize_results(true_counts, predicted_counts, confidences, output_dir)

        # 创建包含所有测试图片的大图
        create_summary_image(all_image_paths, output_dir)

    except Exception as e:
        logger.error(f"An error occurred during testing: {str(e)}", exc_info=True)


def visualize_results(true_counts, predicted_counts, confidences, output_dir):
    # Scatter plot
    plt.figure(figsize=EVALUATION_CONFIG['visualization']['scatter_plot']['figsize'])
    plt.scatter(true_counts, predicted_counts)
    plt.plot([0, max(true_counts)], [0, max(true_counts)], 'r--')
    plt.xlabel('True Count')
    plt.ylabel('Predicted Count')
    plt.title('True vs Predicted Cone Counts')
    plt.savefig(os.path.join(output_dir, EVALUATION_CONFIG['visualization']['scatter_plot']['filename']))
    plt.close()

    # Confusion matrix
    max_count = max(max(true_counts), max(predicted_counts))
    cm = confusion_matrix(true_counts, predicted_counts, labels=range(max_count + 1))
    plt.figure(figsize=EVALUATION_CONFIG['visualization']['confusion_matrix']['figsize'])
    sns.heatmap(cm, annot=True, fmt='d', cmap=EVALUATION_CONFIG['visualization']['confusion_matrix']['cmap'])
    plt.xlabel('Predicted Count')
    plt.ylabel('True Count')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, EVALUATION_CONFIG['visualization']['confusion_matrix']['filename']))
    plt.close()

    # Error distribution
    errors = np.array(predicted_counts) - np.array(true_counts)
    plt.figure(figsize=EVALUATION_CONFIG['visualization']['error_distribution']['figsize'])
    plt.hist(errors, bins=range(min(errors), max(errors) + 2, 1), align='left', rwidth=0.8)
    plt.xlabel('Error (Predicted - True)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.savefig(os.path.join(output_dir, EVALUATION_CONFIG['visualization']['error_distribution']['filename']))
    plt.close()

    # Confidence distribution
    plt.figure(figsize=EVALUATION_CONFIG['visualization']['confidence_distribution']['figsize'])
    plt.hist(confidences, bins=EVALUATION_CONFIG['visualization']['confidence_distribution']['bins'])
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Confidence Distribution')
    plt.savefig(os.path.join(output_dir, EVALUATION_CONFIG['visualization']['confidence_distribution']['filename']))
    plt.close()

    # Precision-Recall curve
    if EVALUATION_CONFIG['error_analysis']['confidence_threshold_analysis']:
        binary_labels = [1 if pred > 0 else 0 for pred in predicted_counts]

        precisions, recalls, thresholds = precision_recall_curve(binary_labels, confidences)
        plt.figure(figsize=EVALUATION_CONFIG['visualization']['precision_recall_curve']['figsize'])
        plt.plot(recalls, precisions)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(os.path.join(output_dir, EVALUATION_CONFIG['visualization']['precision_recall_curve']['filename']))
        plt.close()

    logger.info(f"Results visualization completed. Images saved in {output_dir}")


def create_summary_image(image_paths, output_dir):
    images = [cv2.imread(path) for path in image_paths]
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)

    # 确定网格大小
    grid_size = int(np.ceil(np.sqrt(len(images))))

    # 创建大图
    summary_image = np.zeros((max_height * grid_size, max_width * grid_size, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        row = i // grid_size
        col = i % grid_size
        summary_image[row * max_height:(row + 1) * max_height, col * max_width:(col + 1) * max_width, :] = cv2.resize(
            img, (max_width, max_height))

    cv2.imwrite(os.path.join(output_dir, 'summary_image.png'), summary_image)
    logger.info(f"Summary image created and saved in {output_dir}")


if __name__ == "__main__":
    try:
        image_folder = r"C:\Users\Jerry\PycharmProjects9491\road_cone improved\data\test_images"
        annotation_file = r"C:\Users\Jerry\PycharmProjects9491\road_cone improved\data\test_images\test_annotations.xlsx"
        test_road_cones(image_folder, annotation_file)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)