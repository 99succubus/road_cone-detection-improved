import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import logging
import cv2
import yaml
from yolo import RoadConeDetector
import matplotlib.pyplot as plt
from config import DATA_CONFIG, EVALUATION_CONFIG, OUTPUT_CONFIG, TRAIN_CONFIG, MODEL_CONFIG
import shutil
from datapreprocess import preprocess_data
import torch
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_learning_curves(lr_history, miou_history, loss_history, output_dir):
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.plot(lr_history)
    plt.title('Learning Rate over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')

    plt.subplot(1, 4, 2)
    plt.plot(miou_history)
    plt.title('MIOU over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MIOU')

    plt.subplot(1, 4, 3)
    plt.plot(loss_history['train'])
    plt.title('Train Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 4, 4)
    plt.plot(loss_history['val'])
    plt.title('Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()

def plot_precision_recall_curve(precisions, recalls, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()

def plot_confusion_matrix(confusion_matrix, output_dir):
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def save_model_summary(model, output_dir):
    summary = str(model)
    with open(os.path.join(output_dir, 'model_summary.txt'), 'w') as f:
        f.write(summary)

def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_device(0)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("GPU not available, using CPU.")

    # 数据预处理
    yaml_file, data = preprocess_data()

    # 创建输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_CONFIG['runs_dir'], f'train_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # 初始化模型
    logger.info("Initializing model...")
    model = RoadConeDetector(yaml_file)

    # 保存模型摘要
    save_model_summary(model.model, output_dir)

    # 训练模型
    logger.info("Starting training...")
    results = model.train_model(name='road_cone_detection', output_dir=output_dir)

    logger.info("Training completed.")
    logger.info(f"Train loss: {results['train_loss']:.4f}")
    logger.info(f"Val loss: {results['val_loss']:.4f}")

    # 绘制学习曲线
    lr_history = results['results'].get('lr', [])
    miou_history = results['results'].get('metrics/mAP50-95(B)', [])
    loss_history = {
        'train': [results['train_box_loss'], results['train_cls_loss'], results['train_dfl_loss']],
        'val': [results['val_box_loss'], results['val_cls_loss'], results['val_dfl_loss']]
    }

    plot_learning_curves(lr_history, miou_history, loss_history, output_dir)

    # 评估模型
    logger.info("Evaluating model...")
    eval_results = model.evaluate(yaml_file)
    logger.info(f"Evaluation results: {eval_results}")

    # 保存评估结果
    with open(os.path.join(output_dir, 'evaluation_results.yaml'), 'w') as f:
        yaml.dump(eval_results, f)

    # 绘制精确度-召回率曲线
    if 'metrics/precision(B)' in eval_results and 'metrics/recall(B)' in eval_results:
        plot_precision_recall_curve(eval_results['metrics/precision(B)'], eval_results['metrics/recall(B)'], output_dir)

    # 绘制混淆矩阵（如果可用）
    if 'confusion_matrix' in eval_results:
        plot_confusion_matrix(eval_results['confusion_matrix'], output_dir)

    # 保存最佳模型
    best_model_path = os.path.join(output_dir, 'road_cone_detection', 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        final_model_path = os.path.join(output_dir, 'best_model.pt')
        shutil.copy(best_model_path, final_model_path)
        logger.info(f"Best model saved at: {final_model_path}")
    else:
        logger.warning(f"Best model file not found at {best_model_path}")

    # 保存训练配置
    config = {
        'TRAIN_CONFIG': TRAIN_CONFIG,
        'MODEL_CONFIG': MODEL_CONFIG,
        'DATA_CONFIG': DATA_CONFIG,
        'EVALUATION_CONFIG': EVALUATION_CONFIG
    }
    with open(os.path.join(output_dir, 'training_config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # 保存数据集统计信息
    dataset_stats = {
        'train_size': len(data['train']),
        'val_size': len(data['val']),
        'total_size': len(data['train']) + len(data['val'])
    }
    with open(os.path.join(output_dir, 'dataset_stats.yaml'), 'w') as f:
        yaml.dump(dataset_stats, f)

    logger.info(f"All outputs saved to: {output_dir}")

    # 示例测试
    test_images = data['val'][:5]  # 详细展示前5个验证图像作为测试
    for img_data in test_images:
        img_path = img_data[0]  # Assuming the image path is the first element in the tuple
        try:
            result = model.detect(img_path)
            visualized_img, num_detections = model.visualize_detection(img_path)
            cv2.imwrite(os.path.join(output_dir, f'detection_{os.path.basename(img_path)}'), visualized_img)
            logger.info(f"Detected {num_detections} road cones in {img_path}")
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {str(e)}")

if __name__ == "__main__":
    main()