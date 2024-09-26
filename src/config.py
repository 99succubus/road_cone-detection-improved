import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '123456',
    'database': 'road_cone'
}

MODEL_CONFIG = {
    'model_size': 'm',  # 使用YOLOv8m模型
    'img_size': 640,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'conf_thres': 0.20,
    'iou_thres': 0.45,
    'detection_color': (0, 255, 0),
    'detection_thickness': 2,
    'pretrained_model_path': 'yolov8m.pt',  # 确保这个文件在正确的位置
    'pretrained': True
}

TRAIN_CONFIG = {
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 0.0001,
    'final_learning_rate': 0.000001,
    'weight_decay': 0.0005,
    'momentum': 0.937,
    'optimizer': 'Adam',
    'lr_scheduler': 'cos',  # 使用余弦退火
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'early_stopping_patience': 40,
    'val_size': 0.2,
    'num_workers': 4,
    'pin_memory': True
}

DATA_CONFIG = {
    'yaml_path': os.path.join(PROJECT_ROOT, 'dataset.yaml'),
    'road_cone_dir': os.path.join(PROJECT_ROOT, 'data', 'images'),
    'labels_dir': os.path.join(PROJECT_ROOT, 'data', 'labels'),
    'train_file': os.path.join(PROJECT_ROOT, 'data', 'train.txt'),
    'val_file': os.path.join(PROJECT_ROOT, 'data', 'val.txt'),
    'test_images_dir': os.path.join(PROJECT_ROOT, 'data', 'test_images'),
}

CLASS_MAPPING = {
    'road_cone': 0
}

OUTPUT_CONFIG = {
    'cache_file': os.path.join(PROJECT_ROOT, 'images.cache'),
    'runs_dir': os.path.join(PROJECT_ROOT, 'models', 'output')
}

EVALUATION_CONFIG = {
    'conf_thres': 0.25,
    'iou_thres': 0.45,
    'error_analysis': {
        'save_false_positives': True,
        'save_false_negatives': True,
        'confidence_threshold_analysis': True
    },
    'visualization': {
        'scatter_plot': {
            'figsize': (10, 6),
            'filename': 'scatter_plot.png'
        },
        'confusion_matrix': {
            'figsize': (12, 10),
            'cmap': 'Blues',
            'filename': 'confusion_matrix.png'
        },
        'error_distribution': {
            'figsize': (10, 6),
            'filename': 'error_distribution.png'
        },
        'confidence_distribution': {
            'figsize': (10, 6),
            'bins': 20,
            'filename': 'confidence_distribution.png'
        },
        'precision_recall_curve': {
            'figsize': (10, 6),
            'filename': 'precision_recall_curve.png'
        }
    }
}