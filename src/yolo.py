import torch
from ultralytics import YOLO
import cv2
import logging
import psutil
import os
import tempfile
import yaml
from datetime import datetime
from config import MODEL_CONFIG, TRAIN_CONFIG, OUTPUT_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")


def get_latest_model_path():
    output_dir = OUTPUT_CONFIG['runs_dir']
    train_dirs = [d for d in os.listdir(output_dir) if d.startswith('train_')]
    if not train_dirs:
        return MODEL_CONFIG['pretrained_model_path']

    def parse_timestamp(dir_name):
        timestamp_str = dir_name.split('_', 1)[1]
        return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')

    latest_train_dir = max(train_dirs, key=parse_timestamp)
    best_model_path = os.path.join(output_dir, latest_train_dir, 'road_cone_detection', 'weights', 'best.pt')

    if os.path.exists(best_model_path):
        return best_model_path
    else:
        logger.warning(f"Best model not found in {best_model_path}. Using pretrained model.")
        return MODEL_CONFIG['pretrained_model_path']


class RoadConeDetector(torch.nn.Module):
    def __init__(self, yaml_file=None, device=None, conf_thres=0.25, iou_thres=0.45, pretrained=True):
        super(RoadConeDetector, self).__init__()
        self.yaml_file = yaml_file
        self.device = device if device else torch.device(MODEL_CONFIG['device'])
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.model_path = get_latest_model_path() if pretrained else MODEL_CONFIG['pretrained_model_path']
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        logger.info(f"Loaded {self.model_path} model on device: {self.device}")

    def train_model(self, name=None, output_dir=None):
        try:
            print_memory_usage()

            if isinstance(self.yaml_file, dict):
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as temp_yaml:
                    yaml.dump(self.yaml_file, temp_yaml)
                    yaml_path = temp_yaml.name
            else:
                yaml_path = self.yaml_file

            logger.info(f"Training with yaml_path: {yaml_path}")
            logger.info(f"yaml_path type: {type(yaml_path)}")

            if not os.path.exists(yaml_path):
                raise FileNotFoundError(f"YAML file not found: {yaml_path}")

            self.model.amp = True

            results = self.model.train(
                data=yaml_path,
                epochs=TRAIN_CONFIG['epochs'],
                imgsz=MODEL_CONFIG['img_size'],
                batch=TRAIN_CONFIG['batch_size'],
                device=self.device,
                workers=TRAIN_CONFIG['num_workers'],
                optimizer=TRAIN_CONFIG['optimizer'],
                lr0=TRAIN_CONFIG['learning_rate'],
                lrf=TRAIN_CONFIG['final_learning_rate'],
                weight_decay=TRAIN_CONFIG['weight_decay'],
                warmup_epochs=TRAIN_CONFIG['warmup_epochs'],
                warmup_momentum=TRAIN_CONFIG['warmup_momentum'],
                warmup_bias_lr=TRAIN_CONFIG['warmup_bias_lr'],
                patience=TRAIN_CONFIG['early_stopping_patience'],
                project=output_dir,
                name=name,
                save_period=-1,
                save=True
            )

            print_memory_usage()

            logger.info(f"Training completed. Results: {results}")

            train_box_loss = results.results_dict.get('train/box_loss', 0)
            train_cls_loss = results.results_dict.get('train/cls_loss', 0)
            train_dfl_loss = results.results_dict.get('train/dfl_loss', 0)
            val_box_loss = results.results_dict.get('val/box_loss', 0)
            val_cls_loss = results.results_dict.get('val/cls_loss', 0)
            val_dfl_loss = results.results_dict.get('val/dfl_loss', 0)

            train_loss = train_box_loss + train_cls_loss + train_dfl_loss
            val_loss = val_box_loss + val_cls_loss + val_dfl_loss

            logger.info(f"Calculated train loss: {train_loss:.4f}")
            logger.info(f"Calculated val loss: {val_loss:.4f}")

            return {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_box_loss': train_box_loss,
                'train_cls_loss': train_cls_loss,
                'train_dfl_loss': train_dfl_loss,
                'val_box_loss': val_box_loss,
                'val_cls_loss': val_cls_loss,
                'val_dfl_loss': val_dfl_loss,
                'results': results.results_dict
            }
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            raise

    def evaluate(self, yaml_path):
        try:
            results = self.model.val(data=yaml_path, device=self.device)
            logger.info(f"Evaluation results keys: {results.keys}")
            return {
                "precision": results.results_dict['metrics/precision(B)'],
                "recall": results.results_dict['metrics/recall(B)'],
                "mAP50": results.results_dict['metrics/mAP50(B)'],
                "mAP50-95": results.results_dict['metrics/mAP50-95(B)']
            }
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
            raise

    def detect(self, image_path):
        try:
            if not isinstance(image_path, str):
                raise TypeError(f"Expected string for image_path, got {type(image_path)}")

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image from {image_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.model(image, conf=self.conf_thres, iou=self.iou_thres, device=self.device)
            return results
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            raise

    def visualize_detection(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image from {image_path}")

            results = self.detect(image_path)
            detected_cones = len(results[0].boxes)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf.item()
                    cv2.rectangle(
                        image,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        MODEL_CONFIG['detection_color'],
                        MODEL_CONFIG['detection_thickness']
                    )
                    cv2.putText(
                        image,
                        f"{conf:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        MODEL_CONFIG['detection_color'],
                        2
                    )
            return image, detected_cones
        except Exception as e:
            logger.error(f"Error during visualization: {str(e)}", exc_info=True)
            raise