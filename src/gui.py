import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget, \
    QMessageBox, QHBoxLayout, QSlider
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import torch
from yolo import RoadConeDetector, get_latest_model_path
from config import MODEL_CONFIG

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Road Cone Detection")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        button_layout = QHBoxLayout()
        self.select_image_button = QPushButton("Select Image")
        self.select_image_button.clicked.connect(self.select_image)
        button_layout.addWidget(self.select_image_button)

        self.detect_button = QPushButton("Detect Road Cones")
        self.detect_button.clicked.connect(self.detect_road_cones)
        button_layout.addWidget(self.detect_button)

        self.select_model_button = QPushButton("Select Model")
        self.select_model_button.clicked.connect(self.select_model)
        button_layout.addWidget(self.select_model_button)

        self.layout.addLayout(button_layout)

        slider_layout = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(int(MODEL_CONFIG['conf_thres'] * 100))
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.valueChanged.connect(self.update_conf_threshold)
        slider_layout.addWidget(QLabel("Confidence Threshold:"))
        slider_layout.addWidget(self.conf_slider)
        self.conf_label = QLabel(f"{MODEL_CONFIG['conf_thres']:.2f}")
        slider_layout.addWidget(self.conf_label)

        self.layout.addLayout(slider_layout)

        self.model_path_label = QLabel("Current Model: Latest trained model")
        self.layout.addWidget(self.model_path_label)

        self.load_model()

    def load_model(self, model_path=None):
        try:
            if model_path is None:
                model_path = get_latest_model_path()
            device = torch.device(MODEL_CONFIG['device'])
            self.model = RoadConeDetector(
                yaml_file=None,
                device=device,
                conf_thres=MODEL_CONFIG['conf_thres'],
                iou_thres=MODEL_CONFIG['iou_thres'],
                pretrained=True
            )
            self.model_path_label.setText(f"Current Model: {model_path}")
            QMessageBox.information(self, "Success", f"Model loaded successfully from {model_path}")
        except (ValueError, FileNotFoundError) as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.model = None

    def select_model(self):
        options = QFileDialog.Options()
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Model Files (*.pt)", options=options)
        if model_path:
            self.load_model(model_path)

    def select_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.bmp)",
                                                   options=options)
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def detect_road_cones(self):
        if not hasattr(self, 'image_path'):
            QMessageBox.warning(self, "Error", "Please select an image first.")
            return

        if self.model is None:
            QMessageBox.warning(self, "Error", "No model loaded. Please load a model first.")
            return

        try:
            img_with_detections, detected_count = self.model.visualize_detection(self.image_path)

            height, width, channel = img_with_detections.shape
            bytes_per_line = 3 * width
            q_image = QImage(img_with_detections.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            self.setWindowTitle(f"Road Cone Detection - Detected Cones: {detected_count}")

            QMessageBox.information(self, "Detection Result", f"Detected {detected_count} road cone(s).")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Detection failed: {str(e)}")

    def update_conf_threshold(self, value):
        conf_thres = value / 100
        MODEL_CONFIG['conf_thres'] = conf_thres
        if self.model:
            self.model.conf_thres = conf_thres
        self.conf_label.setText(f"{conf_thres:.2f}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())