import sys
import cv2
import joblib
from skimage.feature import hog
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import uic

class ImageClassifierApp(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi('classification.ui', self) 

        self.model = None
        self.image_path = None
        self.processed_image = None

        self.setup_connections()
        self.load_model()

    def setup_connections(self):
        self.button_loadCitra.clicked.connect(self.load_image)
        self.button_predict.clicked.connect(self.predict_image)  

    def load_model(self):
        try:
            self.model = joblib.load('best_svc_model.joblib')
            if hasattr(self, 'label_status'):
                self.label_status.setText("Model loaded - Ready to classify")
        except Exception as e:
            QMessageBox.critical(self, "Model Error", 
                               f"Failed to load model 'best_svc_model.joblib':\n{str(e)}")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Image", 
            "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)"
        )

        if file_path:
            try:
                self.image_path = file_path
                pixmap = QPixmap(file_path)

                if hasattr(self, 'label_originalImage'):
                    scaled_pixmap = pixmap.scaled(
                        self.label_originalImage.width(), 
                        self.label_originalImage.height(), 
                        Qt.KeepAspectRatio, 
                        Qt.SmoothTransformation
                    )
                    self.label_originalImage.setPixmap(scaled_pixmap)

                image = cv2.imread(file_path)
                if len(image.shape) == 3:
                    self.processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    self.processed_image = image

                if hasattr(self, 'label_status'):
                    self.label_status.setText("Image loaded - Ready to predict")

                if hasattr(self, 'label_predict_output'):
                    self.label_predict_output.setText("")
                if hasattr(self, 'label_proba_output'):
                    self.label_proba_output.setText("")

            except Exception as e:
                QMessageBox.warning(self, "Image Error", f"Failed to load image:\n{str(e)}")

    def convert_to_hog(self):
        if self.processed_image is None:
            return None, None

        try:
            resized_image = cv2.resize(self.processed_image, (128, 128))
            hog_features, hog_image = hog(
                resized_image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=True,
                block_norm='L2-Hys',
                feature_vector=True
            )
            return hog_features, hog_image
        except Exception as e:
            print(f"HOG conversion error: {str(e)}")
            return None, None

    def predict_image(self):
        if self.model is None:
            QMessageBox.warning(self, "Model Error", "Model not loaded!")
            return

        if self.processed_image is None:
            QMessageBox.warning(self, "Image Error", "Please load an image first!")
            return

        try:
            hog_features, _ = self.convert_to_hog()
            
            if hog_features is None:
                QMessageBox.critical(self, "Processing Error", "Failed to process image!")
                return

            features = hog_features.reshape(1, -1)
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]

            if hasattr(self, 'label_predict_output'):
                self.label_predict_output.setText(f"{prediction}")

            if hasattr(self, 'label_proba_output'):
                max_prob = max(probabilities)
                self.label_proba_output.setText(f"{max_prob:.4f}")

            if hasattr(self, 'label_status'):
                self.label_status.setText(f"Prediction complete - Class: {prediction}")

        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", f"Failed to predict:\n{str(e)}")
            if hasattr(self, 'label_predict_output'):
                self.label_predict_output.setText("Error")
            if hasattr(self, 'label_proba_output'):
                self.label_proba_output.setText("Error")


def main():
    app = QApplication(sys.argv)
    try:
        window = ImageClassifierApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Failed to load UI: {str(e)}")
        print("Make sure your .ui file is in the correct path")


if __name__ == '__main__':
    main()