import sys
import os
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi
from skimage.feature import hog
from skimage import exposure

class B3DatasetApp(QDialog):
    def __init__(self):
        super().__init__()
        loadUi('dataset_builder.ui', self)
        
        self.original_image = None
        self.processed_image = None
        self.grayscale_image = None
        self.hog_features = None
        self.hog_image = None
        
        self.create_directories()
        
        self.connect_buttons()
        
        self.csv_filename = 'dataset/hog_features.csv'
        self.initialize_csv()
        
    def create_directories(self):
        directories = ['dataset', 'raw', 'dataset/hog_visualizations']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
    def connect_buttons(self):
        try:
            self.button_loadCitra.clicked.connect(self.load_image)
            self.button_convertHOG.clicked.connect(self.convert_to_hog)
            self.button_save.clicked.connect(self.save_data)
            
            self.slider_contrast.valueChanged.connect(self.update_preprocessing)
            self.slider_brightness.valueChanged.connect(self.update_preprocessing)
            self.slider_blur.valueChanged.connect(self.update_preprocessing)
            
        except AttributeError as e:
            print(f"Button tidak ditemukan: {e}")
            print("Pastikan nama button di GUI.ui sesuai dengan yang digunakan di kode")
    
    def initialize_csv(self):
        if not os.path.exists(self.csv_filename):
            df = pd.DataFrame(columns=['filename', 'label'] + [f'f{i}' for i in range(1, 3781)])
            df.to_csv(self.csv_filename, index=False)
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Pilih Gambar", 
            "", 
            "Image files (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                QMessageBox.warning(self, "Error", "Gagal memuat gambar!")
                return
                
            self.grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.processed_image = self.grayscale_image.copy()
            
            self.display_image(self.original_image, self.label_originalImage)
            
            self.display_image(self.grayscale_image, self.label_grayscaleImage, is_grayscale=True)
            
            if hasattr(self, 'label_hogImage'):
                self.label_hogImage.clear()
                
            self.current_filename = os.path.basename(file_path)
            
            self.enable_preprocessing_controls()
    
    def enable_preprocessing_controls(self):
        try:
            self.slider_contrast.setEnabled(True)
            self.slider_brightness.setEnabled(True)
            self.slider_blur.setEnabled(True)
            
            self.slider_contrast.setValue(50)
            self.slider_brightness.setValue(0)
            self.slider_blur.setValue(0)
            
        except AttributeError:
            pass
    
    def update_preprocessing(self):
        if self.grayscale_image is None:
            return
            
        try:
            contrast = self.slider_contrast.value() / 50.0
            brightness = self.slider_brightness.value()
            blur_size = self.slider_blur.value()
            
            processed = self.grayscale_image.copy().astype(np.float32)
            
            processed = processed * contrast + brightness
            processed = np.clip(processed, 0, 255).astype(np.uint8)
            
            if blur_size > 0:
                kernel_size = blur_size * 2 + 1
                processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)
            
            self.processed_image = processed
            
            self.display_image(self.processed_image, self.label_grayscaleImage, is_grayscale=True)
            
        except AttributeError:
            pass
    
    def display_image(self, image, label_widget, is_grayscale=False):
        try:
            if len(image.shape) == 3:
                height, width, channel = image.shape
                bytes_per_line = 3 * width
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            else:
                height, width = image.shape
                bytes_per_line = width
                q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label_widget.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def rotate_image(self, image, angle):
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return image
    
    def convert_to_hog(self):
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "Tidak ada gambar yang dimuat!")
            return
        
        try:
            resized_image = cv2.resize(self.processed_image, (128, 128))
            
            self.hog_features, self.hog_image = hog(
                resized_image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                visualize=True,
                block_norm='L2-Hys',
                feature_vector=True
            )
            
            self.hog_image = exposure.rescale_intensity(self.hog_image, in_range=(0, 10))
            
            hog_display = (self.hog_image * 255).astype(np.uint8)
            
            self.display_image(hog_display, self.label_hogImage, is_grayscale=True)
            
            QMessageBox.information(self, "Success", f"HOG features extracted! Features shape: {self.hog_features.shape}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error extracting HOG features: {str(e)}")
    
    def save_data(self):
        if self.hog_features is None:
            QMessageBox.warning(self, "Warning", "Belum ada HOG features! Lakukan konversi HOG terlebih dahulu.")
            return
        
        try:
            label_text = self.lineEdit_label.text().strip()
            if not label_text:
                QMessageBox.warning(self, "Warning", "Masukkan label untuk data!")
                return
            
            rotation_enabled = False
            try:
                rotation_enabled = self.checkbox_rotate.isChecked()
            except AttributeError:
                pass
            
            base_name = os.path.splitext(self.current_filename)[0]
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            angles = [0, 90, 180, 270] if rotation_enabled else [0]
            
            saved_files = []
            
            for angle in angles:
                if rotation_enabled and angle != 0:
                    unique_name = f"{base_name}_{timestamp}_rot{angle}"
                else:
                    unique_name = f"{base_name}_{timestamp}"
                
                processed_for_save = cv2.resize(self.processed_image, (128, 128))
                
                rotated_image = self.rotate_image(processed_for_save, angle)
                
                raw_path = f"raw/{unique_name}.jpg"
                cv2.imwrite(raw_path, rotated_image)
                
                hog_features_rotated, hog_image_rotated = hog(
                    rotated_image,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    visualize=True,
                    block_norm='L2-Hys',
                    feature_vector=True
                )
                
                hog_image_rotated = exposure.rescale_intensity(hog_image_rotated, in_range=(0, 10))
                hog_viz_path = f"dataset/hog_visualizations/{unique_name}_hog.jpg"
                hog_save = (hog_image_rotated * 255).astype(np.uint8)
                cv2.imwrite(hog_viz_path, hog_save)
                
                self.save_to_csv_with_features(unique_name, label_text, hog_features_rotated)
                
                saved_files.append(f"Raw: {raw_path}, HOG viz: {hog_viz_path}")
            
            files_info = "\n".join(saved_files)
            QMessageBox.information(self, "Success", 
                                  f"Data berhasil disimpan!\n"
                                  f"Total files: {len(angles)} set(s)\n"
                                  f"{files_info}\n"
                                  f"Features saved to CSV")
            
            self.clear_current_data()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving data: {str(e)}")
    
    def save_to_csv_with_features(self, filename, label, hog_features):
        try:
            if os.path.exists(self.csv_filename):
                df = pd.read_csv(self.csv_filename)
            else:
                num_features = len(hog_features)
                df = pd.DataFrame(columns=['filename', 'label'] + [f'f{i}' for i in range(1, num_features + 1)])
            
            new_row = {'filename': filename, 'label': label}
            
            for i, feature in enumerate(hog_features):
                new_row[f'f{i+1}'] = feature
            
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)
            
            df.to_csv(self.csv_filename, index=False)
            
            print(f"Data saved: {filename}, Features: {len(hog_features)}")
            
        except Exception as e:
            raise Exception(f"Error saving to CSV: {str(e)}")
    
    def save_to_csv(self, filename, label):
        self.save_to_csv_with_features(filename, label, self.hog_features)
    
    def clear_current_data(self):
        try:
            self.label_originalImage.clear()
            self.label_grayscaleImage.clear()
            self.label_hogImage.clear()
            self.lineEdit_label.clear()
            
            self.original_image = None
            self.processed_image = None
            self.grayscale_image = None
            self.hog_features = None
            self.hog_image = None
            
            self.slider_contrast.setEnabled(False)
            self.slider_brightness.setEnabled(False)
            self.slider_blur.setEnabled(False)
            
        except AttributeError:
            pass

def main():
    app = QApplication(sys.argv)
    window = B3DatasetApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()