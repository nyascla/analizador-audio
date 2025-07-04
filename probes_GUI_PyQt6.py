import sys 
import os
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
                             QFileDialog, QMessageBox, QHBoxLayout, QGridLayout)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import librosa

# Parámetros globales
FS = 44100
DURATION = 10

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesamiento de Audio - Menú Principal")
        self.resize(1000, 700)  # Ventana más grande

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.info_label = QLabel("Selecciona una opción")
        main_layout.addWidget(self.info_label)

        # Botones para opciones
        self.btn1 = QPushButton("1. Grabar y aplicar efecto Darth Vader")
        self.btn1.clicked.connect(self.opcion1)
        main_layout.addWidget(self.btn1)

        self.btn2 = QPushButton("2. Procesar archivo WAV existente")
        self.btn2.clicked.connect(self.opcion2)
        main_layout.addWidget(self.btn2)

        self.btn3 = QPushButton("3. Detectar nota de piano (pendiente)")
        self.btn3.clicked.connect(self.opcion3)
        main_layout.addWidget(self.btn3)

        self.btnSalir = QPushButton("Salir")
        self.btnSalir.clicked.connect(self.close)
        main_layout.addWidget(self.btnSalir)

        # Área para mostrar 4 imágenes (espectros y espectrogramas antes y después)
        self.imagenes_layout = QGridLayout()
        main_layout.addLayout(self.imagenes_layout)

        # Labels para las imágenes
        self.label_espectro_antes = QLabel("Espectro Antes")
        self.label_espectro_antes.setFixedSize(460, 320)
        self.label_espectro_antes.setStyleSheet("border: 1px solid black;")
        self.imagenes_layout.addWidget(self.label_espectro_antes, 0, 0)

        self.label_espectrograma_antes = QLabel("Espectrograma Antes")
        self.label_espectrograma_antes.setFixedSize(460, 320)
        self.label_espectrograma_antes.setStyleSheet("border: 1px solid black;")
        self.imagenes_layout.addWidget(self.label_espectrograma_antes, 0, 1)

        self.label_espectro_despues = QLabel("Espectro Después")
        self.label_espectro_despues.setFixedSize(460, 320)
        self.label_espectro_despues.setStyleSheet("border: 1px solid black;")
        self.imagenes_layout.addWidget(self.label_espectro_despues, 1, 0)

        self.label_espectrograma_despues = QLabel("Espectrograma Después")
        self.label_espectrograma_despues.setFixedSize(460, 320)
        self.label_espectrograma_despues.setStyleSheet("border: 1px solid black;")
        self.imagenes_layout.addWidget(self.label_espectrograma_despues, 1, 1)

    def aplicar_efecto_vader(self, audio, fs):
        audio = librosa.effects.pitch_shift(audio, sr=fs, n_steps=-4)
        delay = int(0.03 * fs)
        echo = np.zeros_like(audio)
        echo[delay:] = 0.6 * audio[:-delay]
        audio = audio + echo
        b, a = butter(4, 3000/(fs/2), btype='low')
        audio = filtfilt(b, a, audio)
        audio = audio * 1.5
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        return audio

    def guardar_espectro(self, audio, fs, filename, etiqueta):
        N = len(audio)
        T = 1.0/fs
        yf = fft(audio)
        xf = fftfreq(N, T)[:N//2]
        plt.figure(figsize=(10, 4))
        plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
        plt.title(f'Espectro de frecuencia: {etiqueta}')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Amplitud')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"espectro_{filename}.png")
        plt.close()

    def guardar_espectrograma(self, audio, fs, filename, etiqueta):
        plt.figure(figsize=(10, 4))
        Pxx, freqs, bins, im = plt.specgram(audio, Fs=fs, NFFT=1024, noverlap=512, cmap='viridis')
        plt.title(f'Espectrograma: {etiqueta}')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Frecuencia (Hz)')
        cbar = plt.colorbar(im)
        cbar.set_label('Intensidad (dB)')
        plt.tight_layout()
        plt.savefig(f"espectrograma_{filename}.png")
        plt.close()

    def mostrar_imagen(self, ruta_imagen, label):
        if os.path.exists(ruta_imagen):
            pixmap = QPixmap(ruta_imagen)
            pixmap = pixmap.scaled(label.width(), label.height(), aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
            label.setPixmap(pixmap)
        else:
            label.setText("No se encontró la imagen")

    def opcion1(self):
        self.info_label.setText("Grabando audio...")
        QApplication.processEvents()
        audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()

        # Amplificar    
        amplificacion = 4.0
        audio = audio * amplificacion

        # Evitar saturación (normalizar si se pasa del rango)
        max_abs = np.max(np.abs(audio))
        if max_abs > 1:
            audio = audio / max_abs

        # Guardar WAV
        wavfile.write("voz_original.wav", FS, (audio * 32767).astype(np.int16))
        self.info_label.setText("Grabación guardada como 'voz_original.wav'")
        QApplication.processEvents()

        # Guardar y mostrar antes del efecto
        self.guardar_espectro(audio, FS, "voz_original_before", "Original (antes)")
        self.guardar_espectrograma(audio, FS, "voz_original_before", "Original (antes)")
        self.mostrar_imagen("espectro_voz_original_before.png", self.label_espectro_antes)
        self.mostrar_imagen("espectrograma_voz_original_before.png", self.label_espectrograma_antes)

        audio_vader = self.aplicar_efecto_vader(audio, FS)
        wavfile.write("voz_vader.wav", FS, (audio_vader * 32767).astype(np.int16))
        self.info_label.setText("Efecto Darth Vader aplicado y guardado.")
        QApplication.processEvents()

        # Guardar y mostrar después del efecto
        self.guardar_espectro(audio_vader, FS, "voz_vader_after", "Darth Vader (después)")
        self.guardar_espectrograma(audio_vader, FS, "voz_vader_after", "Darth Vader (después)")
        self.mostrar_imagen("espectro_voz_vader_after.png", self.label_espectro_despues)
        self.mostrar_imagen("espectrograma_voz_vader_after.png", self.label_espectrograma_despues)

        self.info_label.setText("Reproduciendo voz original...")
        QApplication.processEvents()
        sd.play(audio, FS)
        sd.wait()

        self.info_label.setText("Reproduciendo voz Darth Vader...")
        QApplication.processEvents()
        sd.play(audio_vader, FS)
        sd.wait()

        self.info_label.setText("Opción 1 finalizada. Elige otra opción.")

    def opcion2(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Selecciona un archivo WAV", "", "WAV Files (*.wav)")
        if not filename:
            self.info_label.setText("No se seleccionó ningún archivo.")
            return
        
        self.info_label.setText(f"Cargando archivo: {filename}")
        QApplication.processEvents()
        try:
            fs, audio = wavfile.read(filename)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32767
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483647
            elif audio.dtype == np.uint8:
                audio = (audio.astype(np.float32) - 128) / 128
            else:
                audio = audio.astype(np.float32)

            base_name = os.path.splitext(os.path.basename(filename))[0]

            # Guardar y mostrar antes del efecto
            self.guardar_espectro(audio, fs, f"{base_name}_before", "Original (antes)")
            self.guardar_espectrograma(audio, fs, f"{base_name}_before", "Original (antes)")
            self.mostrar_imagen(f"espectro_{base_name}_before.png", self.label_espectro_antes)
            self.mostrar_imagen(f"espectrograma_{base_name}_before.png", self.label_espectrograma_antes)

            audio_vader = self.aplicar_efecto_vader(audio, fs)
            wavfile.write(f"{base_name}_vader.wav", fs, (audio_vader * 32767).astype(np.int16))
            self.info_label.setText(f"Efecto Darth Vader aplicado y guardado en {base_name}_vader.wav")
            QApplication.processEvents()

            # Guardar y mostrar después del efecto
            self.guardar_espectro(audio_vader, fs, f"{base_name}_after", "Darth Vader (después)")
            self.guardar_espectrograma(audio_vader, fs, f"{base_name}_after", "Darth Vader (después)")
            self.mostrar_imagen(f"espectro_{base_name}_after.png", self.label_espectro_despues)
            self.mostrar_imagen(f"espectrograma_{base_name}_after.png", self.label_espectrograma_despues)

            self.info_label.setText("Opción 2 finalizada. Elige otra opción.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo procesar el archivo:\n{e}")
            self.info_label.setText("Error al procesar archivo.")

    def opcion3(self):
        QMessageBox.information(self, "Pendiente", "Funcionalidad de detección de nota pendiente de implementar.")
        self.info_label.setText("Opción 3 en desarrollo. Elige otra opción.")

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 
