import sys 
import os
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
                             QFileDialog, QMessageBox, QGridLayout)
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

        self.btn3 = QPushButton("3. Detectar nota de piano")
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

    def limpiar_imagenes(self):
        """Limpia todas las etiquetas de imagen para ocultarlas."""
        self.label_espectro_antes.clear()
        self.label_espectrograma_antes.clear()
        self.label_espectro_despues.clear()
        self.label_espectrograma_despues.clear()

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

    def frecuencia_a_nota_librosa(self, f):
        if f <= 0:
            return "Sin nota"
        return librosa.hz_to_note(f)

    def detectar_nota(self, segmento, Fs):
        y = segmento.astype(np.float64)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=1200, sr=Fs)
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) == 0:
            return "Sin nota", 0
        f0_median = np.median(f0_valid)
        nota = self.frecuencia_a_nota_librosa(f0_median)
        return nota, f0_median

    def opcion1(self):
        self.limpiar_imagenes()  # Limpiar imágenes al iniciar opción 1
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
        self.limpiar_imagenes()  # Limpiar imágenes al iniciar opción 2
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
        self.limpiar_imagenes()  # Limpiar imágenes previas para opción 3
        filename, _ = QFileDialog.getOpenFileName(self, "Selecciona un archivo WAV para detectar notas", "", "WAV Files (*.wav)")
        if not filename:
            self.info_label.setText("No se seleccionó ningún archivo.")
            return

        self.info_label.setText(f"Detectando notas en: {os.path.basename(filename)}")
        QApplication.processEvents()

        try:
            Fs, data = wavfile.read(filename)

            if data.dtype == np.int16:
                audio = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                audio = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                audio = (data.astype(np.float32) - 128) / 128.0
            else:
                audio = data.astype(np.float32)

            if audio.ndim > 1:
                audio = audio[:, 0]

            sd.play(audio, Fs)
            sd.wait()

            frame_size = int(0.1 * Fs)
            step_size = int(0.05 * Fs)
            energies = []

            for i in range(0, len(audio) - frame_size, step_size):
                window = audio[i:i+frame_size]
                energies.append(np.sum(window**2))
            energies = np.array(energies)
            energies /= np.max(energies)

            threshold = 0.1
            active = energies > threshold
            bounds = np.diff(active.astype(int))
            inicios = np.where(bounds == 1)[0] * step_size
            finales = np.where(bounds == -1)[0] * step_size

            if len(finales) == 0 or (len(inicios) > 0 and inicios[0] > finales[0]):
                pass
            while len(finales) < len(inicios):
                finales = np.append(finales, len(audio))

            notas_previas = []
            umbral_tiempo = 0.3
            mensaje = ""
            contador = 0

            for ini, fin in zip(inicios, finales):
                segmento = audio[ini:fin]
                dur = (fin - ini) / Fs
                energia_segmento = np.sum(segmento**2)

                if dur < 0.15 and energia_segmento < 0.01:
                    continue
                elif dur < 0.1:
                    continue
                if len(segmento) < int(0.05 * Fs):
                    continue

                nota, freq = self.detectar_nota(segmento, Fs)
                if nota == "Sin nota":
                    continue

                if notas_previas and nota == notas_previas[-1][0] and (ini / Fs - notas_previas[-1][2]) < umbral_tiempo:
                    continue

                contador += 1
                notas_previas.append((nota, freq, fin / Fs))
                mensaje += f"Nota {contador}: {nota} — {freq:.1f} Hz — duración {dur:.2f}s\n"
                print(f"Nota {contador}: {nota} — {freq:.1f} Hz — duración {dur:.2f}s")

            if mensaje:
                self.info_label.setText("Notas detectadas:\n" + mensaje)
            else:
                self.info_label.setText("No se detectaron notas útiles.")

            # Mostrar gráfico energía + onsets detectados
            plt.figure(figsize=(10, 4))
            plt.plot(energies, label="Energía")
            plt.axhline(threshold, color='r', linestyle='--', label='Umbral')
            inicio_muestras = inicios // step_size
            plt.vlines(inicio_muestras, 0, 1, color='g', linestyle='--', label='Inicios detectados')
            plt.title("Energía normalizada por ventana (100 ms)")
            plt.xlabel("Ventana (cada 50 ms)")
            plt.ylabel("Energía (normalizada)")
            plt.legend()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al analizar el archivo:\n{e}")
            self.info_label.setText("Error al procesar archivo.")

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
