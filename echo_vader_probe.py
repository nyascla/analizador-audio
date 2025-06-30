import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import struct
from scipy.signal import butter, filtfilt
from scipy.fft import fft
import librosa

# ------------------------------
# Clase de procesamiento de audio
# ------------------------------
class AudioProcessing:
    def __init__(self, input_audio, sample_freq):
        self.sample_freq = sample_freq
        self.audio_data = input_audio.astype(np.float32)

    def bajar_tono(self, semitonos=-3):
        # Bajar el tono sin afectar la duración usando librosa
        self.audio_data = librosa.effects.pitch_shift(self.audio_data, self.sample_freq, n_steps=semitonos)

    def eco(self, retardo_segundos=0.03, ganancia=0.6):
        # Aplica un eco simple tipo comb-filter
        retardo_muestras = int(retardo_segundos * self.sample_freq)
        if retardo_muestras == 0:
            return
        salida = np.copy(self.audio_data)
        salida[retardo_muestras:] += ganancia * self.audio_data[:-retardo_muestras]
        self.audio_data = salida

    def filtro_pasabajos(self, frecuencia_corte=3000, orden=4):
        # Quita frecuencias agudas para hacer la voz más grave
        nyquist = self.sample_freq / 2.0
        b, a = butter(orden, frecuencia_corte / nyquist, btype='lowpass')
        self.audio_data = filtfilt(b, a, self.audio_data)

    def volumen(self, factor=1.5):
        # Aumenta la amplitud de la señal
        self.audio_data *= factor

    def normalizar(self):
        # Normaliza para evitar saturación
        max_val = np.max(np.abs(self.audio_data)) + 1e-6
        self.audio_data /= max_val

    def aplicar_efecto_vader(self):
        # Aplica todos los efectos en orden para simular la voz de Darth Vader
        self.bajar_tono(-3)            # Baja 3 semitonos
        self.eco(0.03, 0.6)            # Añade eco corto
        self.filtro_pasabajos(3000)    # Elimina agudos
        self.volumen(1.5)              # Aumenta volumen
        self.normalizar()              # Normaliza al final

# ------------------------------
# Configuración
# ------------------------------
BLOQUE_MUESTRAS = 4096
FS = 44100
DURACION = 10   # segundos de grabación
BUFFER_SIZE = FS  # Procesa 1 segundo a la vez
UMBRAL_ENERGIA = 30  # Umbral de energía para procesar voz

# ------------------------------
# Inicializar audio
# ------------------------------
p = pyaudio.PyAudio()
flujo_entrada = p.open(format=pyaudio.paInt16,
                       channels=1,
                       rate=FS,
                       input=True,
                       frames_per_buffer=BLOQUE_MUESTRAS)
flujo_salida = p.open(format=pyaudio.paInt16,
                      channels=1,
                      rate=FS,
                      output=True,
                      frames_per_buffer=BLOQUE_MUESTRAS)

# ------------------------------
# Inicializar gráficos
# ------------------------------
plt.ion()
figura, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
x = np.arange(BLOQUE_MUESTRAS)
x_fft = np.linspace(0, FS / 2, BLOQUE_MUESTRAS // 2)
linea_onda, = ax1.plot(x, np.zeros(BLOQUE_MUESTRAS))
linea_fft, = ax2.plot(x_fft, np.zeros(BLOQUE_MUESTRAS // 2))
ax1.set_ylim(-32768, 32767)
ax2.set_ylim(0, 1)
ax2.set_xlim(20, FS / 2)
ax2.set_xscale('log')
plt.tight_layout()

# ------------------------------
# Bucle principal: capturar, aplicar efecto y reproducir
# ------------------------------
print(" Grabando y aplicando efecto Darth Vader en tiempo real...")

buffer_audio = np.array([], dtype=np.float32)

try:
    for _ in range(int(FS / BLOQUE_MUESTRAS * DURACION)):
        datos = flujo_entrada.read(BLOQUE_MUESTRAS, exception_on_overflow=False)
        datos_float = np.frombuffer(datos, dtype=np.int16).astype(np.float32)
        buffer_audio = np.concatenate((buffer_audio, datos_float))

        if len(buffer_audio) >= BUFFER_SIZE:
            energia = np.mean(np.abs(buffer_audio))
            if energia > UMBRAL_ENERGIA:
                procesador = AudioProcessing(buffer_audio, FS)
                procesador.aplicar_efecto_vader()
                bloque_vader = procesador.audio_data
            else:
                bloque_vader = np.zeros_like(buffer_audio)

            bloque_vader_int16 = np.int16(np.clip(bloque_vader * 32767, -32768, 32767))

            # Reproduce en bloques
            for i in range(0, len(bloque_vader_int16), BLOQUE_MUESTRAS):
                sub_bloque = bloque_vader_int16[i:i+BLOQUE_MUESTRAS]
                if len(sub_bloque) < BLOQUE_MUESTRAS:
                    sub_bloque = np.pad(sub_bloque, (0, BLOQUE_MUESTRAS - len(sub_bloque)), mode='constant')
                flujo_salida.write(sub_bloque.tobytes())

                # Actualiza gráficos
                magnitud_fft = np.abs(fft(sub_bloque))[:BLOQUE_MUESTRAS // 2]
                magnitud_fft /= (np.max(magnitud_fft) + 1e-6)
                linea_onda.set_ydata(sub_bloque)
                linea_fft.set_ydata(magnitud_fft)
                plt.pause(0.001)

            buffer_audio = np.array([], dtype=np.float32)

except KeyboardInterrupt:
    print("\n Grabación interrumpida por el usuario.")

finally:
    plt.ioff()
    plt.show()
    flujo_entrada.stop_stream()
    flujo_entrada.close()
    flujo_salida.stop_stream()
    flujo_salida.close()
    p.terminate()
    print(" Proceso terminado.")