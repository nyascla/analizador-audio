import numpy as np
import matplotlib.pyplot as plt
import pyaudio
from scipy.fft import fft
from scipy import signal

# ------------------------------
# Clase de procesamiento de audio
# ------------------------------
class AudioProcessing:
    def __init__(self, input_audio, sample_freq):
        self.sample_freq = sample_freq
        self.audio_data = input_audio.astype(np.float32)

    def set_audio_speed(self, speed_factor):
        idx = np.round(np.arange(0, len(self.audio_data), speed_factor)).astype(int)
        idx = idx[idx < len(self.audio_data)]
        self.audio_data = self.audio_data[idx]

    def set_echo(self, delay):
        output_audio = np.copy(self.audio_data)
        output_delay = int(delay * self.sample_freq)
        if output_delay == 0:
            return
        output_audio[output_delay:] += 0.5 * self.audio_data[:-output_delay]
        self.audio_data = output_audio

    def set_lowpass(self, cutoff, order=4):
        nyquist = self.sample_freq / 2.0
        b, a = signal.butter(order, cutoff / nyquist, btype='lowpass')
        self.audio_data = signal.filtfilt(b, a, self.audio_data)

    def set_volume(self, level):
        self.audio_data *= level

    def vader_effect(self):
        self.set_audio_speed(0.8)
        self.set_echo(0.02)
        self.set_lowpass(2500)
        self.set_volume(1.2)
        max_val = np.max(np.abs(self.audio_data))
        if max_val > 0:
            self.audio_data /= max_val

# ------------------------------
# Configuración
# ------------------------------
BLOQUE_MUESTRAS = 4096
FS = 44100
DURACION = 10  # segundos
BUFFER_SIZE = FS  # 1 segundo de audio
ENERGY_THRESHOLD = 50  # Ajusta según tu micro

# ------------------------------
# Inicializar audio
# ------------------------------
p = pyaudio.PyAudio()
flujo_audio = p.open(format=pyaudio.paInt16,
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
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
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
# Captura, aplica efecto y reproduce
# ------------------------------
print("🎤 Grabando y aplicando efecto Darth Vader en tiempo real (bufferizado)...")
buffer_audio = np.array([], dtype=np.float32)

try:
    for _ in range(int(FS / BLOQUE_MUESTRAS * DURACION)):
        datos = flujo_audio.read(BLOQUE_MUESTRAS, exception_on_overflow=False)
        datos_float = np.frombuffer(datos, dtype=np.int16).astype(np.float32)
        buffer_audio = np.concatenate((buffer_audio, datos_float))

        if len(buffer_audio) >= BUFFER_SIZE:
            energy = np.mean(np.abs(buffer_audio))
            if energy > ENERGY_THRESHOLD:
                processor = AudioProcessing(buffer_audio, FS)
                processor.vader_effect()
                bloque_vader = processor.audio_data
            else:
                bloque_vader = np.zeros_like(buffer_audio)

            # CORRECCIÓN AQUÍ:
            bloque_vader_int16 = np.int16(np.clip(bloque_vader * 32767, -32768, 32767))

            for i in range(0, len(bloque_vader_int16), BLOQUE_MUESTRAS):
                sub_bloque = bloque_vader_int16[i:i+BLOQUE_MUESTRAS]
                if len(sub_bloque) < BLOQUE_MUESTRAS:
                    sub_bloque = np.pad(sub_bloque, (0, BLOQUE_MUESTRAS - len(sub_bloque)), mode='constant')
                flujo_salida.write(sub_bloque.tobytes())

                # FFT para mostrar espectro
                magnitud_fft = np.abs(fft(sub_bloque))[:BLOQUE_MUESTRAS // 2]
                magnitud_fft /= (np.max(magnitud_fft) + 1e-6)
                linea_onda.set_ydata(sub_bloque)
                linea_fft.set_ydata(magnitud_fft)
                plt.pause(0.001)

            buffer_audio = np.array([], dtype=np.float32)

except KeyboardInterrupt:
    print("\n⏹️ Grabación interrumpida por el usuario.")

finally:
    plt.ioff()
    plt.show()
    flujo_audio.stop_stream()
    flujo_audio.close()
    flujo_salida.stop_stream()
    flujo_salida.close()
    p.terminate()
    print("✅ Proceso terminado.")

