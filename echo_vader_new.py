import numpy as np
import matplotlib.pyplot as plt
import pyaudio
from scipy.fft import fft
from scipy import signal

# ------------------------------
# Clase de procesamiento de audio: define métodos para alterar el audio
# ------------------------------
class AudioProcessing:
    def __init__(self, input_audio, sample_freq):
        self.sample_freq = sample_freq                    # Frecuencia de muestreo (Hz)
        self.audio_data = input_audio.astype(np.float32)  # Conversión a float para procesado

    def set_audio_speed(self, speed_factor):
        # Cambia la velocidad del audio. <1 lo ralentiza (más grave), >1 acelera (más agudo)
        idx = np.round(np.arange(0, len(self.audio_data), speed_factor)).astype(int)
        idx = idx[idx < len(self.audio_data)]
        self.audio_data = self.audio_data[idx]

    def set_echo(self, delay):
        # Añade un eco simple (retardo multiplicado y sumado)
        output_audio = np.copy(self.audio_data)
        output_delay = int(delay * self.sample_freq)
        if output_delay == 0:
            return
        output_audio[output_delay:] += 0.5 * self.audio_data[:-output_delay]
        self.audio_data = output_audio

    def set_lowpass(self, cutoff, order=4):
        # Aplica un filtro paso bajo (suaviza y atenúa agudos)
        nyquist = self.sample_freq / 2.0
        b, a = signal.butter(order, cutoff / nyquist, btype='lowpass')
        self.audio_data = signal.filtfilt(b, a, self.audio_data)

    def set_volume(self, level):
        # Ajusta el volumen multiplicando la señal
        self.audio_data *= level

    def vader_effect(self):
        # Combinación de efectos para simular voz tipo Darth Vader
        self.set_audio_speed(0.75)  # Hace la voz más grave
        self.set_echo(0.04)         # Añade eco corto
        self.set_lowpass(350)       # Filtra agudos (voz más opaca)
        self.set_volume(1.3)        # Aumenta volumen
        max_val = np.max(np.abs(self.audio_data))
        if max_val > 0:
            self.audio_data /= max_val  # Normaliza amplitud para evitar distorsión

# ------------------------------
# Configuración
# ------------------------------
BLOQUE_MUESTRAS = 8192       # Muestras por bloque: tamaño del chunk de audio
FS = 44100                   # Frecuencia de muestreo (Hz)
DURACION = 10                # Duración total de la ejecución (segundos)
BUFFER_SIZE = FS / 2         # Acumular 0.5 s de audio antes de procesar
ENERGY_THRESHOLD = 5         # Umbral mínimo de energía para aplicar efecto (evita ruido ambiente)

# ------------------------------
# Inicializar flujo de entrada/salida de audio
# ------------------------------
p = pyaudio.PyAudio()

# Flujo de entrada (micrófono)
flujo_audio = p.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=FS,
                     input=True,
                     frames_per_buffer=BLOQUE_MUESTRAS)

# Flujo de salida (altavoces)
flujo_salida = p.open(format=pyaudio.paInt16,
                      channels=1,
                      rate=FS,
                      output=True,
                      frames_per_buffer=BLOQUE_MUESTRAS)

# ------------------------------
# Inicializar visualización con matplotlib
# ------------------------------
plt.ion()  # Activar modo interactivo
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))  # Crear dos gráficas (onda + espectro)
x = np.arange(BLOQUE_MUESTRAS)
x_fft = np.linspace(0, FS / 2, BLOQUE_MUESTRAS // 2)  # Eje de frecuencias para FFT

# Inicializar líneas de gráfico (vacías al principio)
linea_onda, = ax1.plot(x, np.zeros(BLOQUE_MUESTRAS))
linea_fft, = ax2.plot(x_fft, np.zeros(BLOQUE_MUESTRAS // 2))

# Configurar límites de las gráficas
ax1.set_ylim(-32768, 32767)         # Rango de valores PCM 16 bits
ax2.set_ylim(0, 1)
ax2.set_xlim(20, FS / 2)
ax2.set_xscale('log')               # Escala logarítmica para mejor visualización de espectro
plt.tight_layout()

# ------------------------------
# Bucle principal: capturar, procesar y reproducir
# ------------------------------
print(" Grabando y aplicando efecto Darth Vader en tiempo real (bufferizado)...")

buffer_audio = np.array([], dtype=np.float32)  # Inicializar buffer de audio

try:
    for _ in range(int(FS / BLOQUE_MUESTRAS * DURACION)):
        # Leer bloque de datos desde el micrófono
        datos = flujo_audio.read(BLOQUE_MUESTRAS, exception_on_overflow=False)
        datos_float = np.frombuffer(datos, dtype=np.int16).astype(np.float32)

        # Acumular en buffer
        buffer_audio = np.concatenate((buffer_audio, datos_float))

        # Cuando el buffer tiene 0.5s de audio, procesar
        if len(buffer_audio) >= BUFFER_SIZE:
            # Calcular energía media del buffer para ignorar silencio
            energy = np.mean(np.abs(buffer_audio))
            if energy > ENERGY_THRESHOLD:
                # Aplicar efecto Vader si supera umbral de energía
                processor = AudioProcessing(buffer_audio, FS)
                processor.vader_effect()
                bloque_vader = processor.audio_data
            else:
                # Si es silencio o ruido bajo, mandar bloque nulo
                bloque_vader = np.zeros_like(buffer_audio)

            # Normalizar y convertir a int16 para reproducir
            bloque_vader_int16 = np.int16(np.clip(bloque_vader * 32767, -32768, 32767))

            # Dividir en subbloques para enviar a salida de audio
            for i in range(0, len(bloque_vader_int16), BLOQUE_MUESTRAS):
                sub_bloque = bloque_vader_int16[i:i + BLOQUE_MUESTRAS]
                # Rellenar si el último bloque es incompleto
                if len(sub_bloque) < BLOQUE_MUESTRAS:
                    sub_bloque = np.pad(sub_bloque, (0, BLOQUE_MUESTRAS - len(sub_bloque)), mode='constant')

                # Reproducir bloque por altavoces
                flujo_salida.write(sub_bloque.tobytes())

                # Calcular y mostrar FFT del bloque procesado
                magnitud_fft = np.abs(fft(sub_bloque))[:BLOQUE_MUESTRAS // 2]
                magnitud_fft /= (np.max(magnitud_fft) + 1e-6)  # Normaliza para gráfico

                # Actualizar las gráficas
                linea_onda.set_ydata(sub_bloque)
                linea_fft.set_ydata(magnitud_fft)
                plt.pause(0.001)

            # Vaciar buffer tras procesar
            buffer_audio = np.array([], dtype=np.float32)

except KeyboardInterrupt:
    # Permitir parar con Ctrl+C
    print("\n Grabación interrumpida por el usuario.")

finally:
    # Cerrar flujos y finalizar
    plt.ioff()
    plt.show()
    flujo_audio.stop_stream()
    flujo_audio.close()
    flujo_salida.stop_stream()
    flujo_salida.close()
    p.terminate()
    print("Proceso terminado.")

