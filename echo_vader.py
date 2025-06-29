import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import struct
from scipy.fft import fft
from vader_efect import vader_effect  # Importa la función del efecto

# ------------------------------
# Configuración
# ------------------------------
BLOQUE_MUESTRAS = 4096
FS = 44100
DURACION = 10  # segundos
CANAL = 1
BUFFER_SIZE = FS  # 1 segundo de audio

# ------------------------------
# Inicializar audio
# ------------------------------
p = pyaudio.PyAudio()
flujo_audio = p.open(format=pyaudio.paInt16,
                     channels=CANAL,
                     rate=FS,
                     input=True,
                     frames_per_buffer=BLOQUE_MUESTRAS)
flujo_salida = p.open(format=pyaudio.paInt16,
                      channels=CANAL,
                      rate=FS,
                      output=True,
                      frames_per_buffer=BLOQUE_MUESTRAS)

# ------------------------------
# Inicializar gráficos
# ------------------------------
plt.ion()
figura, (ax1, ax2) = plt.subplots(2, 1)
x = np.arange(0, BLOQUE_MUESTRAS)
x_fft = np.linspace(0, FS / 2, BLOQUE_MUESTRAS // 2)
linea_onda, = ax1.plot(x, np.random.rand(BLOQUE_MUESTRAS))
linea_fft, = ax2.plot(x_fft, np.random.rand(BLOQUE_MUESTRAS // 2))
ax1.set_ylim(-32768, 32767)
ax2.set_ylim(0, 1)
ax2.set_xlim(20, FS / 2)

# ------------------------------
# Captura, acumula, aplica efecto y reproduce
# ------------------------------
print("Grabando y aplicando efecto Darth Vader en tiempo real (bufferizado)...")
buffer_audio = np.array([], dtype=np.float32)

for _ in range(0, int(FS / BLOQUE_MUESTRAS * DURACION)):
    datos = flujo_audio.read(BLOQUE_MUESTRAS, exception_on_overflow=False)
    datos_int = np.array(struct.unpack(str(BLOQUE_MUESTRAS) + 'h', datos)).astype(np.float32)
    buffer_audio = np.concatenate((buffer_audio, datos_int))

    # Cuando el buffer alcanza el tamaño deseado, procesa y reproduce
    if len(buffer_audio) >= BUFFER_SIZE:
        bloque_vader = vader_effect(buffer_audio, FS)

        # Normaliza y convierte a int16 para reproducir
        bloque_vader = bloque_vader / (np.max(np.abs(bloque_vader)) + 1e-6)
        bloque_vader_int16 = np.int16(bloque_vader * 32767)

        # Reproduce el audio procesado en bloques
        for i in range(0, len(bloque_vader_int16), BLOQUE_MUESTRAS):
            sub_bloque = bloque_vader_int16[i:i+BLOQUE_MUESTRAS]
            if len(sub_bloque) < BLOQUE_MUESTRAS:
                sub_bloque = np.pad(sub_bloque, (0, BLOQUE_MUESTRAS - len(sub_bloque)), mode='constant')
            flujo_salida.write(sub_bloque.tobytes())

            # FFT para mostrar espectro
            magnitud_fft = np.abs(fft(sub_bloque))[:BLOQUE_MUESTRAS // 2]
            magnitud_fft = magnitud_fft / np.max(magnitud_fft + 1e-6)
            frecuencia_dominante = x_fft[np.argmax(magnitud_fft)]

            linea_onda.set_ydata(sub_bloque)
            linea_fft.set_ydata(magnitud_fft)
            ax2.set_title(f"Frecuencia dominante: {frecuencia_dominante:.1f} Hz")
            figura.canvas.draw()
            figura.canvas.flush_events()

        buffer_audio = np.array([], dtype=np.float32)  # Vacía el buffer

