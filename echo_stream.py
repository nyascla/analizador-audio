import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import struct
from scipy.fft import fft

# ------------------------------
# Configuración
# ------------------------------
BLOQUE_MUESTRAS = 4096
FS = 44100
DURACION = 10  # segundos
CANAL = 1

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
# Captura y reproducción directa
# ------------------------------
print("Monitorizando audio en tiempo real...")
for _ in range(0, int(FS / BLOQUE_MUESTRAS * DURACION)):
    datos = flujo_audio.read(BLOQUE_MUESTRAS, exception_on_overflow=False)
    datos_int = np.array(struct.unpack(str(BLOQUE_MUESTRAS) + 'h', datos))

    # Reproduce directamente lo que entra
    flujo_salida.write(datos)

    # FFT para mostrar espectro
    magnitud_fft = np.abs(fft(datos_int))[:BLOQUE_MUESTRAS // 2]
    magnitud_fft = magnitud_fft / np.max(magnitud_fft + 1e-6)
    vector_frecuencias = np.linspace(0, FS / 2, BLOQUE_MUESTRAS // 2)
    frecuencia_dominante = vector_frecuencias[np.argmax(magnitud_fft)]

    linea_onda.set_ydata(datos_int)
    linea_fft.set_ydata(magnitud_fft)
    ax2.set_title(f"Frecuencia dominante: {frecuencia_dominante:.1f} Hz")
    figura.canvas.draw()
    figura.canvas.flush_events()

