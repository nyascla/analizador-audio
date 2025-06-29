import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import struct
from scipy.signal import butter, lfilter    # Para aplicar filtros digitales (pasa bajos)
from scipy.fft import fft

# ------------------------------
# Configuración de parámetros de audio y grabación
# ------------------------------
BLOQUE_MUESTRAS = 4096   # Muestras procesadas por bloque (≈46 ms de audio)
FS = 44100     # Frecuencia de muestreo (Hz)
DURACION = 10  # Segundos de grabación
CANAL = 1     # Audio mono

# ------------------------------
# Inicializar PyAudio y abrir flujos de entrada/salida
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
# Inicializar gráficos en tiempo real
# ------------------------------
plt.ion() # Modo interactivo para actualizar gráficos en tiempo real
figura, (ax1, ax2) = plt.subplots(2, 1) 
x = np.arange(0, BLOQUE_MUESTRAS)
x_fft = np.linspace(0, FS / 2, BLOQUE_MUESTRAS // 2)
linea_onda, = ax1.plot(x, np.random.rand(BLOQUE_MUESTRAS))  # Onda de audio en el tiempo
linea_fft, = ax2.plot(x_fft, np.random.rand(BLOQUE_MUESTRAS // 2))  # Espectro de frecuencias
ax1.set_ylim(-32768, 32767)
ax2.set_ylim(0, 1)
ax2.set_xlim(20, FS / 2)

# ------------------------------
# Bucle principal: captura, reproducción y visualización
# ------------------------------
print("Monitorizando audio en tiempo real...")
for _ in range(0, int(FS / BLOQUE_MUESTRAS * DURACION)):
    datos = flujo_audio.read(BLOQUE_MUESTRAS, exception_on_overflow=False)
    datos_int = np.array(struct.unpack(str(BLOQUE_MUESTRAS) + 'h', datos))

    # Reproduce directamente lo que entra (eco en tiempo real)
    flujo_salida.write(datos)

    # Calcula la FFT para mostrar el espectro de frecuencias
    magnitud_fft = np.abs(fft(datos_int))[:BLOQUE_MUESTRAS // 2]
    magnitud_fft = magnitud_fft / np.max(magnitud_fft + 1e-6)
    vector_frecuencias = np.linspace(0, FS / 2, BLOQUE_MUESTRAS // 2)
    frecuencia_dominante = vector_frecuencias[np.argmax(magnitud_fft)]

    # Actualiza las gráficas en tiempo real
    linea_onda.set_ydata(datos_int)
    linea_fft.set_ydata(magnitud_fft)
    ax2.set_title(f"Frecuencia dominante: {frecuencia_dominante:.1f} Hz")
    figura.canvas.draw()
    figura.canvas.flush_events()

