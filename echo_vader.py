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
# Activa el modo interactivo de matplotlib para actualizar gráficos en tiempo real
plt.ion()
# Crea una figura con dos subgráficas (una arriba y otra abajo)
figura, (ax1, ax2) = plt.subplots(2, 1)
# Vector de tiempo para la gráfica de la onda
x = np.arange(0, BLOQUE_MUESTRAS)
# Vector de frecuencias para la gráfica de espectro
x_fft = np.linspace(0, FS / 2, BLOQUE_MUESTRAS // 2)
# Inicializa la línea de la onda de audio (tiempo)
linea_onda, = ax1.plot(x, np.random.rand(BLOQUE_MUESTRAS))
# Inicializa la línea del espectro de frecuencias (FFT)
linea_fft, = ax2.plot(x_fft, np.random.rand(BLOQUE_MUESTRAS // 2))
# Fija los límites del eje Y para la onda (rango de int16)
ax1.set_ylim(-32768, 32767)
# Fija los límites del eje Y para el espectro (magnitud normalizada)
ax2.set_ylim(0, 1)
# Fija los límites del eje X para el espectro (de 20 Hz a la frecuencia de Nyquist)
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
        # Aplica el efecto Darth Vader al buffer de audio acumulado
        bloque_vader = vader_effect(buffer_audio, FS)

        # Normaliza el audio procesado para evitar saturación y lo convierte a int16 para reproducir
        bloque_vader = bloque_vader / (np.max(np.abs(bloque_vader)) + 1e-6)
        bloque_vader_int16 = np.int16(bloque_vader * 32767)

        # Reproduce el audio procesado en bloques del mismo tamaño que la entrada
        for i in range(0, len(bloque_vader_int16), BLOQUE_MUESTRAS):
            sub_bloque = bloque_vader_int16[i:i+BLOQUE_MUESTRAS]
            # Si el último sub_bloque es más pequeño, lo rellena con ceros para evitar errores
            if len(sub_bloque) < BLOQUE_MUESTRAS:
                sub_bloque = np.pad(sub_bloque, (0, BLOQUE_MUESTRAS - len(sub_bloque)), mode='constant')
            flujo_salida.write(sub_bloque.tobytes())

            # Calcula la FFT del sub_bloque para mostrar el espectro de frecuencias
            magnitud_fft = np.abs(fft(sub_bloque))[:BLOQUE_MUESTRAS // 2]
            magnitud_fft = magnitud_fft / np.max(magnitud_fft + 1e-6)
            frecuencia_dominante = x_fft[np.argmax(magnitud_fft)]

            # Actualiza las gráficas en tiempo real con la onda y el espectro
            linea_onda.set_ydata(sub_bloque)
            linea_fft.set_ydata(magnitud_fft)
            ax2.set_title(f"Frecuencia dominante: {frecuencia_dominante:.1f} Hz")
            figura.canvas.draw()
            figura.canvas.flush_events()

        # Vacía el buffer para acumular el siguiente segundo de audio
        buffer_audio = np.array([], dtype=np.float32)

