import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import struct
from scipy.signal import butter, lfilter
from scipy.fft import fft
import soundfile as sf

# ------------------------------
# Configuración
# ------------------------------i
BLOQUE_MUESTRAS = 2048
FS = 44100
DURACION = 10  # segundos
CANAL = 1
ARCHIVO_SALIDA = "darth_vader_output.wav"

# ------------------------------
# Efecto Darth Vader
# ------------------------------
def efecto_vader(audio, fs):
    audio = audio.astype(np.float32)
    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs
    else:
        audio = audio

    # Pitch shift simple: repetir muestras para bajar el tono (no introduce tanto ruido como resample)
    pitch_factor = 0.7  # 0.5 sería una octava, 0.7 es más natural
    indices = (np.arange(0, len(audio)) * pitch_factor).astype(int)
    indices = np.clip(indices, 0, len(audio) - 1)
    pitch_bajo = audio[indices]
    if len(pitch_bajo) < len(audio):
        pitch_bajo = np.pad(pitch_bajo, (0, len(audio) - len(pitch_bajo)), mode='constant')
    else:
        pitch_bajo = pitch_bajo[:len(audio)]

    # Filtro pasa bajos suave para quitar ruido agudo
    b, a = butter(3, 800 / (fs / 2), btype='low')
    filtrado = lfilter(b, a, pitch_bajo)

    max_vad = np.max(np.abs(filtrado))
    if max_vad > 0:
        filtrado = filtrado / max_vad
    return filtrado.astype(np.float32)

# ------------------------------
# Inicializar audio
# ------------------------------
p = pyaudio.PyAudio()
flujo_audio = p.open(format=pyaudio.paInt16,
                     channels=CANAL,
                     rate=FS,
                     input=True,
                     frames_per_buffer=BLOQUE_MUESTRAS)
# Nuevo: flujo de salida para reproducir el audio procesado
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
# Captura y procesamiento
# ------------------------------
print("Grabando con efecto Darth Vader en tiempo real...")
bloques_audio = []
for _ in range(0, int(FS / BLOQUE_MUESTRAS * DURACION)):
    datos = flujo_audio.read(BLOQUE_MUESTRAS)
    datos_int = np.array(struct.unpack(str(BLOQUE_MUESTRAS) + 'h', datos))
    bloques_audio.append(datos_int)

    procesado = efecto_vader(datos_int, FS)
    # Normalizar y convertir a int16 para reproducir
    procesado_int16 = np.int16(procesado / np.max(np.abs(procesado)) * 32767)
    flujo_salida.write(procesado_int16.tobytes())

    magnitud_fft = np.abs(fft(datos_int))[:BLOQUE_MUESTRAS // 2]
    magnitud_fft = magnitud_fft / np.max(magnitud_fft + 1e-6)
    vector_frecuencias = np.linspace(0, FS / 2, BLOQUE_MUESTRAS // 2)
    frecuencia_dominante = vector_frecuencias[np.argmax(magnitud_fft)]

    linea_onda.set_ydata(procesado * 32767)  # Escala para que se vea en el eje
    linea_fft.set_ydata(magnitud_fft)
    ax2.set_title(f"Frecuencia dominante: {frecuencia_dominante:.1f} Hz")
    figura.canvas.draw()
    figura.canvas.flush_events()

print("Grabación finalizada")

# ------------------------------
# Guardar en archivo
# ------------------------------
print("Guardando archivo con efecto...")
audio_final = np.concatenate([efecto_vader(bloque, FS) for bloque in bloques_audio])
audio_final = audio_final / np.max(np.abs(audio_final))
sf.write(ARCHIVO_SALIDA, audio_final, FS)
print(f"Archivo guardado como {ARCHIVO_SALIDA}")

flujo_audio.stop_stream()
flujo_audio.close()
flujo_salida.stop_stream()
flujo_salida.close()
p.terminate()

plt.ioff()
plt.show()
