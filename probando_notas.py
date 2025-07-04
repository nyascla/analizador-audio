import numpy as np
import scipy.io.wavfile as wav
from scipy.fft import fft
from scipy.signal import find_peaks
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd  # Para reproducir audio

# Función que convierte frecuencia a nombre de nota con librosa
def frecuencia_a_nota_librosa(f):
    if f <= 0:
        return "Sin nota"
    return librosa.hz_to_note(f)

# Detección robusta de frecuencia fundamental
def detectar_fundamental(picos, frecs_valid):
    picos_ordenados = sorted(picos, key=lambda p: frecs_valid[p])  # ordenar por frecuencia ascendente
    for p in picos_ordenados:
        f_cand = frecs_valid[p]
        es_fund = True
        for mult in range(2, 6):  # armónicos
            if not any(abs(f_cand * mult - frecs_valid[q]) < 5 for q in picos):
                es_fund = False
                break
        if es_fund:
            return f_cand
    # Si no encuentra fundamental con armónicos, devuelve el pico más bajo
    return frecs_valid[picos_ordenados[0]]

# Detecta la nota dominante en un segmento usando librosa.pyin
def detectar_nota(segmento, Fs):
    # librosa.pyin necesita audio mono y float64
    y = segmento.astype(np.float64)

    # Extraer pitch fundamental con pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=1200, sr=Fs)

    # f0 es un array con la frecuencia estimada para cada frame de pyin, nan si no detecta nota
    # Queremos tomar la mediana de las frecuencias detectadas (ignorando nans)
    f0_valid = f0[~np.isnan(f0)]
    if len(f0_valid) == 0:
        return "Sin nota", 0

    f0_median = np.median(f0_valid)
    nota = frecuencia_a_nota_librosa(f0_median)
    return nota, f0_median

# Cargar archivo
Fs, data = wav.read('DO_MI_SOL_DO.wav')

# Comprobar tipo y normalizar si es entero
if data.dtype == np.int16:
    audio_float = data.astype(np.float32) / 32768.0  # Escala -1 a 1
elif data.dtype == np.int32:
    audio_float = data.astype(np.float32) / 2147483648.0
elif data.dtype == np.uint8:
    audio_float = (data.astype(np.float32) - 128) / 128.0
else:
    audio_float = data.astype(np.float32)

# Si es estéreo, coger sólo un canal
if audio_float.ndim > 1:
    audio_float = audio_float[:,0]

# Reproducimos el audio con sounddevice
print("Reproduciendo ...")
sd.play(audio_float, Fs)
sd.wait()

# ----------- Análisis de energía para segmentar notas ----------
frame_size = int(0.1 * Fs)  # Ventana de 100 ms
step_size = int(0.05 * Fs)  # Paso de 50 ms (solapamiento del 50%)
energies = []

# Calcular energía en cada ventana deslizante
for i in range(0, len(audio_float) - frame_size, step_size):
    window = audio_float[i:i+frame_size]
    energies.append(np.sum(window**2))  # Energía = suma de cuadrados
energies = np.array(energies)
energies /= np.max(energies)  # Normalizamos entre 0 y 1

# Umbral para detectar cuándo hay una nota
threshold = 0.1
active = energies > threshold  # True si la energía supera el umbral
bounds = np.diff(active.astype(int))  # Detecta subidas (1) y bajadas (-1)
inicios = np.where(bounds == 1)[0] * step_size
finales = np.where(bounds == -1)[0] * step_size

# Alinear inicios y finales por si faltan extremos
if len(finales) == 0 or (len(inicios) > 0 and inicios[0] > finales[0]):
    pass
while len(finales) < len(inicios):
    finales = np.append(finales, len(audio_float))

# ----------- Evitar repeticiones inmediatas de la misma nota ----------
notas_previas = []
umbral_tiempo = 0.3  # Segundos mínimo para considerar notas diferentes

print("\nNotas detectadas:")
contador = 0
for ini, fin in zip(inicios, finales):
    segmento = audio_float[ini:fin]
    dur = (fin - ini) / Fs
    energia_segmento = np.sum(segmento**2)

    if dur < 0.15 and energia_segmento < 0.01:
        print(f"Segmento descartado por corto y débil: duración {dur:.2f}s")
        continue
    elif dur < 0.1:
        print(f"Segmento descartado por muy corto: duración {dur:.2f}s")
        continue
    if len(segmento) < int(0.05 * Fs):
        continue  # Ignorar segmentos muy cortos (<50ms)

    nota, freq = detectar_nota(segmento, Fs)
    if nota == "Sin nota":
        continue  # Ignorar segmentos sin nota identificable

    # Evitar repeticiones muy próximas
    if notas_previas and nota == notas_previas[-1][0] and (ini / Fs - notas_previas[-1][2]) < umbral_tiempo:
        continue

    contador += 1
    notas_previas.append((nota, freq, fin / Fs))
    print(f"Nota {contador}: {nota} — {freq:.1f} Hz — duración {dur:.2f}s")

# ----------- Gráfico de energía con onsets detectados ----------
plt.figure(figsize=(10, 4))
plt.plot(energies, label="Energía")
plt.axhline(threshold, color='r', linestyle='--', label='Umbral')

# Marcar inicios detectados para segmentación (inicio de notas)
inicio_muestras = inicios // step_size
plt.vlines(inicio_muestras, 0, 1, color='g', linestyle='--', label='Inicios detectados')

plt.title("Energía normalizada por ventana (100 ms)")
plt.xlabel("Ventana (cada 50 ms)")
plt.ylabel("Energía (normalizada)")
plt.legend()
plt.tight_layout()
plt.show()
