import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

FS = 44100
DURATION = 5

print("Grabando...")
audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
sd.wait()
print("Grabación finalizada.")

# Amplificar y normalizar
amplificado = audio * 4
max_val = np.max(np.abs(amplificado))
if max_val > 1:
    amplificado = amplificado / max_val

write("test.wav", FS, (amplificado * 32767).astype(np.int16))
print("Archivo guardado como test.wav")