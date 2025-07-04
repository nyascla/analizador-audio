import sounddevice as sd
import numpy as np

FS = 44100
DURACION = 3  # segundos

print("Grabando...")
audio = sd.rec(int(DURACION * FS), samplerate=FS, channels=1, dtype='float32')
sd.wait()
print("Reproduciendo...")
sd.play(audio, FS)
sd.wait()