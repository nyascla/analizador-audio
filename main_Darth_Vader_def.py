import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import librosa

# Parámetros
FS = 44100
DURATION = 10

def aplicar_efecto_vader(audio, fs):
    audio = librosa.effects.pitch_shift(audio, sr=fs, n_steps=-4)
    delay = int(0.03 * fs)
    echo = np.zeros_like(audio)
    echo[delay:] = 0.6 * audio[:-delay]
    audio = audio + echo
    b, a = butter(4, 3000/(fs/2), btype='low')
    audio = filtfilt(b, a, audio)
    audio = audio * 1.5
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    return audio

def guardar_espectro(audio, fs, filename, etiqueta):
    N = len(audio)
    T = 1.0/fs
    yf = fft(audio)
    xf = fftfreq(N, T)[:N//2]
    plt.figure(figsize=(10, 4))
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.title(f'Espectro de frecuencia: {etiqueta}')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"espectro_{filename}.png")
    plt.close()

def guardar_espectrograma(audio, fs, filename, etiqueta):
    plt.figure(figsize=(10, 4))
    Pxx, freqs, bins, im = plt.specgram(audio, Fs=fs, NFFT=1024, noverlap=512, cmap='viridis')
    plt.title(f'Espectrograma: {etiqueta}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Frecuencia (Hz)')
    cbar = plt.colorbar(im)
    cbar.set_label('Intensidad (dB)')
    plt.tight_layout()
    plt.savefig(f"espectrograma_{filename}.png")
    plt.close()

print("Grabando...")
audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
sd.wait()
audio = audio.flatten()

wavfile.write("voz_original.wav", FS, (audio * 32767).astype(np.int16))
print("Grabación guardada como 'voz_original.wav'")

audio_vader = aplicar_efecto_vader(audio, FS)
wavfile.write("voz_vader.wav", FS, (audio_vader * 32767).astype(np.int16))
print("Efecto Darth Vader aplicado. Guardado como 'voz_vader.wav'")

guardar_espectro(audio, FS, "voz_original", "Original")
guardar_espectrograma(audio, FS, "voz_original", "Original")

guardar_espectro(audio_vader, FS, "voz_vader", "Darth Vader")
guardar_espectrograma(audio_vader, FS, "voz_vader", "Darth Vader")

print("Espectros y espectrogramas guardados como imágenes.")

print("Reproduciendo voz original...")
sd.play(audio, FS)
sd.wait()

print("Reproduciendo voz Darth Vader...")
sd.play(audio_vader, FS)
sd.wait()

print("Proceso completado.")