import sounddevice as sd
import numpy as np
import scipy.signal

DURATION = 2
FS = 44100

def vader_effect(audio, fs):
    audio = audio.flatten()

    # --- 1. Filtro pasa bajos más agresivo (grave, oscuro)
    b, a = scipy.signal.butter(6, 300 / (fs / 2), btype='low')
    filtered = scipy.signal.lfilter(b, a, audio)

    # --- 2. Añadir distorsión controlada
    distorted = np.tanh(filtered * 5)

    # --- 3. Vibrato (frecuencia modulada tipo Vader)
    t = np.arange(len(distorted)) / fs
    vibrato = 5 * np.sin(2 * np.pi * 6 * t)  # vibrato lento y grave
    modulated = np.interp(
        np.clip(np.arange(len(distorted)) + vibrato, 0, len(distorted) - 1),
        np.arange(len(distorted)),
        distorted
    )

    # --- 4. Eco metálico (tipo comb filter)
    delay_samples = int(0.03 * fs)
    comb = np.zeros_like(modulated)
    comb[delay_samples:] = modulated[:-delay_samples] * 0.6
    with_echo = modulated + comb

    # --- 5. Reverberación básica (cola de ecos)
    decay = 0.5
    reverb = np.zeros_like(with_echo)
    for i in range(1, 6):
        d = delay_samples * i
        if d < len(reverb):
            reverb[d:] += with_echo[:-d] * (decay ** i)

    vaderized = with_echo + reverb

    # --- 6. Normalizar
    vaderized = vaderized / np.max(np.abs(vaderized))
    return vaderized.reshape(-1, 1)

print("🎤 Grabando...")
audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
sd.wait()

print("🧪 Aplicando efecto Darth Vader...")
vader_audio = vader_effect(audio, FS)

print("🔊 Reproduciendo...")
sd.play(vader_audio, FS)
sd.wait()
print("✅ Listo.")
