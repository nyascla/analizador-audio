import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def guardar_espectro(audio, fs, filename, etiqueta):
    N = len(audio)
    T = 1.0/fs
    yf = fft(audio)
    xf = fftfreq(N, T)[:N//2]
    magnitud = 2.0/N * np.abs(yf[:N//2])
    magnitud_db = 20 * np.log10(magnitud + 1e-10)  # dB, evitar log(0)

    plt.figure(figsize=(10, 4))
    plt.plot(xf, magnitud_db)
    plt.title(f'Espectro de frecuencia: {etiqueta}')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud (dB)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"espectro_{filename}.png")
    plt.close()