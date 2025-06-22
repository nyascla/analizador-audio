import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import sounddevice as sd
import argparse

def leer_raw(filename, dtype=np.int16, channels=1):
    samples = np.fromfile(filename, dtype=dtype)
    # Como es estéreo, reestructuramos para que sounddevice lo interprete bien:
    # samples debe ser una matriz (num_muestras, num_canales)
    samples = samples.reshape(-1, channels)
    return samples

def mostrar_forma_onda(samples, fullpath):
    plt.plot(samples)
    plt.title('Forma de onda del audio .raw')
    plt.xlabel('Muestra')
    plt.ylabel('Amplitud')

    filename = fullpath.split('/')[-1] 
    filename = filename.split('.')[0]
    plt.savefig(f"forma_onda_{filename}.png")

def mostrar_estadisticas(samples):
    print("======================================")
    print("Estadísticas del audio:")
    print("Max:", samples.max())
    print("Min:", samples.min())
    print("Rango:", samples.max() - samples.min())
    print("RMS:", np.sqrt(np.mean(samples**2)))
    print("======================================")


def reproducir_audio(samples, fs, channels):
    print("Reproduciendo audio...")
    sd.play(samples, fs)
    sd.wait()
    print("Reproducción finalizada.")

def mostrar_espectro(samples, fs, fullpath):
    N = len(samples)
    T = 1.0 / fs
    yf = fft(samples)
    xf = fftfreq(N, T)[:N//2]

    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.title('Espectro de frecuencia')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    filename = fullpath.split('/')[-1] 
    filename = filename.split('.')[0]
    plt.savefig(f"espectro_{filename}.png")

def main():
    parser = argparse.ArgumentParser(description="Analiza un fichero de audio RAW")
    parser.add_argument("file", help="Ruta del fichero .raw")
    parser.add_argument("--dtype", default="int16", help="Tipo de dato (int16, int8, float32...)")
    parser.add_argument("--fs", type=int, default=44100, help="Frecuencia de muestreo (Hz)")
    parser.add_argument("--channels", type=int, default=1, help="Número de canales")
    parser.add_argument("--action", choices=["waveform", "stats", "play", "spectrum"], required=True,
                        help="Acción a realizar: 'waveform' (forma de onda), 'stats' (estadísticas), 'play' (reproducir), 'spectrum' (espectro)")
    args = parser.parse_args()

    # Convertir dtype string a tipo numpy
    try:
        dtype = np.dtype(args.dtype)
    except TypeError:
        print(f"Tipo de dato inválido: {args.dtype}")
        return

    samples = leer_raw(args.file, dtype)

    # Si hay más de un canal, separarlos
    if args.channels > 1:
        samples = samples.reshape(-1, args.channels)
        # Para análisis simplificamos y tomamos solo el primer canal
        samples = samples[:, 0]

    if args.action == "waveform":
        mostrar_forma_onda(samples)
    elif args.action == "stats":
        mostrar_estadisticas(samples)
    elif args.action == "play":
        reproducir_audio(samples, args.fs, args.channels)
    elif args.action == "spectrum":
        mostrar_espectro(samples, args.fs)

def test():
    file = "samples/p1.raw"
    channels = 2
    fs = 44100
    dtype = np.int16 

    samples = leer_raw(file, dtype=dtype, channels=channels)
    mostrar_forma_onda(samples, file)
    mostrar_estadisticas(samples)
    mostrar_espectro(samples, fs, file)
    reproducir_audio(samples, 44100, 2)


if __name__ == "__main__":
    test()  