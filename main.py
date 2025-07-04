import numpy as np                # procesar arrays numericos
import matplotlib.pyplot as plt   # mostrar graficamente los espectros, ...
from scipy.fft import fft, fftfreq  # para calcular la FFT
from scipy.io import wavfile
import sounddevice as sd        # reproducir el audio
import argparse                 # crear una interfaz por línea de comandos (menu en terminal)

# funcion que carga los datos binarios (.raw) y los convierte a un array de muestras
def leer_raw(filename, dtype=np.int16, channels=1):
    samples = np.fromfile(filename, dtype=dtype)
    # Como es estéreo, reestructuramos para que sounddevice lo interprete bien:
    # samples debe ser una matriz (num_muestras, num_canales)
    samples = samples.reshape(-1, channels)
    return samples

def leer_wav(filename):
    fs, samples = wavfile.read(filename)
    return samples, fs

# muestra graficamente el audio en el dominio del tiempo 
def mostrar_forma_onda(samples, fullpath):
    plt.plot(samples)
    plt.title('Forma de onda del audio .raw')
    plt.xlabel('Muestra')
    plt.ylabel('Amplitud')

    filename = fullpath.replace('\\', '/').split('/')[-1] 
    filename = filename.split('.')[0]
    plt.savefig(f"forma_onda_{filename}.png")

# imprime estadisticas de la señal  
def mostrar_estadisticas(samples):
    print("======================================")
    print("Estadísticas del audio:")
    print("Max:", samples.max())
    print("Min:", samples.min())
    rango = int(samples.max()) - int(samples.min())  
    print("Rango:", rango)
    print("RMS:", np.sqrt(np.mean(samples.astype(np.float64)**2)))
    print("======================================")

# uso de libreria sounddevice para reproducir audio
def reproducir_audio(samples, fs, channels):
    print("Reproduciendo audio...")
    sd.play(samples, fs)
    sd.wait()
    print("Reproducción finalizada.")

# aplica la FFT y muestra graficamente las frecuencias del audio ("foto")
def mostrar_espectro(samples, fs, fullpath):
    N = len(samples)
    T = 1.0 / fs
    yf = fft(samples)
    xf = fftfreq(N, T)[:N//2]
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.title('Espectro de frecuencia')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    filename = fullpath.replace('\\', '/').split('/')[-1] 
    filename = filename.split('.')[0]
    plt.savefig(f"espectro_{filename}.png")
  
# espectograma (contenido frec. a lo largo del tiempo) mediante plt.specgram  
def mostrar_espectrograma(samples, fs, fullpath):
    plt.figure(figsize=(10, 4))
    plt.specgram(samples, Fs=fs, NFFT=1024, noverlap=512, cmap='viridis', vmin=1e-10) 
    plt.title('Espectrograma')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Frecuencia (Hz)')
    filename = fullpath.replace('\\', '/').split('/')[-1] 
    filename = filename.split('.')[0]
    plt.colorbar(label='Intensidad (dB)')
    plt.savefig(f"espectrograma_{filename}.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analiza un fichero de audio RAW") # inicio el "parser" para leer los argumentos pasados por terminal por el usuario
    parser.add_argument("file", help="Ruta del fichero .raw")   # archivo de audio
    parser.add_argument("--dtype", default="int16", help="Tipo de dato (int16, int8, float32...)")   # tipo de datos de las muestras
    parser.add_argument("--fs", type=int, default=44100, help="Frecuencia de muestreo (Hz)")   # muestras por segundo que tiene el archivo de audio
    parser.add_argument("--channels", type=int, default=1, help="Número de canales")     # audio mono o estereo
    parser.add_argument("--action", choices=["waveform", "stats", "play", "spectrum", "spectrogram"], required=True,
                    help="Acción a realizar: 'waveform' (forma de onda), 'stats' (estadísticas), 'play' (reproducir), 'spectrum' (FFT), 'spectrogram' (espectrograma)")
    
    args = parser.parse_args()   # se recogen los argumentos que hemos pasado por terminal

    # Convertir dtype string a tipo numpy
    try:
        dtype = np.dtype(args.dtype)
    except TypeError:  # y verifica que el tipo de datos sea valido
        print(f"Tipo de dato inválido: {args.dtype}")
        return

    if args.file.lower().endswith(".wav"):
        samples, fs = leer_wav(args.file)
        # si stereo, coger solo canal 0 para análisis
        if samples.ndim > 1:
            samples = samples[:, 0]
    else:
        samples = leer_raw(args.file, dtype=dtype, channels=args.channels)
        fs = args.fs
        # Si hay más de un canal, separarlos
        if args.channels > 1:
            samples = samples.reshape(-1, args.channels)
            # Para análisis simplificamos y tomamos solo el primer canal
            samples = samples[:, 0]    # y se toma solo el canal 0 para analisis

    if args.action == "waveform":
        mostrar_forma_onda(samples, args.file)
    elif args.action == "stats":
        mostrar_estadisticas(samples)
    elif args.action == "play":
        reproducir_audio(samples, args.fs, args.channels)
    elif args.action == "spectrum":
        mostrar_espectro(samples, args.fs, args.file)
    elif args.action == "spectrogram":
        mostrar_espectrograma(samples, args.fs, args.file)

def test():
    file = "samples/p1.raw"
    channels = 2
    fs = 44100
    dtype = np.int16 
    samples = leer_raw(file, dtype=dtype, channels=channels)
    channel = samples[:, 0]
    mostrar_forma_onda(channel, file)
    mostrar_estadisticas(channel)
    mostrar_espectro(channel, fs, file)
    mostrar_espectrograma(channel, fs, file)
    reproducir_audio(channel, fs, channels)

    # Con un wav:
    # wav_file = "samples/audio.wav"
    # samples_wav, fs_wav = leer_wav(wav_file)
    # if samples_wav.ndim > 1:
    #     samples_wav = samples_wav[:, 0]
    # mostrar_forma_onda(samples_wav, wav_file)
    # mostrar_estadisticas(samples_wav)
    # mostrar_espectro(samples_wav, fs_wav, wav_file)
    # mostrar_espectrograma(samples_wav, fs_wav, wav_file)
    # reproducir_audio(samples_wav, fs_wav, 1)

if __name__ == "__main__":
    test()  