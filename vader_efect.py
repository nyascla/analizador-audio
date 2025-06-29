import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from scipy import signal


# Clase principal con soporte para arrays directos
class AudioProcessing:
    def __init__(self, input_audio, sample_freq, from_array=True):
        # Inicializa el objeto con la frecuencia de muestreo y el audio (convertido a mono si es necesario)
        self.sample_freq = sample_freq # guarda fs como atributo del objeto para usarla en otros métodos
        self.audio_data = self.convert_to_mono_audio(input_audio) if from_array else input_audio

    def set_audio_pitch(self, n, window_size=2**13, h=2**11):
        # Cambia el tono del audio (pitch shifting) usando estiramiento y cambio de velocidad
        factor = 2 ** (1.0 * n / 12.0)
        self._set_stretch(1.0 / factor, window_size, h)
        self.audio_data = self.audio_data[window_size:]
        self.set_audio_speed(factor)

    def _set_stretch(self, factor, window_size, h):
        # Realiza el estiramiento de tiempo usando la transformada de Fourier y ventana de Hann
        # 'factor' es cuánto se estira o comprime el audio (mayor a 1: más lento, menor a 1: más rápido)
        # 'window_size' es el tamaño de la ventana de análisis (en muestras)
        # 'h' es el salto entre ventanas (overlap)
        phase = np.zeros(window_size)  # Vector para acumular la fase entre ventanas
        hanning_window = np.hanning(window_size)  # Ventana de Hann para suavizar los bordes de cada bloque
        result = np.zeros(int(len(self.audio_data) / factor + window_size))  # Array de salida para el audio estirado

        # Itera sobre el audio en bloques solapados
        for i in np.arange(0, len(self.audio_data) - (window_size + h), h * factor):
            a1 = self.audio_data[int(i): int(i + window_size)]  # Primer bloque de audio
            a2 = self.audio_data[int(i + h): int(i + window_size + h)]  # Segundo bloque desplazado

            s1 = np.fft.fft(hanning_window * a1)  # FFT del primer bloque
            s2 = np.fft.fft(hanning_window * a2)  # FFT del segundo bloque

            # Calcula la diferencia de fase entre bloques y la acumula
            phase = (phase + np.angle(s2 / s1)) % (2 * np.pi)
            # Reconstruye el segundo bloque con la nueva fase
            a2_rephased = np.fft.ifft(np.abs(s2) * np.exp(1j * phase))
            i2 = int(i / factor)  # Índice de salida correspondiente
            # Suma el bloque reconstruido al resultado, aplicando la ventana de Hann
            result[i2: i2 + window_size] += hanning_window * a2_rephased.real

        # Normaliza el resultado y lo convierte a int16
        result = ((2 ** (16 - 4)) * result / result.max())
        self.audio_data = result.astype('int16')

    def set_audio_speed(self, speed_factor):
        # Cambia la velocidad de reproducción del audio
        sound_index = np.round(np.arange(0, len(self.audio_data), speed_factor))
        self.audio_data = self.audio_data[sound_index[sound_index < len(self.audio_data)].astype(int)]

    def set_echo(self, delay):
        # Añade eco al audio con un retardo y un factor de mezcla
        output_audio = np.zeros(len(self.audio_data))
        output_delay = int(delay * self.sample_freq)
        for count in range(output_delay, len(self.audio_data)):
            output_audio[count] = self.audio_data[count] + 0.6 * self.audio_data[count - output_delay]
        self.audio_data = output_audio

    def set_lowpass(self, cutoff_low, order=5):
        # Aplica un filtro pasa bajos para eliminar frecuencias altas
        nyquist = self.sample_freq / 2.0
        cutoff = cutoff_low / nyquist
        b, a = signal.butter(order, cutoff, btype='lowpass')
        self.audio_data = signal.filtfilt(b, a, self.audio_data)

    def set_volume(self, level):
        # Ajusta el volumen del audio
        self.audio_data = self.audio_data * level

    def apply_darth_vader_effect(self):
        # Aplica una combinación de efectos para simular la voz de Darth Vader
        # Puedes activar/desactivar efectos comentando las líneas
        #self.set_audio_pitch(-4)
        #self.set_lowpass(500)
        #self.set_echo(0.03)
        #self.set_volume(1.5)

        self.set_audio_speed(.8)
        self.set_echo(0.02)
        self.set_lowpass(2500)
    
    @staticmethod
    def convert_to_mono_audio(input_audio):
        # Convierte el audio a mono y tipo float32
        input_audio = input_audio.squeeze()
        return input_audio.astype(np.float32)


# Función para aplicar el efecto Darth Vader a un array de audio

def vader_effect(audio_array, sample_rate):
    processor = AudioProcessing(audio_array, sample_rate, False)
    processor.apply_darth_vader_effect()
    return processor.audio_data.astype(np.float32)

if __name__ == "__main__":
    # Grabación, efecto y reproducción
    DURATION = 4  # segundos
    FS = 44100    # frecuencia de muestreo

    print("Grabando...")
    audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()

    print("Aplicando efecto Darth Vader...")
    vader_audio = vader_effect(audio, FS)

    print("Reproduciendo...")
    sd.play(vader_audio, FS)
    sd.wait()
    print("Listo.")
