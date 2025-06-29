import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from scipy import signal


# Clase principal con soporte para arrays directos
class AudioProcessing:
    def __init__(self, input_audio, sample_freq, from_array=True):
        self.sample_freq = sample_freq
        self.audio_data = self.convert_to_mono_audio(input_audio) if from_array else input_audio

    def set_audio_pitch(self, n, window_size=2**13, h=2**11):
        factor = 2 ** (1.0 * n / 12.0)
        self._set_stretch(1.0 / factor, window_size, h)
        self.audio_data = self.audio_data[window_size:]
        self.set_audio_speed(factor)

    def _set_stretch(self, factor, window_size, h):
        phase = np.zeros(window_size)
        hanning_window = np.hanning(window_size)
        result = np.zeros(int(len(self.audio_data) / factor + window_size))

        for i in np.arange(0, len(self.audio_data) - (window_size + h), h * factor):
            a1 = self.audio_data[int(i): int(i + window_size)]
            a2 = self.audio_data[int(i + h): int(i + window_size + h)]

            s1 = np.fft.fft(hanning_window * a1)
            s2 = np.fft.fft(hanning_window * a2)

            phase = (phase + np.angle(s2 / s1)) % 2 * np.pi
            a2_rephased = np.fft.ifft(np.abs(s2) * np.exp(1j * phase))
            i2 = int(i / factor)
            result[i2: i2 + window_size] += hanning_window * a2_rephased.real

        result = ((2 ** (16 - 4)) * result / result.max())
        self.audio_data = result.astype('int16')

    def set_audio_speed(self, speed_factor):
        sound_index = np.round(np.arange(0, len(self.audio_data), speed_factor))
        self.audio_data = self.audio_data[sound_index[sound_index < len(self.audio_data)].astype(int)]

    def set_echo(self, delay):
        output_audio = np.zeros(len(self.audio_data))
        output_delay = int(delay * self.sample_freq)
        for count in range(output_delay, len(self.audio_data)):
            output_audio[count] = self.audio_data[count] + 0.6 * self.audio_data[count - output_delay]
        self.audio_data = output_audio

    def set_lowpass(self, cutoff_low, order=5):
        nyquist = self.sample_freq / 2.0
        cutoff = cutoff_low / nyquist
        b, a = signal.butter(order, cutoff, btype='lowpass')
        self.audio_data = signal.filtfilt(b, a, self.audio_data)

    def set_volume(self, level):
        self.audio_data = self.audio_data * level

    def apply_darth_vader_effect(self):
        #self.set_audio_pitch(-4)
        #self.set_lowpass(500)
        #self.set_echo(0.03)
        #self.set_volume(1.5)

        self.set_audio_speed(.8)
        self.set_echo(0.02)
        self.set_lowpass(2500)
    
    @staticmethod
    def convert_to_mono_audio(input_audio):
        input_audio = input_audio.squeeze()
        return input_audio.astype(np.float32)


def vader_effect(audio_array, sample_rate):
    processor = AudioProcessing(audio_array, sample_rate, False)
    processor.apply_darth_vader_effect()
    return processor.audio_data.astype(np.float32)

if __name__ == "__main__":
    # 🎤 Grabación, efecto y reproducción
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
    print("✅ Listo.")
