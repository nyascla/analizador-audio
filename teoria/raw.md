# Cómo se estructura un fichero de audio RAW (PCM signed 16-bit little endian)

- **Formato RAW** significa que el archivo contiene únicamente datos de audio en bruto, sin cabecera ni metadatos.
- En PCM signed 16-bit little endian:
  - Cada muestra de audio ocupa **2 bytes** (16 bits).
  - El valor es un entero con signo, que puede ir desde **-32768** hasta **32767**.
  - **Little endian** significa que el byte menos significativo (LSB) viene primero, seguido del byte más significativo (MSB).
- La señal se representa como una secuencia de valores enteros, cada uno representando la amplitud del audio en un instante.
- Para audio **mono**, las muestras están en secuencia directa.
- Para audio **estéreo**, las muestras se alternan por canal, por ejemplo:
  - Byte 0-1: muestra canal 1
  - Byte 2-3: muestra canal 2
  - Byte 4-5: muestra canal 1
  - Byte 6-7: muestra canal 2
  - Y así sucesivamente.
- Para interpretar los datos:
  - Se leen bloques de 2 bytes.
  - Cada bloque se convierte a un entero con signo.
  - Estos valores se pueden usar para procesar, visualizar o reproducir el audio.