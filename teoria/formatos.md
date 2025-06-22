## 1. `.raw` – Audio en bruto (sin cabecera)

- **Descripción:** Contiene únicamente las muestras de audio digitalizadas. No incluye ningún tipo de metadatos ni cabecera.
- **Características:**
  - No tiene información sobre frecuencia de muestreo, canales ni profundidad de bits.
  - Generalmente en formato PCM (modulación por impulsos codificados).
  - Necesita que el reproductor conozca externamente sus parámetros.
- **Ventajas:**
  - Muy simple y rápido de generar.
  - Útil para procesamiento de bajo nivel o aprendizaje técnico.
- **Desventajas:**
  - No es autocontenible: necesita saber los parámetros para reproducirse.
  - No es compatible directamente con la mayoría de reproductores.

---

## 2. `.wav` – Audio sin compresión con cabecera

- **Descripción:** Es un contenedor que almacena audio en formato PCM (sin compresión), pero incluye una **cabecera** con metadatos.
- **Características:**
  - Incluye información como frecuencia de muestreo, número de canales, formato, duración, etc.
  - Usa típicamente audio PCM firmado de 8/16/24/32 bits.
- **Ventajas:**
  - Reproducible por casi cualquier software sin configuración adicional.
  - Buena fidelidad de audio (calidad sin pérdida).
- **Desventajas:**
  - Archivos grandes (sin compresión).
  - No ideal para transmisión o almacenamiento en línea.

---

## 3. `.mp3` – Audio comprimido con pérdida

- **Descripción:** Formato comprimido que utiliza codificación con pérdida para reducir el tamaño del archivo, eliminando partes menos perceptibles del sonido.
- **Características:**
  - Basado en el estándar MPEG-1 Layer III.
  - Permite ajustar el bitrate (calidad frente a tamaño).
- **Ventajas:**
  - Muy ligero y eficiente para distribución.
  - Alta compatibilidad con dispositivos y reproductores.
- **Desventajas:**
  - Pérdida de calidad respecto al original (especialmente a bajos bitrates).
  - No ideal para edición o archivo profesional.

---

## 📊 Comparativa rápida

| Característica         | `.raw`                  | `.wav`                        | `.mp3`                         |
|------------------------|--------------------------|-------------------------------|--------------------------------|
| **Compresión**         | ❌ No                    | ❌ No                          | ✅ Sí (con pérdida)            |
| **Cabecera/Metadatos** | ❌ No                    | ✅ Sí                          | ✅ Sí                          |
| **Calidad**            | ⭐⭐⭐⭐ (sin pérdida)      | ⭐⭐⭐⭐ (sin pérdida)            | ⭐⭐–⭐⭐⭐ (según bitrate)        |
| **Tamaño del archivo** | 📂 Grande                | 📂 Grande                      | 📁 Pequeño                     |
| **Facilidad de uso**   | ⚠️ Requiere configuración | ✅ Plug-and-play               | ✅ Muy compatible              |
