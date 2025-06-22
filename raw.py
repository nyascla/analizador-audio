def leer_muestras_en_posicion(filepath, offset_bytes, num_bytes=16):
    """
    Lee num_bytes del archivo empezando en offset_bytes.
    Devuelve los datos crudos y las muestras interpretadas.
    """
    with open(filepath, "rb") as f:
        f.seek(offset_bytes)
        data = f.read(num_bytes)

    muestras = []
    for i in range(0, len(data), 2):
        par_bytes = data[i:i+2]
        if len(par_bytes) < 2:
            break  # Si queda byte suelto, se ignora
        valor = int.from_bytes(par_bytes, byteorder='little', signed=True)
        muestras.append(valor)

    return data, muestras

def main():
    filepath = "samples/p1.raw"
    num_bytes = 16  # cantidad de bytes a leer en cada bloque

    # Abrimos para saber el tamaño total del archivo
    import os
    filesize = os.path.getsize(filepath)
    print(f"Tamaño del archivo: {filesize} bytes\n")

    # Inicio: offset 0
    print("=== Muestras del inicio ===")
    data_inicio, muestras_inicio = leer_muestras_en_posicion(filepath, 0, num_bytes)
    print(f"Bytes hex: {' '.join(f'{b:02x}' for b in data_inicio)}")
    print(f"Muestras: {muestras_inicio}\n")

    # Medio: offset en la mitad del archivo (ajustado para que sea múltiplo de 2)
    offset_medio = (filesize // 2) - ((filesize // 2) % 2)
    print("=== Muestras del medio ===")
    data_medio, muestras_medio = leer_muestras_en_posicion(filepath, offset_medio, num_bytes)
    print(f"Bytes hex: {' '.join(f'{b:02x}' for b in data_medio)}")
    print(f"Muestras: {muestras_medio}\n")

    # Final: offset para leer desde el final (ajustado para que sea múltiplo de 2)
    offset_final = filesize - num_bytes
    if offset_final % 2 != 0:
        offset_final -= 1
    print("=== Muestras del final ===")
    data_final, muestras_final = leer_muestras_en_posicion(filepath, offset_final, num_bytes)
    print(f"Bytes hex: {' '.join(f'{b:02x}' for b in data_final)}")
    print(f"Muestras: {muestras_final}")

if __name__ == "__main__":
    main()
