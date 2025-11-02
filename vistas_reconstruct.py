import numpy as np
import cv2
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
flags = cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR

outputdir = 'dice'
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
real_file = 'dices4k_real.exr'
imag_file = 'dices4k_imag.exr'
z_dados = [0.0018, 0.0025, 0.0030]
z_aviao = [0.01, 0.0115, 0.0122]
z = z_dados[0]

pitch = 0.4e-06 #1e-06
lbdr = 6.4e-07
lbdg = 5.32e-07
lbdb = 4.73e-07

rows = 4
cols = 4

def propASM(u1, pitch, lbd, z):
    M, N = u1.shape
    dx = pitch
    dy = pitch
    fx = np.linspace(-1.0 / (2.0 * dx), 1.0 / (2.0 * dx), N, endpoint=False)
    fy = np.linspace(-1.0 / (2.0 * dy), 1.0 / (2.0 * dy), M, endpoint=False)
    FX, FY = np.meshgrid(fx, fy)

    w = np.sqrt(np.maximum(0.0, (1.0 / lbd) ** 2 - FX**2 - FY**2))
    H = np.exp(-2j * np.pi * w * z)

    U1 = np.fft.fftshift(np.fft.fft2(u1))
    U2 = U1 * H
    u2 = np.fft.ifft2(np.fft.ifftshift(U2))

    return u2

def view(holo, r, c, view_size_r, view_size_c):
    holo_view = np.zeros_like(holo, dtype=holo.dtype) # matriz vazia

    # calcula a posição inicial com base no índice (r, c)
    start_row = r * view_size_r
    start_col = c * view_size_c

    # calcula a posição final com base na inicial e tamanho da vista
    end_row = start_row + view_size_r
    end_col = start_col + view_size_c

    # copia a porção do holograma correspondente às posições calculadas
    # o resto mantem-se a zeros
    holo_view[start_row:end_row, start_col:end_col, :] = holo[start_row:end_row, start_col:end_col, :]

    return holo_view

# versão para vistas sobrepostas
def view_sobreposto(holo, r, c, view_size, step_size):
    M, N, _ = holo.shape
    holo_view = np.zeros_like(holo, dtype=holo.dtype)

    # calcula a posição inicial com base no step_size
    start_row = r * step_size
    start_col = c * step_size

    # garantir que a vista está dentro dos limites da matriz
    # e que não há coordenadas negativas
    if start_row + view_size > M:
        start_row = M - view_size
    if start_col + view_size > N:
        start_col = N - view_size
    start_row = max(0, start_row)
    start_col = max(0, start_col)

    # igual ao anterior
    end_row = start_row + view_size
    end_col = start_col + view_size

    holo_view[start_row:end_row, start_col:end_col, :] = holo[start_row:end_row, start_col:end_col, :]
    
    return holo_view

real  = cv2.imread(real_file, flags).astype(np.float32)
imag  = cv2.imread(imag_file, flags).astype(np.float32)

holo_base = real + 1j * imag
holo = np.dstack([holo_base, holo_base, holo_base])

M, N, _ = holo.shape
view_size_r = M // rows
view_size_c = N // cols 
grid_indices = [(r, c) for r in range(rows) for c in range(cols)]

# para vistas sobrepostas
# view_size = 2048
# step_size = 682

for i, (r, c) in enumerate(grid_indices, 1): # 1 a 16
    print(f" Vista {i}: ({r}, {c}) - Tamanho: {view_size_r}x{view_size_c}px")

    # criar vista do holograma para a posição (r, c)
    holo_view= view(holo, r, c, view_size_r, view_size_c)
    # holo_view= view_sobreposto(holo, r, c, view_size, step_size)

    u2R_R = propASM(holo_view[:, :, 2], pitch, lbdr, z)
    u2G_R = propASM(holo_view[:, :, 1], pitch, lbdg, z)
    u2B_R = propASM(holo_view[:, :, 0], pitch, lbdb, z)
    Irgb = np.dstack([
        np.abs(u2R_R),
        np.abs(u2G_R),
        np.abs(u2B_R)
    ])
    Irgb = np.log(1 + Irgb / Irgb.max()) if Irgb.max() > 0 else Irgb

    # guardar a imagem
    img_uint8_rgb = (Irgb * 255).astype(np.uint8)
    img_uint8_bgr = cv2.cvtColor(img_uint8_rgb, cv2.COLOR_RGB2BGR)
    filename_individual = os.path.join(outputdir, f'frame{r}{c}.png')
    cv2.imwrite(filename_individual, img_uint8_bgr)
