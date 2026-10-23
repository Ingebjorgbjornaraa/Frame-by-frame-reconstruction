"""
POLYNOMIAL REGRESSION WITH SVD ON IMAGES
Follows the note «Regression with SVD» in «Kodene mine» (my codes).
Finds the polynomial of a given degree that fits the points as well as possible, without going exactly through them.

THE THEORY (from the note)
    A polynomial for each R, G and B for each pixel(wh)  =>  MANY polynomials,
    but they share ONE A-matrix.

        p(t) = x_1 + x_2·t + x_3·t² + ... + x_{degree+1}·t^degree

    A : the Vandermonde matrix, (number of photos) × (degree + 1)
          row i = (1, t_i, t_i², ..., t_i^degree) — only numbers about time, no images
    b : the photos, (number of photos) × (w·h·3)
          row i = photo i, rolled out to one long row of R, G, B values
    x : the coefficients, (degree + 1) × (w·h·3)
          row k = x_{k+1}, the coefficient in front of t^k, one number for every pixel and colour

    degree ≥ number of photos - 1  ->  interpolation (we do NOT want that)
    degree < number of photos - 1  ->  overdetermined system  ->  solved with SVD

    The pseudoinverse A† (the Moore–Penrose inverse):
        1. AᵀA
        2. eigenvalues λ and eigenvectors v_k of AᵀA   ->  V
        3. σ_k = √λ_k                                  ->  Σ, Σ⁻¹
        4. u_k = (1/σ_k) · A · v_k                     ->  U
        5. A† = V Σ⁻¹ Uᵀ

    x = A† b

    x is calculated ONE PHOTO AT A TIME, so we never need the whole b in memory:
        x = (column 0 of A†)·(row 0 of b) + (column 1 of A†)·(row 1 of b) + ...
    Row i of b is just photo i, so we read one photo, add the contribution, and take the next.

    Now we have x, and can generate any frame we like, for any t we like:
        a(t) = (1, t, t², ..., t^degree)       («a new row of A»)
        frame(t) = a(t) · x

HOW I USE IT («how I use it»)
    - the script uses the images in the image folder that FROM_IMAGE, TO_IMAGE and USE_EVERY choose
    - photo no. i (among the chosen ones) is at t = i  (t = 0, 1, 2, ..., n)
    - for every pixel(wh) we have R, G and B -> three separate polynomials
    - A and A† are calculated ONE time and used for all pixels and colours
    - frames are made both at the real t's and between them (STEP)
    - the photos are read one by one, so it works fine with thousands of frames
"""

import os
import numpy as np
from PIL import Image, ImageOps


# =====================================================================
# SETTINGS — the only thing you need to change
# =====================================================================

IMAGES = "eriksamtale"   # the name of the folder with the images: "testbilder", "eriksamtale", ...

# Which images in the folder are used (number = the place in the folder, sorted by file name, first = 0)
FROM_IMAGE = 0      # first image that is included
TO_IMAGE = None     # last image that is included. None = all the way to the end of the folder
USE_EVERY = 1       # 1 = all the images, 3 = every 3rd, 30 = every 30th, ...
                         # (the script looks upwards from «code/regression» until it finds it)

DEGREE = 20        # 1 = straight line, 2 = parabola, 3 = S-shape, ...
                  # must be smaller than number of photos - 1 (otherwise interpolation)

STEP = 1          # distance between the frames in t. 1 = one frame per photo (same length
                  # as the original), 0.5 = one extra frame between every photo, ...

MAX_SIZE = 1920         # longest side in pixels. Eriksamtale is already 1920 × 1080.
                        # The test images (4608 × 3456) are scaled down to 1920 × 1440.
                        # None = full size.

SCALE_T = True    # See the explanation in part 1. Gives the SAME frames, just more exact
                  # calculation on the computer. Means almost nothing for degree 1–3.


# Find the folders (does not need to be changed)
FOLDER = os.path.dirname(os.path.abspath(__file__))   # the folder this script is in
IMAGE_FOLDER = None
place = FOLDER
for _ in range(4):                                    # FOLDER, then one, two and three levels up
    if os.path.isdir(os.path.join(place, IMAGES)):
        IMAGE_FOLDER = os.path.join(place, IMAGES)
        break
    place = os.path.dirname(place)
if IMAGE_FOLDER is None:
    raise SystemExit(f"Did not find the folder «{IMAGES}» near {FOLDER}.")
OUT_FOLDER = os.path.join(FOLDER, f"result_{IMAGES}_degree{DEGREE}")


# =====================================================================
# 1. THE THEORY — straight from the note
# =====================================================================

def time(image_number, n, scale):
    """
    Which t a photo is at. Without scaling: t = image_number (0, 1, ..., n),
    exactly as in the note.

    With scaling: t goes from -1 (first photo) to 1 (last photo) instead.
    It gives EXACTLY the same curves and frames (the x-numbers just become different),
    but the computer rounds off, and big t (like 25^8) makes one σ very small.
    Then 1/σ in Σ⁻¹ becomes huge and amplifies rounding errors. With t between -1 and 1
    that does not happen.
    """
    if not scale:
        return image_number
    return (image_number - n / 2) / (n / 2)


def make_A(t_values, degree):
    """A: one row per photo, row i = (1, t_i, t_i², ..., t_i^degree)
    = the numbers in front of x_1, x_2, ... in the equation for photo i."""
    return np.array([[t ** p for p in range(degree + 1)] for t in t_values], dtype=float)


def a(t, degree):
    """a(t) = (1, t, t², ..., t^degree) — a new row of A, for a t we choose ourselves"""
    return np.array([t ** p for p in range(degree + 1)], dtype=float)


def pseudoinverse(A, show=False):
    """A† = V Σ⁻¹ Uᵀ, step by step as in the note"""

    # 1. AᵀA
    ATA = A.T @ A

    # 2. eigenvalues λ and eigenvectors v_k of AᵀA (biggest first)
    λ, V = np.linalg.eigh(ATA)
    order = np.argsort(λ)[::-1]
    λ = λ[order]
    V = V[:, order]

    # 3. σ_k = √λ_k, Σ and Σ⁻¹
    σ = np.sqrt(λ)
    Σ = np.diag(σ)
    Σ_inv = np.diag(1 / σ)

    # 4. u_k = (1/σ_k) · A · v_k
    U = np.zeros((A.shape[0], len(σ)))
    for k in range(len(σ)):
        U[:, k] = (1 / σ[k]) * A @ V[:, k]

    # 5. A† = V Σ⁻¹ Uᵀ
    A_plus = V @ Σ_inv @ U.T

    if show:
        print("AᵀA =\n", ATA)
        print("λ =", λ, "  σ =", σ)
        print("V =\n", V)
        print("Σ =\n", Σ)
        print("Σ⁻¹ =\n", Σ_inv)
        print("U =\n", U)
        print("A† · 6 =\n", A_plus * 6)
    return A_plus


def frame(t, degree, x):
    """frame(t) = a(t) · x  — one value per pixel(wh) and colour"""
    return a(t, degree) @ x


# =====================================================================
# 2. CHECK AGAINST THE EXAMPLE IN THE NOTE
#    t = 0, 1, 2, degree 1, two pixels
#    The sign of a column in V and U can be flipped (both are equally correct),
#    but A†, x and frame(1.5) always become the same.
# =====================================================================

np.set_printoptions(precision=3, suppress=True)

A_ex = make_A([0, 1, 2], degree=1)
#                pixel 1,1           pixel 1,2
#               R    G    B        R    G    B
b_ex = np.array([[10,  40,  90,     200, 100,  50],     # t = 0
                 [20,  35,  88,     195, 102,  55],     # t = 1
                 [15,  38,  91,     190, 105,  52]],    # t = 2
                dtype=float)

print("The example from the note:")
print("A =\n", A_ex)
A_plus_ex = pseudoinverse(A_ex, show=True)       # A† · 6 = [[5, 2, -1], [-3, 0, 3]]
x_ex = A_plus_ex @ b_ex
print("x = A† b =\n", x_ex)                       # [[12.5 38.7 89.2 200  99.8 51.3]
                                                  #  [ 2.5 -1.0  0.5  -5   2.5  1.0]]
print("frame(1.5) =", frame(1.5, 1, x_ex))        # [16.25 37.2 89.9 192.5 103.6 52.8]

# The same x, calculated one photo at a time (as part 5 does with the real images):
#   x = (column 0 of A†)·(row 0 of b) + (column 1 of A†)·(row 1 of b) + (column 2 of A†)·(row 2 of b)
x_one_at_a_time = np.zeros_like(x_ex)
for i in range(3):
    x_one_at_a_time += A_plus_ex[:, [i]] @ b_ex[[i], :]
print("x calculated one photo at a time =\n", x_one_at_a_time)   # exactly the same
print()


# =====================================================================
# 3. FIND ALL THE IMAGES IN THE FOLDER
#    The images are sorted by file name, so the order in the folder becomes the order in time.
#    We do not read them in yet — we only find the names and the size.
# =====================================================================

ALL_IMAGE_FILES = [f for f in sorted(os.listdir(IMAGE_FOLDER))
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
end = None if TO_IMAGE is None else TO_IMAGE + 1
IMAGE_FILES = ALL_IMAGE_FILES[FROM_IMAGE:end:USE_EVERY]
print(f"Using {len(IMAGE_FILES)} of {len(ALL_IMAGE_FILES)} images "
      f"(from no. {FROM_IMAGE}, to {'the end' if TO_IMAGE is None else f'no. {TO_IMAGE}'}, "
      f"every {USE_EVERY}.)")


sizes = {}   # for every size: (height, width, 3) of the first photo that was opened


def load_photo(file_name, size=MAX_SIZE):
    """Opens one photo as a table height × width × 3 (R, G, B), numbers from 0 to 255.
    size = longest side in pixels. None = full size.
    The first photo decides how big it becomes, and all the others must be the same size."""
    photo = ImageOps.exif_transpose(Image.open(os.path.join(IMAGE_FOLDER, file_name))).convert("RGB")
    if size is not None:
        photo.thumbnail((size, size))
    photo = np.asarray(photo, dtype=float)
    if size not in sizes:
        sizes[size] = photo.shape
    elif photo.shape != sizes[size]:
        raise SystemExit(f"{file_name} has size {photo.shape}, but {IMAGE_FILES[0]} has "
                         f"{sizes[size]}. All the images must be the same size.")
    return photo


height, width, _ = load_photo(IMAGE_FILES[0]).shape
number_of_photos = len(IMAGE_FILES)
n = number_of_photos - 1

print(f"Found {number_of_photos} images in {IMAGE_FOLDER}")
print(f"  photo 0 (t = 0)  →  {IMAGE_FILES[0]}")
print(f"  photo {n} (t = {n})  →  {IMAGE_FILES[-1]}")
print(f"The images are {height} × {width} pixels → {3 * height * width:,} polynomials, but only ONE A.")
print()


# =====================================================================
# 4. SET UP A
#    b is still «one row per photo, one column per pixel(wh) and colour»,
#    but we never build it completely — row i of b is just photo i, rolled out.
# =====================================================================

if DEGREE >= number_of_photos - 1:
    raise SystemExit(f"Degree {DEGREE} with {number_of_photos} photos gives INTERPOLATION. "
                     f"For regression the degree must be at most {number_of_photos - 2}.")

t_values = [time(i, n, SCALE_T) for i in range(number_of_photos)]
A = make_A(t_values, DEGREE)

print(f"A is {A.shape[0]} × {A.shape[1]}  (one photo per row, degree {DEGREE} → {DEGREE + 1} columns)")
print(f"b would have been {number_of_photos} × {3 * height * width:,}  (read one photo at a time)")
print()


# =====================================================================
# 5. FIND A† AND x
#    x = A† b, calculated one photo at a time:
#    x = Σ (column i of A†) · (row i of b)
# =====================================================================

A_plus = pseudoinverse(A)

x = np.zeros((DEGREE + 1, height * width * 3))
show_every = max(1, number_of_photos // 20)           # print progress approx. 20 times

print(f"Calculating x from {number_of_photos} photos:")
for i, file_name in enumerate(IMAGE_FILES):
    image = load_photo(file_name)

    row_i = image.reshape(1, -1)                      # row i of b
    x += A_plus[:, [i]] @ row_i                       # (column i of A†) · (row i of b)

    if i % show_every == 0 or i == n:
        print(f"  photo {i:>5} of {n}")

print(f"x is {x.shape[0]} × {x.shape[1]:,}")

# How much does the regression miss the photos? (0 would have meant interpolation)
# Row i of A multiplied by x is the regression's version of photo i.
# We check up to 50 photos spread out, so it does not take too long.
selection = np.unique(np.linspace(0, n, min(50, number_of_photos)).astype(int))
squares = [np.mean((A[i] @ x - load_photo(IMAGE_FILES[i]).reshape(-1)) ** 2) for i in selection]
print(f"Average miss on the real photos: {np.sqrt(np.mean(squares)):.1f} (of 0–255, "
      f"measured on {len(selection)} photos)")
print()


# =====================================================================
# 6. MAKE FRAMES:  frame(t) = a(t) · x
#    Pixel values must be between 0 and 255 to be saved,
#    so values outside are clipped to the nearest edge.
# =====================================================================

os.makedirs(OUT_FOLDER, exist_ok=True)
image_numbers = np.arange(0, n + STEP / 2, STEP)

show_every = max(1, len(image_numbers) // 20)

print(f"Making {len(image_numbers)} frames, from photo 0 to photo {n}:")
for no, i in enumerate(image_numbers):
    t = time(i, n, SCALE_T)
    new = frame(t, DEGREE, x).reshape(height, width, 3)

    if no % show_every == 0 or no == len(image_numbers) - 1:
        type_ = "real photo" if i == int(i) else "between"
        print(f"  photo {i:>8.2f}  ({type_:10})  values from {new.min():>6.0f} to {new.max():>6.0f}")

    to_save = np.clip(np.round(new), 0, 255).astype(np.uint8)
    Image.fromarray(to_save).save(os.path.join(OUT_FOLDER, f"{no:05d}_photo{i:07.2f}.jpg"), quality=98)

print()
print(f"Done. The images are in {OUT_FOLDER}")
