"""
CP TENSOR DECOMPOSITION ON IMAGES
Follows the note «CP tensor decomposition» in «Kodene mine» (my codes).

THE THEORY (from the note)
    the tensor X ∈ ℝ^(time × space × channel)
        time    : t images
        space   : w·h pixels (every image flattened out to one long list)
        channel : colour, R G B
    x_{t,wh} = (R, G, B) ∈ ℝ³   =>  fiber

    Canonical polyadic decomposition  =>  approximates X as a sum of rank 1 tensors

        X ≈ Σ a^(s) ⊚ b^(s) ⊚ c^(s),     s = 1, ..., r     (r = the rank we choose)
              A[:,s]  B[:,s]  C[:,s]

    A (time × rank)    : t rows and r columns, weights for each frame
    B (space × rank)   : pixel 1, ..., pixel n downwards,  b1, ..., br across
    C (channel × rank) : R, G, B downwards,                c1, ..., cr across

    base image s = b_s ⊚ c_s
        every pixel value in b_s is multiplied by each of the three colour numbers in c_s,
        and you get a whole image (w·h × 3)
    we multiply every base image by its weight from A:
        image_i ≈ A[i,1]·G_1 + A[i,2]·G_2 + ... + A[i,r]·G_r      (G_s = base image s)

    We find A, B and C with the ALS algorithm:
        1. Start with random values in A, B and C
        2. Find the best A weights for B and C, save these
        3. Find the best values in B for A and C, save these
        4. Find the best values in C for A and B, save these
        5. Repeat 2, 3 and 4 until the error converges

    => the higher rank, the closer we come to the original input frames
       ↳ we let «time» go, and compute our approximation of X where
         every aspect is decided in relation to each other

HOW «FIND THE BEST» IS CALCULATED (step 2, 3 and 4)
    It is regression, exactly as in «Regression with SVD»:  x = A† b
    Step 2, with B and C locked:
        design matrix (like A in the regression) : C ⊙ B   — one column per base image, flattened out
        data          (like b in the regression) : X₍₁₎ᵀ   — one column per photo, flattened out
        unknowns      (like x in the regression) : Aᵀ      — the weights
        =>  Aᵀ = (C ⊙ B)† X₍₁₎ᵀ
    Step 3 and 4 are the same, seen along the space axis and the colour axis:
        Bᵀ = (C ⊙ A)† X₍₂₎ᵀ          Cᵀ = (B ⊙ A)† X₍₃₎ᵀ
    X₍₁₎, X₍₂₎, X₍₃₎ : the tensor unfolded (Def. 2.4 in the tensor notes)
    ⊙               : Khatri–Rao, column-wise Kronecker (Lemma 2.10)
    A†              : the pseudoinverse, step by step as in the regression note

HOW I USE IT («how I use it»)
    - the script uses ALL the images in the image folder, photo no. i is at t = i
    - RANK decides how many base images all the photos must share
    - the result: every photo built up again from the r base images,
      the base images themselves, and the weights in A
    - NEW_FRAMES = True: «we let time go» — new rows in A with Lagrange,
      and new images from the same base images
    - ALS is calculated on small images (MAX_SIZE) to save memory. Afterwards
      step 3 is calculated one more time on big images (OUT_SIZE), so the result becomes sharp
"""

import os
import numpy as np
from PIL import Image, ImageOps


# =====================================================================
# SETTINGS — the only thing you need to change
# =====================================================================

FOLDER = os.path.dirname(os.path.abspath(__file__))   # the folder this script is in
IMAGE_FOLDER = os.path.join(FOLDER, "nala")           # ALL the images here are used
for _ in range(3):                                    # the script is in «code/cp»,
    if os.path.isdir(IMAGE_FOLDER):                   # so we look upwards for the test images
        break
    IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.dirname(IMAGE_FOLDER)), "nala")
OUT_FOLDER = os.path.join(FOLDER, "cp_result_nala_every25")

# Which images in the folder are used (number = the place in the folder, sorted by file name, first = 0)
FROM_IMAGE = 0      # first image that is included
TO_IMAGE = None     # last image that is included. None = all the way to the end of the folder
USE_EVERY = 25      # 1 = all the images, 3 = every 3rd, 30 = every 30th, ...
                    # Many images use a lot of memory: stay below approx. 100–150 images

RANK = 15            # r, number of base images. Low = painterly and bound together,
                    # high = closer to the originals

MAX_ROUNDS = 200    # step 5: how many times 2, 3 and 4 are repeated at most
TOLERANCE = 1e-6    # step 5: stop when the error changes less than this (relatively)
SEED = 0            # step 1: which random start values. Same seed = same result

MAX_SIZE = 400          # longest side in pixels. The photos are 4608 × 3456; bigger images
                        # give a sharper result, but ALS takes much longer.
                        # None = full size.

OUT_SIZE = 1920         # longest side in pixels on the images that are SAVED. ALS is calculated on
                        # MAX_SIZE, but B is calculated again in this size
                        # (see part 5). Smaller photos are not enlarged.
                        # None = the same size as MAX_SIZE.

NEW_FRAMES = False  # «we let time go»: make new frames by Lagrange-interpolating A
EXTRA = 1           # only if NEW_FRAMES: goes from t = -EXTRA to t = n + EXTRA
STEP = 0.5          # only if NEW_FRAMES: distance between the frames in t


# =====================================================================
# 1. THE THEORY — the building blocks from the note and the tensor notes
# =====================================================================

def unfold(X, mode):
    """
    Matricization X₍mode₎ (Def. 2.4): one axis becomes rows, the two others are rolled out
    to columns, «first index first» as in the notes.
        X₍₁₎ : one row per photo      (t × w·h·3)
        X₍₂₎ : one row per pixel      (w·h × t·3)
        X₍₃₎ : one row per colour     (3 × t·w·h)
    """
    return np.moveaxis(X, mode - 1, 0).reshape(X.shape[mode - 1], -1, order="F")


def khatri_rao(P, Q):
    """P ⊙ Q: column s is p_s ⊗ q_s (column-wise Kronecker)"""
    rows_P, r = P.shape
    rows_Q, _ = Q.shape
    return (P[:, None, :] * Q[None, :, :]).reshape(rows_P * rows_Q, r)


def pseudoinverse(A):
    """A† = V Σ⁻¹ Uᵀ, step by step as in the regression note"""

    # 1. AᵀA
    ATA = A.T @ A

    # 2. eigenvalues λ and eigenvectors v_k of AᵀA (biggest first)
    λ, V = np.linalg.eigh(ATA)
    order = np.argsort(λ)[::-1]
    λ = λ[order]
    V = V[:, order]

    # 3. σ_k = √λ_k and Σ⁻¹   (λ can become a tiny bit negative from rounding, therefore max 0)
    σ = np.sqrt(np.maximum(λ, 0))
    Σ_inv = np.diag(1 / σ)

    # 4. u_k = (1/σ_k) · A · v_k
    U = np.zeros((A.shape[0], len(σ)))
    for k in range(len(σ)):
        U[:, k] = (1 / σ[k]) * A @ V[:, k]

    # 5. A† = V Σ⁻¹ Uᵀ
    return V @ Σ_inv @ U.T


def error(X1, A, B, C):
    """‖X − Σ a⊚b⊚c‖, measured on X₍₁₎ = A (C ⊙ B)ᵀ (Lemma 2.10)"""
    return np.linalg.norm(X1 - A @ khatri_rao(C, B).T)


def als(X, rank, max_rounds, tolerance, seed, show=True):
    """We find A, B and C with the ALS algorithm — step 1 to 5 from the note"""
    t, wh, channels = X.shape
    X1, X2, X3 = unfold(X, 1), unfold(X, 2), unfold(X, 3)

    # 1. Start with random values in A, B and C  (random numbers between 0 and 1)
    random = np.random.default_rng(seed)
    A = random.random(size=(t, rank))
    B = random.random(size=(wh, rank))
    C = random.random(size=(channels, rank))

    previous = np.inf
    for round_ in range(1, max_rounds + 1):
        # 2. Find the best A weights for B and C, save these
        A = (pseudoinverse(khatri_rao(C, B)) @ X1.T).T

        # 3. Find the best values in B for A and C, save these
        B = (pseudoinverse(khatri_rao(C, A)) @ X2.T).T

        # 4. Find the best values in C for A and B, save these
        C = (pseudoinverse(khatri_rao(B, A)) @ X3.T).T

        # 5. Repeat 2, 3 and 4 until the error converges
        now = error(X1, A, B, C)
        if show and (round_ == 1 or round_ % 10 == 0):
            print(f"  round {round_:>4}:  error = {now:,.1f}")
        if round_ > 1 and abs(previous - now) <= tolerance * previous:
            if show:
                print(f"  round {round_:>4}:  error = {now:,.1f}   ← converged")
            break
        previous = now

    return A, B, C


def base_image(B, C, s):
    """base image s = b_s ⊚ c_s  (w·h × 3)"""
    return np.outer(B[:, s], C[:, s])


def image(A, B, C, i):
    """image_i ≈ A[i,1]·G_1 + A[i,2]·G_2 + ... + A[i,r]·G_r"""
    return sum(A[i, s] * base_image(B, C, s) for s in range(A.shape[1]))


def lagrange_basis_function(i, x, nodes):
    """ℓ_i(x) = ∏ (x - x_j) / (x_i - x_j) for all j ≠ i
    The curriculum calls this «Lagrangian basis functions» φ_i (lecture 19–20, page 5).
    ℓ_i(x_i) = 1 and ℓ_i(x_j) = 0 for j ≠ i."""
    product = 1.0
    for j in range(len(nodes)):
        if j != i:
            product = product * (x - nodes[j]) / (nodes[i] - nodes[j])
    return product


def new_row_in_A(t_new, A):
    """«we let time go»: a new row of A for a t we choose ourselves, p_n(t) = Σ ℓ_i(t) · A[i,:]"""
    nodes = range(A.shape[0])
    return sum(lagrange_basis_function(i, t_new, nodes) * A[i] for i in nodes)


# =====================================================================
# 2. CHECK AGAINST THE EXAMPLE (from «The tensor behind the images»)
#    3 images, 4 pixels, R G B, made from ONE building block:
#    a = [0.5, 1.0, 1.5]   b = [1.0, 0.6, 0.9, 0.3]   c = [1.0, 0.7, 0.4]
#    With rank 1 ALS should find it again, and the error should become ~0.
# =====================================================================

np.set_printoptions(precision=3, suppress=True)

a_ex = np.array([0.5, 1.0, 1.5])
b_ex = np.array([1.0, 0.6, 0.9, 0.3])
c_ex = np.array([1.0, 0.7, 0.4])
X_ex = 180 * np.einsum("i,j,k->ijk", a_ex, b_ex, c_ex)    # a ⊚ b ⊚ c, multiplied by 180

print("The example: X = 180 · a ⊚ b ⊚ c")
print("image t=1 (pixel downwards, R G B across):\n", X_ex[1])  # [[180 126 72] ...]

A_e, B_e, C_e = np.array([a_ex]).T, np.array([b_ex]).T, np.array([c_ex]).T
print("Lemma 2.10, X₍₁₎ = A (C ⊙ B)ᵀ holds:",
      np.allclose(unfold(X_ex, 1), 180 * A_e @ khatri_rao(C_e, B_e).T))

A_e, B_e, C_e = als(X_ex, rank=1, max_rounds=50, tolerance=1e-12, seed=0, show=False)
print(f"ALS with rank 1: error = {error(unfold(X_ex, 1), A_e, B_e, C_e):.2e}")
print("reconstructed image t=1:\n", image(A_e, B_e, C_e, 1))
print()


# =====================================================================
# 3. LOAD ALL THE IMAGES IN THE FOLDER  →  the tensor X
#    The images are sorted by file name, so the order in the folder becomes the order in time.
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

images = [load_photo(file_name) for file_name in IMAGE_FILES]

# the tensor X ∈ ℝ^(time × space × channel): every photo h × w × 3 is flattened out to (w·h) × 3
X = np.stack([photo.reshape(height * width, 3) for photo in images])
t, wh, channels = X.shape
n = t - 1

print(f"Found {t} images in {IMAGE_FOLDER}:")
for i, file_name in enumerate(IMAGE_FILES):
    print(f"  photo {i:>2} (t = {i})  →  {file_name}")
print(f"X is {t} × {wh:,} × {channels}   (time × space × channel, space = {height}·{width})")
print(f"fiber x_(t=0, pixel 1) = (R, G, B) = {X[0, 0]}")
print()


# =====================================================================
# 4. ALS — find A, B and C
# =====================================================================

print(f"ALS with rank r = {RANK}:")
A, B, C = als(X, RANK, MAX_ROUNDS, TOLERANCE, SEED)

# Make the base images easier to read: the biggest value in every b_s and c_s becomes 1,
# and the scale is moved over into A. It does NOT change X ≈ Σ a⊚b⊚c, only where the numbers are.
for s in range(RANK):
    for M in (B, C):
        scale = M[np.argmax(np.abs(M[:, s])), s]
        M[:, s] /= scale
        A[:, s] *= scale

print()
print(f"A is {A.shape[0]} × {A.shape[1]}, B is {B.shape[0]:,} × {B.shape[1]}, "
      f"C is {C.shape[0]} × {C.shape[1]}")
print("C (the colour of every base image, R G B downwards, c1 ... cr across):\n", C)
print("A (the weights, one photo per row, one base image per column):\n", A)
print()


# =====================================================================
# 5. STEP 3 ONE MORE TIME, IN OUT_SIZE
#    A (time) and C (colour) do not depend on how big the images are, only B (where in the image).
#    So we keep A and C from ALS and only calculate B again, from the big photos.
# =====================================================================

# How much does the model miss the real photos? (measured on the small images, before B is replaced)
deviation = [np.mean((image(A, B, C, i) - X[i]) ** 2) for i in range(t)]
print(f"Average miss on the real photos: {np.sqrt(np.mean(deviation)):.1f} (of 0–255)")


def step_3_again(A, C):
    """Step 3 in ALS one time, with A and C locked, on the photos in OUT_SIZE:

        B = unfold(X_big, 2) @ pseudoinverse(khatri_rao(C, A)).T

    It is Bᵀ = (C ⊙ A)† X₍₂₎ᵀ without transposing:  B = X₍₂₎ · M,  where M = ((C ⊙ A)†)ᵀ.
    The whole X₍₂₎ for big images does not fit in memory, so we take one photo at a time:

        B = Σ_i  X_i · M_i

    X_i : photo i, one row per pixel, R G B across   (the three columns in X₍₂₎ that belong to photo i)
    M_i : row i, t + i and 2t + i in M                (the three rows in M that belong to photo i)
    """
    t = A.shape[0]                                   # number of photos
    M = pseudoinverse(khatri_rao(C, A)).T            # (t·3) × r, row no. k·t + i = colour k in photo i
    B = 0
    for i, file_name in enumerate(IMAGE_FILES):
        X_i = load_photo(file_name, OUT_SIZE).reshape(-1, 3)    # one row per pixel, R G B across
        M_i = M[[i, t + i, 2 * t + i], :]                       # the rows for R, G and B in photo i
        B = B + X_i @ M_i                                       # add the contribution from photo i
        if (i + 1) % 20 == 0 or i == t - 1:
            print(f"  {i + 1} of {t} photos")
    return B


out_height, out_width = height, width
if OUT_SIZE is not None and OUT_SIZE != MAX_SIZE:
    print(f"\nStep 3 one more time, in size {OUT_SIZE}:")
    B = step_3_again(A, C)
    out_height, out_width, _ = sizes[OUT_SIZE]
    print(f"B is now {B.shape[0]:,} × {B.shape[1]}   (the images are {out_height} × {out_width})")


# =====================================================================
# 6. SAVE THE RESULT
#    Pixel values must be between 0 and 255 to be saved,
#    so values outside are clipped to the nearest edge.
# =====================================================================

def save(values, file_name):
    to_save = np.clip(np.round(values), 0, 255).astype(np.uint8)
    Image.fromarray(to_save.reshape(out_height, out_width, 3)).save(os.path.join(OUT_FOLDER, file_name),
                                                                   quality=98)

os.makedirs(OUT_FOLDER, exist_ok=True)

# the base images, with stretched contrast so the pattern shows (lowest value black, highest white)
for s in range(RANK):
    G = base_image(B, C, s)
    save(255 * (G - G.min()) / (G.max() - G.min()), f"base_image_{s + 1}.jpg")

# every photo built up again: image_i ≈ Σ A[i,s]·G_s
for i in range(t):
    save(image(A, B, C, i), f"photo{i:04d}_rank{RANK}.jpg")


# =====================================================================
# 7. «WE LET TIME GO» — new frames (only if NEW_FRAMES = True)
#    B and C are locked. A new frame is a new row in A, made with Lagrange,
#    and built from the SAME base images.
# =====================================================================

if NEW_FRAMES:
    times = np.arange(-EXTRA, n + EXTRA + STEP / 2, STEP)
    print(f"\nMaking {len(times)} frames, from t = {-EXTRA} to t = {n + EXTRA}:")
    for no, t_new in enumerate(times):
        row = new_row_in_A(t_new, A)
        new = sum(row[s] * base_image(B, C, s) for s in range(RANK))
        type_ = "real photo" if t_new == int(t_new) and 0 <= t_new <= n else (
                "between" if 0 <= t_new <= n else "outside")
        print(f"  t = {t_new:>6.2f}  ({type_:10})  weights {row}")
        save(new, f"new_{no:04d}_t{t_new:+06.2f}.jpg")

print()
print(f"Done. The images are in {OUT_FOLDER}")
