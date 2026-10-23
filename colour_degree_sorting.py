"""
COLOUR DEGREE SORTING
Follows the note «Counting-Sort» line by line.

THE NOTE SAYS
    we sort the pixels x(t,w,h) by colour (not brightness)
    for all t, for the same w,h per sorting

    x(t,w,h) has R, G and B  =>  we calculate the colour degree

        h(t,w,h) = R·u_R + G·u_G + B·u_B,     u_θ = (cos θ, sin θ)

        colour degree = atan2(h_y, h_x) · 180/π   mod 360

    and the sorting is done with Counting-Sort.

WHY IT DOES NOT BECOME A GREY AXIS
    The three unit vectors are 120 degrees from each other and therefore sum
    to the zero vector. If you put the same amount of light in all three channels,
    h moves along that sum — so nowhere. The colour degree is blind to brightness.
    Three numbers go in and two come out, and the one that is thrown away is exactly
    the grey axis. That is the reason this key does not give a brightness ramp.

WHAT THE CODE DOES
    A window of WINDOW frames slides through the video. For every position:
        - calculate the colour degree in every pixel, in each of the WINDOW frames
        - Counting-Sort the WINDOW colours in every pixel position (w,h) separately
        - save ONE of them: rank RANK
    Then the window moves STEP frames forward, and we do it again.

    The movement in the original is kept, because the window follows along in time.
    The memory use is independent of how long the video is.
"""

import os, math, time
import numpy as np
from PIL import Image, ImageOps


# =====================================================================
# SETTINGS — the only thing you need to change
# =====================================================================

IMAGES = "mamma video"

WINDOW   = 12  # how many frames are sorted together in every position
DISTANCE = 2   # distance between the frames INSIDE the window. 2 = every other
RANK     = 6   # which of the sorted colours is saved, 1 to WINDOW.
               # 1 = first on the colour circle from START_COLOUR, WINDOW = last.
               # In the middle (WINDOW//2) is calmest.

START = 0      # first position the window is in
END   = None   # last position. None = as far as there are frames left
STEP  = 1      # how many frames the window moves per out-frame.
               # 1 = as many out-frames as the original (same speed)

START_COLOUR = 60    # where on the colour circle the order begins (degrees)
MIN_LENGTH = 0.0     # 0 = sort everything. Pixels with shorter |h| than this are
                     # almost grey and cannot be sorted meaningfully; they
                     # are left as they are. Try 10 if it gets grainy.

MAX_SIZE = 1920       # longest side in the out-image

K = 255         # k in Counting-Sort(A, n, k): number of colour degree buckets minus one.
               # 360/(K+1) degrees per bucket. Measured on mamma video, k=63,
               # k=127 and k=255 are visually the same (74–88 % of the pixels completely
               # identical, average deviation 5 of 255), but k=63 is 3.3 times
               # faster than k=255. Set it to 255 if you want the finest
               # possible resolution and have time.

PROCESSES = 4  # how many cores work at the same time. 1 = simple and easy
               # to debug. Every process gets its own continuous piece
               # of the video, so the images become exactly the same regardless.

STRIPE = 60000 # how many pixels are sorted at a time (memory use)

MAX_STORAGE = 36 # how many frames are kept in memory at a time, per process.
                 # Independent of DISTANCE, see section 4. 36 frames in 960 x 540
                 # is approx. 150 MB per process.

ZERO_LIMIT = 1e-6   # shorter h than this counts as grey


# Find the folders (does not need to be changed)
FOLDER = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = None
place = FOLDER
for _ in range(4):
    if os.path.isdir(os.path.join(place, IMAGES)):
        IMAGE_FOLDER = os.path.join(place, IMAGES)
        break
    place = os.path.dirname(place)
if IMAGE_FOLDER is None:
    raise SystemExit(f"Did not find the folder «{IMAGES}» near {FOLDER}.")
OUT_FOLDER = os.path.join(FOLDER, f"work_{IMAGES}_w{WINDOW}d{DISTANCE}_rank{RANK}_from{START_COLOUR}_k{K}")

assert WINDOW <= 255, "C counts up to WINDOW, and is stored as uint8"
assert K <= 255, "the key is stored as uint8"


# =====================================================================
# 1. THE THEORY — straight from the note
# =====================================================================

# The three channels as unit vectors on the hue colour wheel: red at 0°, green at 120°, blue at 240°.
# They are equal — and it is exactly the even distribution that makes them
# sum to the zero vector.
U_R = (math.cos(math.radians(0)),   math.sin(math.radians(0)))     # ( 1   ,  0    )
U_G = (math.cos(math.radians(120)), math.sin(math.radians(120)))   # (-1/2 , +√3/2)
U_B = (math.cos(math.radians(240)), math.sin(math.radians(240)))   # (-1/2 , -√3/2)


def colour_degree(colour):
    """The colour degree of one colour, in degrees 0-360. None if the colour is completely grey.

        h = R·u_R + G·u_G + B·u_B
        colour degree = atan2(h_y, h_x) · 180/π   mod 360

    Every channel is the LENGTH of an arrow; the angle is the direction. atan2 and not
    atan, because atan only covers the right half of the circle — atan2 uses
    the signs of x and y to choose the right quadrant, and covers all of it.

    NOTE the order: the vector is written (x, y), but the function is called
    atan2(y, x) — y first. The same two numbers, opposite order.

    If R = G = B the three arrows cancel each other out and h becomes the zero vector,
    which has no direction. Floating point numbers do not give exactly zero, only something
    vanishingly short, so we test the length against ZERO_LIMIT.
    """
    R, G, B = [float(v) for v in colour]
    hx = R * U_R[0] + G * U_G[0] + B * U_B[0]
    hy = R * U_R[1] + G * U_G[1] + B * U_B[1]
    if math.hypot(hx, hy) < ZERO_LIMIT:
        return None
    return math.degrees(math.atan2(hy, hx)) % 360


def counting_sort(A, key, k=K):
    """Counting-Sort for ONE pixel, word for word from the note.

        Counting-Sort(A, n, k)
         1  new Output[1:n], C[0:k]
         2  for i = 0 to k:
         3      C[i] = 0
         4  for j = 1 to n:
         5      C[A[j]] = C[A[j]] + 1
         6  for i = 1 to k:
         7      C[i] = C[i] + C[i-1]
         8  for j = n downto 1:
         9      Output[C[A[j]]] = A[j]
        10      C[A[j]] = C[A[j]] - 1
        11  return Output

    A is the colours and key[j] is the colour degree bucket of A[j]. The whole colour
    is moved; only the lookup uses the key. We create no new colours.

    The note is 1-indexed, so there it is placed first (line 9) and subtracted
    afterwards (line 10). Python is 0-indexed, so we subtract first and
    place afterwards. The same place, just shifted indexing.

    Line 8 goes BACKWARDS. That is what gives stability: two pixels with the same
    colour degree keep the order they had in time.
    """
    n = len(A)
    Output = [None] * n
    C = [0] * (k + 1)
    for j in range(n):                      # line 4-5: count
        C[key[j]] += 1
    for i in range(1, k + 1):               # line 6-7: "number <= i"
        C[i] += C[i - 1]
    for j in range(n - 1, -1, -1):          # line 8-10: place, backwards
        C[key[j]] -= 1
        Output[C[key[j]]] = A[j]
    return Output


def degrees_to_key(degrees):
    """0-360 degrees -> integer 0..K, so Counting-Sort can be used.
    At the same time we rotate the circle so START_COLOUR ends up at 0."""
    bucket = np.floor(np.asarray(degrees) / 360 * (K + 1)).astype(np.int64)
    return (bucket - round(START_COLOUR / 360 * (K + 1))) % (K + 1)


# =====================================================================
# 2. THE KEY FOR A WHOLE IMAGE
#    Exactly the same calculation as colour_degree(), just done on all
#    the pixels at the same time instead of one at a time.
# =====================================================================

def key_for(image):
    f = image.astype(np.float32)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]

    hx = R * U_R[0] + G * U_G[0] + B * U_B[0]
    hy = R * U_R[1] + G * U_G[1] + B * U_B[1]

    degrees = np.degrees(np.arctan2(hy, hx)) % 360.0
    length = np.hypot(hx, hy)
    degrees[length < ZERO_LIMIT] = 0.0       # grey: no colour degree

    return degrees_to_key(degrees).astype(np.uint8), length.astype(np.float32)


# =====================================================================
# 3. FIND THE FRAMES
# =====================================================================

ALL = [f for f in sorted(os.listdir(IMAGE_FOLDER))
       if f.lower().endswith((".jpg", ".jpeg", ".png"))]
if len(ALL) == 0:
    raise SystemExit(f"Did not find any images in {IMAGE_FOLDER}.")

WIDTH_IN_FRAMES = (WINDOW - 1) * DISTANCE
LAST_POSSIBLE = len(ALL) - 1 - WIDTH_IN_FRAMES
if END is None:
    END = LAST_POSSIBLE
END = min(END, LAST_POSSIBLE)
POSITIONS = list(range(START, END + 1, STEP))

# THE ORDER WE WORK IN — not the one the out-images are saved in.
#
# Window no. q uses the frames q, q+DISTANCE, q+2·DISTANCE ... q+11·DISTANCE.
# If we go forward one position at a time, window q and q+1 share no frames
# when DISTANCE > 1 — they are shifted. To keep all the frames
# in between, the storage would have to hold 11·DISTANCE frames, and with DISTANCE = 25 in
# 960 x 540 that is 1.1 GB per process.
#
# But window q and window q+DISTANCE share 11 of 12! If we sort the positions
# by (q mod DISTANCE, q) we jump DISTANCE forward every time, and the storage
# never needs more than the window itself. Every out-image is independent of the
# others, so which order we calculate them in does not matter —
# the file name follows the position, not the order.
ORDER = sorted(range(len(POSITIONS)),
               key=lambda j: (POSITIONS[j] % max(DISTANCE, 1), POSITIONS[j]))


# =====================================================================
# 4. READ IMAGES — with a storage that follows the window
# =====================================================================

storage = {}        # frame number -> (the image, the key, the length |h|)


def read(index):
    """Read frame no. index, scale down, and calculate the key one time.

    The storage has a fixed ceiling of MAX_STORAGE frames. If it gets full, the one
    that was used longest ago goes. The memory use is therefore the same regardless
    of how big DISTANCE is — it is the ORDER we work in that makes sure the
    ceiling holds (see above).
    """
    if index in storage:
        storage[index] = storage.pop(index)    # move to the back: recently used
        return storage[index]
    image = ImageOps.exif_transpose(
        Image.open(os.path.join(IMAGE_FOLDER, ALL[index]))).convert("RGB")
    if max(image.size) > MAX_SIZE:
        scale = MAX_SIZE / max(image.size)
        image = image.resize((int(image.width * scale) // 2 * 2,
                              int(image.height * scale) // 2 * 2), Image.LANCZOS)
    X = np.asarray(image, dtype=np.uint8)
    storage[index] = (X, *key_for(X))
    while len(storage) > MAX_STORAGE:
        storage.pop(next(iter(storage)))       # oldest out
    return storage[index]


# =====================================================================
# 5. SORT ONE POSITION — Counting-Sort in stripes
#
#    This is counting_sort() above, run on many pixels at the same time.
#    C has one row per pixel in the stripe and K+1 columns — so one whole
#    C-array per pixel, exactly as in the note, just many next to
#    each other. Therefore STRIPE is only a memory choice, not an algorithm.
#
#    C never counts higher than WINDOW, so uint8 is enough, and it is four
#    times less to move around than with full-width integers.
# =====================================================================

def sort_position(X, N, rank):
    n, height, width, _ = X.shape
    P = height * width
    F = X.reshape(n, P, 3)
    Nf = N.reshape(n, P)
    out = np.empty((P, 3), dtype=np.uint8)

    for s in range(0, P, STRIPE):
        e = min(s + STRIPE, P)
        m = e - s
        row = np.arange(m)
        colours = F[:, s:e, :]
        keys = Nf[:, s:e]

        C = np.zeros((m, K + 1), dtype=np.uint8)      # line 1-3
        for j in range(n):                            # line 4-5: count
            C[row, keys[j]] += 1
        np.cumsum(C, axis=1, out=C)                   # line 6-7

        Output = np.empty((n, m, 3), dtype=np.uint8)
        for j in range(n - 1, -1, -1):                # line 8-10: backwards
            kj = keys[j]
            C[row, kj] -= 1
            Output[C[row, kj], row] = colours[j]

        out[s:e] = Output[rank - 1]                   # we only need one rank
    return out.reshape(height, width, 3)


# =====================================================================
# 6. ONE OUT-FRAME
# =====================================================================

def make_frame(counter, pos):
    file_name = os.path.join(OUT_FOLDER, f"{counter:06d}.jpg")
    if os.path.exists(file_name):
        return False

    parts = [read(pos + i * DISTANCE) for i in range(WINDOW)]
    X = np.stack([d[0] for d in parts])
    N = np.stack([d[1] for d in parts])

    out = sort_position(X, N, RANK)

    if MIN_LENGTH > 0:
        L = np.stack([d[2] for d in parts])
        grey = L.mean(axis=0) < MIN_LENGTH
        out[grey] = X[RANK - 1][grey]

    Image.fromarray(out).save(file_name, quality=98)
    return True


def run_block(block):
    """One continuous piece of the video. Every process gets its own, so
    the storage in read() still hits on the neighbouring windows."""
    i0, i1 = block
    done = 0
    for counter in ORDER[i0:i1]:
        if make_frame(counter, POSITIONS[counter]):
            done += 1
    return done


# =====================================================================
# 7. RUN
#    Finished images are skipped, so you can stop (ctrl + C) and start again.
# =====================================================================

if __name__ == "__main__":

    # --- check against the note: the three unit vectors ---
    print("u_R =", tuple(round(v, 4) for v in U_R))
    print("u_G =", tuple(round(v, 4) for v in U_G))
    print("u_B =", tuple(round(v, 4) for v in U_B))
    sx = U_R[0] + U_G[0] + U_B[0]
    sy = U_R[1] + U_G[1] + U_B[1]
    print(f"u_R + u_G + u_B = ({sx:+.1e}, {sy:+.1e})  -> the zero vector, so no grey axis")
    print()

    # --- check against the example in the note: (150, 171, 163) ---
    R, G, B = 150, 171, 163
    hx = R * U_R[0] + G * U_G[0] + B * U_B[0]
    hy = R * U_R[1] + G * U_G[1] + B * U_B[1]
    print(f"The example in the note, rgb = ({R}, {G}, {B}):")
    print(f"  h = ({hx:.2f}, {hy:.2f})        the note: (-17.00 ; 6.93)")
    print(f"  colour degree = {colour_degree((R,G,B)):.1f} degrees   the note: 157.8")
    print()

    # --- check against the landmarks on the circle ---
    for name, colour, answer in [("red",(255,0,0),0), ("yellow",(255,255,0),60),
                                 ("green",(0,255,0),120), ("cyan",(0,255,255),180),
                                 ("blue",(0,0,255),240), ("magenta",(255,0,255),300)]:
        v = colour_degree(colour)
        assert abs(v - answer) < 1e-6, f"{name} gave {v}, should have been {answer}"
    assert colour_degree((120,120,120)) is None, "grey should not have a colour degree"
    print("The landmarks are correct: red 0, yellow 60, green 120, cyan 180, blue 240, magenta 300")
    print("Grey, black and white have no colour degree")
    print()

    # --- check of Counting-Sort against the example in the note ---
    A = [2, 5, 3, 0, 2, 3, 0, 3]
    assert counting_sort(A, A, k=5) == [0, 0, 2, 2, 3, 3, 3, 5]
    print(f"Counting-Sort({A}) = {counting_sort(A, A, k=5)}")
    print()

    print(f"{len(ALL)} frames in «{IMAGES}».")
    print(f"Window: {WINDOW} frames, every {DISTANCE}. — stretches over {WIDTH_IN_FRAMES} frames")
    print(f"Positions: {START} to {END}, step {STEP}  ->  {len(POSITIONS)} out-frames")
    print(f"Saving rank {RANK} of {WINDOW}.   k = {K}  ({360/(K+1):.1f} degrees per bucket)")
    print()

    os.makedirs(OUT_FOLDER, exist_ok=True)

    # --- check: is the stripe version equal to counting_sort()? ---
    parts = [read(POSITIONS[0] + i * DISTANCE) for i in range(WINDOW)]
    Xk = np.stack([d[0] for d in parts]); Nk = np.stack([d[1] for d in parts])
    outk = sort_position(Xk, Nk, RANK)
    rng = np.random.default_rng(0)
    for y, x in zip(rng.integers(0, outk.shape[0], 30), rng.integers(0, outk.shape[1], 30)):
        answer = counting_sort([tuple(int(v) for v in Xk[t, y, x]) for t in range(WINDOW)],
                               [int(Nk[t, y, x]) for t in range(WINDOW)])
        if answer[RANK - 1] != tuple(int(v) for v in outk[y, x]):
            raise SystemExit(f"The stripe version is NOT equal to counting_sort() in pixel ({y},{x}).")
    print(f"Checked against counting_sort() on 30 pixels: equal")
    print(f"Image size: {outk.shape[1]} × {outk.shape[0]}")
    print()

    storage.clear()
    t_start = time.time()

    if PROCESSES <= 1:
        for no, counter in enumerate(ORDER):
            make_frame(counter, POSITIONS[counter])
            if (no + 1) % 10 == 0 or no == 0:
                per = (time.time() - t_start) / (no + 1)
                left = (len(POSITIONS) - no - 1) * per
                print(f"\r  {no+1:>6} of {len(POSITIONS)}   {per:.2f} s per frame"
                      f"   approx {left/3600:.1f} h left   ", end="", flush=True)
    else:
        import multiprocessing as mp
        limit = np.linspace(0, len(ORDER), PROCESSES + 1).astype(int)
        blocks = [(int(limit[i]), int(limit[i+1])) for i in range(PROCESSES)
                  if limit[i+1] > limit[i]]
        print(f"Running on {len(blocks)} cores, one continuous block each.")
        print("(no running counter when several processes write at the same time —")
        print(" see how many files are in the out-folder along the way)")
        with mp.Pool(len(blocks)) as pool:
            pool.map(run_block, blocks)

    used = time.time() - t_start
    print()
    print(f"Done in {used/60:.1f} min. {len(POSITIONS)} images in:")
    print(f"  {OUT_FOLDER}")
    print("Make a video with make_video.py, or:")
    print(f"  ffmpeg -framerate 25 -i '{OUT_FOLDER}/%06d.jpg' -c:v libx264 -pix_fmt yuv420p out.mp4")
