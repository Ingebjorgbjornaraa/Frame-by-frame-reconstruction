"""
LAGRANGE INTERPOLATION ON IMAGES — ONE CODE, EVERYTHING IS CONTROLLED WITH THE SETTINGS
Follows the note «Lagrange interpolation» in «Kodene mine» (my codes).

THE THEORY (the box in the note)
    ℓ_i(x) = ∏ (x - x_j) / (x_i - x_j)      j = 0..n,  j ≠ i   («the others than x_i»)
    p_n(x) = Σ ℓ_i(x) · y_i                   i = 0..n
    x_0, ..., x_n are the interpolation nodes, and they must be distinct.

HOW I USE IT («how I use it»)
    - the images in the folder are at the times t = 0, 1, 2, ...  (t = the image's number in the folder)
    - the nodes x_i are the times t of some of these images (the y_i are the images themselves)
    - for every pixel(wh) we have R, G and B
    - y_i is the colour value of pixel(wh) in the image at t = x_i
    - so we need p_n^R, p_n^G, p_n^B for every pixel
    - ℓ_0, ..., ℓ_n are the SAME for every pixel and every colour

WHICH IMAGES BECOME NODES (the only thing that separates the different variants)
    NUMBER_OF_NODES = None  →  MIND MAP: every DISTANCE-th image in the whole folder is a node,
                               and there is one polynomial for the whole video.
                               DISTANCE = 1 means that all the images are nodes.
    NUMBER_OF_NODES = 8     →  SLIDING: for every new image only 8 images around x are used,
                               with DISTANCE between them. A small mind map for every image.
"""

import os
import numpy as np
from PIL import Image, ImageOps


# =====================================================================
# SETTINGS — the only thing you need to change
# =====================================================================

IMAGE_FOLDER_NAME = "konsert"   # the folder with the images (found next to the script or further up)

# --- which images become nodes ---
DISTANCE = 380                # how many images there are between two nodes (1 = neighbouring images)
FIRST_NODE = 304              # only for MIND MAP: which image the first node is at.
                              #   304 makes concert image 1824 (the guitarist jumping, 1:00.8)
                              #   a node, because 304 + 4·380 = 1824. Then the jump is shown exactly.
                              #   The nodes become 304, 684, 1064, 1444, 1824, ..., 5624 (15 nodes).
                              #   The images before the first node become extrapolation (glitch intro, approx. 10 sec).
NUMBER_OF_NODES = None        # None = MIND MAP (every DISTANCE-th image in the whole folder)
                              # a number = SLIDING (only that many nodes at a time)
PLACEMENT = 0.5               # only for SLIDING: where x is, in node steps from the first node
                              #   0.5 = between the two first nodes     (double exposure)
                              #   NUMBER_OF_NODES = x after the last node     (guessed from the past)

# --- which x we make new images for ---
SYNC_WITH_SOUND = True        # True: one new image per image in the folder, x = 0, 1, 2, ..., last.
                              #   Then the video becomes as long as the original, and new image k belongs
                              #   to original image k, so the same moment in the music.
                              #   Make the video with 30 images per second and the sound from the original:
                              #   python3 ../make_video.py "lagrange_result/<the folder>" 30 "../../videoene/100_2495.MOV"
EXTRA = 1                     # only when SYNC_WITH_SOUND = False: how many node steps before the first
                              #   and after the last node we go (extrapolation, where it glitches)
STEP = 0.05                   # only when SYNC_WITH_SOUND = False: how far it is between two new
                              #   images, in node steps

# Which images in the folder are used (number = the place in the folder, sorted by file name, first = 0)
FROM_IMAGE = 0      # first image that is included
TO_IMAGE = None     # last image that is included. None = all the way to the end of the folder
USE_EVERY = 1       # 1 = all the images, 3 = every 3rd, 30 = every 30th, ...
                    # NB: with SYNC_WITH_SOUND, USE_EVERY must be 1, otherwise the image goes faster than the sound

MAX_SIZE = None               # longest side in pixels. None = full size. E.g. 600 for a quick test.

# RECIPES
#   Mind map with the test images:  "testbilder", DISTANCE = 1,   NUMBER_OF_NODES = None, SYNC_WITH_SOUND = False
#   Concert, sync with sound (15 nodes, the pain threshold):
#                                   "konsert",    DISTANCE = 380, NUMBER_OF_NODES = None, SYNC_WITH_SOUND = True
#   Concert, very crazy (50 nodes):
#                                   "konsert",    DISTANCE = 114, NUMBER_OF_NODES = None
#   Sliding double exposure:        "konsert",    DISTANCE = 50,  NUMBER_OF_NODES = 8, PLACEMENT = 0.5
#   Sliding, guessed from the past: "konsert",    DISTANCE = 50,  NUMBER_OF_NODES = 4, PLACEMENT = 4

FOLDER = os.path.dirname(os.path.abspath(__file__))   # the folder this script is in
IMAGE_FOLDER = os.path.join(FOLDER, IMAGE_FOLDER_NAME)
for _ in range(3):                                    # if it is not next to the script,
    if os.path.isdir(IMAGE_FOLDER):                   # we look upwards in the folders
        break
    IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.dirname(IMAGE_FOLDER)), IMAGE_FOLDER_NAME)


# =====================================================================
# 1. THE THEORY — straight from the box
# =====================================================================

def lagrange_basis_function(i, x, nodes):
    """ℓ_i(x) = ∏ (x - x_j) / (x_i - x_j) for all j ≠ i
    The curriculum calls this «Lagrangian basis functions» φ_i (lecture 19–20, page 5).
    ℓ_i(x_i) = 1 and ℓ_i(x_j) = 0 for j ≠ i."""
    product = 1.0
    for j in range(len(nodes)):
        if j != i:                                   # «the others than x_i»
            product = product * (x - nodes[j]) / (nodes[i] - nodes[j])
    return product


def p(x, nodes, y):
    """p_n(x) = Σ ℓ_i(x) · y_i"""
    total = 0
    for i in range(len(nodes)):
        total = total + lagrange_basis_function(i, x, nodes) * y[i]
    return total


# =====================================================================
# 2. CHECK AGAINST THE EXAMPLE IN THE NOTE
#    pixel(wh): R = 10 at t=0, R = 20 at t=1, R = 15 at t=2
# =====================================================================

print("The example from the note, x = 0.5:")
print(f"  ℓ_0(0.5) = {lagrange_basis_function(0, 0.5, [0, 1, 2])}")             # 0.375
print(f"  ℓ_1(0.5) = {lagrange_basis_function(1, 0.5, [0, 1, 2])}")             # 0.75
print(f"  ℓ_2(0.5) = {lagrange_basis_function(2, 0.5, [0, 1, 2])}")             # -0.125
print(f"  p_2^R(0.5) = {p(0.5, [0, 1, 2], [10, 20, 15])}")  # 16.875
print()


# =====================================================================
# 3. FIND THE IMAGES IN THE FOLDER
#    They are sorted by file name, so the order in the folder becomes the order in time.
#    Image number k (among the chosen ones) is at t = k. The images are only loaded when they are needed,
#    so a long video does not fill up the memory.
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
h, w = height, width
NUMBER_OF_IMAGES = len(IMAGE_FILES)
print(f"{NUMBER_OF_IMAGES} images in {IMAGE_FOLDER}, {h} × {w} pixels")


# =====================================================================
# 4. WHICH IMAGES ARE NODES FOR THE TIME x?
# =====================================================================

ALL_NODES = list(range(FIRST_NODE, NUMBER_OF_IMAGES, DISTANCE))   # t = FIRST_NODE, + DISTANCE, + 2·DISTANCE, ...

if NUMBER_OF_NODES is None:
    # MIND MAP: the same nodes for all x
    if len(ALL_NODES) < 2:
        raise SystemExit(f"DISTANCE = {DISTANCE} gives only {len(ALL_NODES)} node. Make DISTANCE smaller.")
    if len(ALL_NODES) > 60:
        raise SystemExit(f"DISTANCE = {DISTANCE} gives {len(ALL_NODES)} nodes. Then the ℓ's become so big that "
                         f"the image just becomes noise. Make DISTANCE bigger (try {NUMBER_OF_IMAGES // 15}).")
    print(f"MIND MAP: {len(ALL_NODES)} nodes  → degree n = {len(ALL_NODES) - 1}")
    print(f"  the nodes are at image {', '.join(str(t) for t in ALL_NODES)}")
else:
    # SLIDING: NUMBER_OF_NODES images with DISTANCE between, placed so that x is
    # PLACEMENT node steps after the first node
    if (NUMBER_OF_NODES - 1) * DISTANCE > NUMBER_OF_IMAGES - 1:
        raise SystemExit(f"{NUMBER_OF_NODES} nodes with {DISTANCE} images between need "
                         f"{(NUMBER_OF_NODES - 1) * DISTANCE + 1} images, but the folder has {NUMBER_OF_IMAGES}.")
    print(f"SLIDING: {NUMBER_OF_NODES} nodes at a time with {DISTANCE} images between, "
          f"x is {PLACEMENT} node steps after the first node  → degree n = {NUMBER_OF_NODES - 1}")

def nodes_for(x):
    if NUMBER_OF_NODES is None:
        return ALL_NODES
    # Example: x = 2050, DISTANCE = 50, PLACEMENT = 4  →  nodes = [1850, 1900, 1950, 2000]
    # At the beginning and the end the nodes are pushed in, so we never ask for an image that does not exist.
    length = (NUMBER_OF_NODES - 1) * DISTANCE
    first = int(np.floor(x - PLACEMENT * DISTANCE))
    first = max(0, min(first, NUMBER_OF_IMAGES - 1 - length))
    return [first + j * DISTANCE for j in range(NUMBER_OF_NODES)]


# =====================================================================
# 5A. NEW IMAGE, PIXEL BY PIXEL — exactly as it is written in the note
#     For every pixel(wh) and every colour: y_i = the colour in image i, then p_n(x).
#     Correct, but slow, so we only use it to check 5B.
# =====================================================================

def new_image_pixel_by_pixel(x, nodes, images):
    h, w, _ = images[0].shape
    result = np.zeros((h, w, 3))
    for row in range(h):
        for col in range(w):
            for colour in range(3):                                  # 0 = R, 1 = G, 2 = B
                y = [image[row, col, colour] for image in images]    # y_0, ..., y_n for this pixel
                result[row, col, colour] = p(x, nodes, y)
    return result


# =====================================================================
# 5B. NEW IMAGE, ALL PIXELS AT ONCE
#     Since ℓ_i(x) is the same for all pixels and colours, we can give p() the whole images as y_i:
#         new image = ℓ_0(x)·image_0 + ℓ_1(x)·image_1 + ... + ℓ_n(x)·image_n
#     It is the same p() as above, just faster.
#     The images that are nodes are remembered, so they are not opened again for every new image.
# =====================================================================

memory = {}

def get_image(no):
    if no not in memory:
        memory[no] = load_photo(IMAGE_FILES[no])
    return memory[no]

def new_image(x, nodes):
    for no in list(memory):                          # forget images that are no longer nodes
        if no not in nodes:
            del memory[no]
    images = [get_image(no) for no in nodes]         # y_0, ..., y_n = the images at the nodes
    return p(x, nodes, images)


# Check that 5A and 5B give the same answer (on a small cut-out, so it goes fast)
check_nodes = nodes_for(0)[:4]
check_x = (check_nodes[0] + check_nodes[1]) / 2
cut_out = [get_image(no)[:20, :20] for no in check_nodes]
difference = np.abs(new_image_pixel_by_pixel(check_x, check_nodes, cut_out)
                    - p(check_x, check_nodes, cut_out)).max()
print(f"Check: biggest difference between pixel-by-pixel and all-at-once = {difference}")
print()


# =====================================================================
# 6. WHICH x WE MAKE NEW IMAGES FOR
# =====================================================================

if SYNC_WITH_SOUND:
    # One new image per image in the folder: x = 0, 1, 2, ..., NUMBER_OF_IMAGES - 1.
    # New image k belongs to original image k, so the same moment in the music.
    new_times = list(range(NUMBER_OF_IMAGES))
    print(f"SYNC WITH SOUND: {NUMBER_OF_IMAGES} new images, x = 0, 1, ..., {NUMBER_OF_IMAGES - 1}")
    if NUMBER_OF_NODES is None:
        print(f"  at every node (every {DISTANCE}th image) the image is exactly the same as the original")
    if NUMBER_OF_NODES is None and ALL_NODES[0] > 0:
        print(f"  before the first node (image 0–{ALL_NODES[0] - 1}) it is extrapolation: glitch intro of "
              f"{ALL_NODES[0]} images (approx. {ALL_NODES[0] / 30:.0f} sec)")
    if NUMBER_OF_NODES is None and ALL_NODES[-1] < NUMBER_OF_IMAGES - 1:
        print(f"  after the last node (image {ALL_NODES[-1]}) it is extrapolation: glitch ending of "
              f"{NUMBER_OF_IMAGES - 1 - ALL_NODES[-1]} images (approx. {(NUMBER_OF_IMAGES - 1 - ALL_NODES[-1]) / 30:.0f} sec)")
    name_x = "sync_with_sound"
else:
    last = ALL_NODES[-1] if NUMBER_OF_NODES is None else NUMBER_OF_IMAGES - 1
    first_t = ALL_NODES[0] if NUMBER_OF_NODES is None else 0
    start = first_t - EXTRA * DISTANCE
    stop = last + EXTRA * DISTANCE
    new_times = np.round(np.arange(start, stop + STEP * DISTANCE / 2, STEP * DISTANCE), 6)
    print(f"{len(new_times)} new images from t = {start:g} to t = {stop:g}")
    name_x = f"extra{EXTRA}_step{STEP}"

name_nodes = "mind_map" if NUMBER_OF_NODES is None else f"sliding{NUMBER_OF_NODES}_placement{PLACEMENT}"
if NUMBER_OF_NODES is None and FIRST_NODE != 0:
    name_nodes += f"_start{FIRST_NODE}"      # own folder, so it is not mixed with the nodes from image 0
name_size = "" if MAX_SIZE is None else f"_max{MAX_SIZE}px"
OUT_FOLDER = os.path.join(FOLDER, "lagrange_result",
                          f"{IMAGE_FOLDER_NAME}_distance{DISTANCE}_{name_nodes}_{name_x}{name_size}")


# =====================================================================
# 7. MAKE THE NEW IMAGES
#    Pixel values must be between 0 and 255 to be saved,
#    so values outside are clipped to the nearest edge.
#    The files are numbered (00000, 00001, ...) so they are in the right order.
#    Finished images are skipped, so you can stop (ctrl + C) and start again.
# =====================================================================

os.makedirs(OUT_FOLDER, exist_ok=True)
print(f"Saved in {OUT_FOLDER}")
show_every = 1 if len(new_times) <= 400 else 100     # do not print thousands of lines

for no, x in enumerate(new_times):
    file_name = os.path.join(OUT_FOLDER, f"{no:05d}_t{x:+.2f}.jpg")
    if os.path.exists(file_name):
        continue
    nodes = nodes_for(x)
    new = new_image(x, nodes)

    if no % show_every == 0:
        weights = [lagrange_basis_function(i, x, nodes) for i in range(len(nodes))]
        type_ = "interpolation" if nodes[0] <= x <= nodes[-1] else "EXTRAPOLATION"
        print(f"  image {no:>5}  t = {x:>8.2f}  ({type_:14})  biggest |ℓ_i| = {max(abs(v) for v in weights):>10.2f}")

    to_save = np.clip(np.round(new), 0, 255).astype(np.uint8)
    # Saved first under a temporary name and renamed when the file is completely written,
    # so there is never a half-finished image left if you stop the script.
    Image.fromarray(to_save).save(file_name + ".tmp", format="JPEG", quality=98)
    os.replace(file_name + ".tmp", file_name)

print()
print(f"Done. The images are in {OUT_FOLDER}")
