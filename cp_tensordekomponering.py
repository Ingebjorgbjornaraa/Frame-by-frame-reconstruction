"""
CP-TENSORDEKOMPONERING PÅ BILDERcaffeinate
Følger notatet «CP-tensordekomponering» i «Kodene mine».

TEORIEN (fra notatet)
    the tensor X ∈ ℝ^(time × space × channel)
        time    : t bilder
        space   : h·b pixler (hvert bilde flatet ut til én lang liste)
        channel : farge, R G B
    x_{t,hb} = (R, G, B) ∈ ℝ³   =>  fiber

    Canonical polyadic decomposition  =>  approksimerer X som en sum av rang-1-tensorer

        X ≈ Σ a^(s) ⊚ b^(s) ⊚ c^(s),     s = 1, ..., r     (r = rangen vi velger)
              A[:,s]  B[:,s]  C[:,s]

    A (time × rank)    : t rader og r kolonner, vekter for hver frame
    B (space × rank)   : pixel 1, ..., pixel n nedover,  b1, ..., br bortover
    C (channel × rank) : R, G, B nedover,                c1, ..., cr bortover

    grunnbilde s = b_s ⊚ c_s
        hver pixelverdi i b_s ganges med hvert av de tre fargetallene i c_s,
        og du får et helt bilde (h·b × 3)
    vi ganger hvert grunnbilde med vekten sin fra A:
        bilde_i ≈ A[i,1]·G_1 + A[i,2]·G_2 + ... + A[i,r]·G_r

    We find A, B and C with the ALS algoritm:
        1. Start with random values in A, B and C
        2. Find the best A weights for B and C, save these
        3. Find the best values in B for A and C, save these
        4. Find the best values in C for A and B, save these
        5. Repeat 2, 3 and 4 until the error converges

    => the higher rank, the closer we come to the original input frames
       ↳ we let «time» go, and compute our approximation of X where
         every aspect is decided in relation to each other

HVORDAN «FINN DE BESTE» REGNES UT (steg 2, 3 og 4)
    Det er regresjon, akkurat som i «Regression med SVD»:  x = A† b
    Steg 2, med B og C låst:
        designmatrise (som A i regresjonen) : C ⊙ B   — én kolonne per grunnbilde, flatet ut
        data          (som b i regresjonen) : X₍₁₎ᵀ   — én kolonne per foto, flatet ut
        ukjente       (som x i regresjonen) : Aᵀ      — vektene
        =>  Aᵀ = (C ⊙ B)† X₍₁₎ᵀ
    Steg 3 og 4 er det samme, sett langs rom-aksen og farge-aksen:
        Bᵀ = (C ⊙ A)† X₍₂₎ᵀ          Cᵀ = (B ⊙ A)† X₍₃₎ᵀ
    X₍₁₎, X₍₂₎, X₍₃₎ : tensoren brettet ut (Def. 2.4 i tensor-notatet)
    ⊙               : Khatri–Rao, kolonnevis Kronecker (Lemma 2.10)
    A†              : pseudoinversen, steg for steg som i regresjonsnotatet

HVORDAN JEG BRUKER DET («how I use it»)
    - scriptet bruker ALLE bildene i bildemappen, foto nr. i står på t = i
    - RANG bestemmer hvor mange grunnbilder alle fotoene må deles om
    - resultatet: hvert foto bygd opp igjen av de r grunnbildene,
      grunnbildene selv, og vektene i A
    - NYE_FRAMES = True: «we let time go» — nye rader i A med Lagrange,
      og nye bilder av de samme grunnbildene
    - ALS regnes på små bilder (MAKS_STØRRELSE) for å spare minne. Etterpå regnes
      steg 3 ut én gang til på store bilder (UT_STØRRELSE), så resultatet blir skarpt
"""

import os
import numpy as np
from PIL import Image, ImageOps


# =====================================================================
# INNSTILLINGER — det eneste du trenger å endre
# =====================================================================

MAPPE = os.path.dirname(os.path.abspath(__file__))   # mappen dette scriptet ligger i
BILDEMAPPE = os.path.join(MAPPE, "nala")     # ALLE bildene her blir brukt
for _ in range(3):                                   # scriptet ligger i «kode/cp»,
    if os.path.isdir(BILDEMAPPE):                    # så vi leter oppover etter testbilder
        break
    BILDEMAPPE = os.path.join(os.path.dirname(os.path.dirname(BILDEMAPPE)), "nala")
UTMAPPE = os.path.join(MAPPE, "cp_resultat_nala_hvert25")

# Hvilke bilder i mappen som brukes (nummer = plassen i mappen, sortert etter filnavn, første = 0)
FRA_BILDE = 0       # første bilde som tas med
TIL_BILDE = None    # siste bilde som tas med. None = helt til slutten av mappen
BRUK_HVERT = 25     # 1 = alle bildene, 3 = hvert 3., 30 = hvert 30., ...
                    # Mange bilder bruker mye minne: hold deg under ca. 100–150 bilder

RANG = 15            # r, antall grunnbilder. Lav = malerisk og sammenbundet,
                    # høy = nærmere originalene

MAKS_RUNDER = 200   # steg 5: hvor mange ganger 2, 3 og 4 gjentas høyst
TOLERANSE = 1e-6    # steg 5: stopp når feilen endrer seg mindre enn dette (relativt)
SEED = 0            # steg 1: hvilke tilfeldige startverdier. Samme seed = samme resultat

MAKS_STØRRELSE = 400    # lengste side i pixler. Fotoene er 4608 × 3456; større bilder
                        # gir skarpere resultat, men ALS tar mye lengre tid.
                        # None = full størrelse.

UT_STØRRELSE = 1920     # lengste side i pixler på bildene som LAGRES. ALS regnes på
                        # MAKS_STØRRELSE, men B regnes ut på nytt i denne størrelsen
                        # (se del 5). Mindre fotoer blir ikke forstørret.
                        # None = samme størrelse som MAKS_STØRRELSE.

NYE_FRAMES = False  # «we let time go»: lag nye frames ved å Lagrange-interpolere A
EKSTRA = 1          # bare hvis NYE_FRAMES: går fra t = -EKSTRA til t = n + EKSTRA
STEG = 0.5          # bare hvis NYE_FRAMES: avstand mellom framene i t


# =====================================================================
# 1. TEORIEN — byggesteinene fra notatet og tensor-notatet
# =====================================================================

def brett_ut(X, modus):
    """
    Matricization X₍modus₎ (Def. 2.4): én akse blir rader, de to andre rulles ut
    til kolonner, «første indeks først» som i notatet.
        X₍₁₎ : én rad per foto      (t × h·b·3)
        X₍₂₎ : én rad per pixel     (h·b × t·3)
        X₍₃₎ : én rad per farge     (3 × t·h·b)
    """
    return np.moveaxis(X, modus - 1, 0).reshape(X.shape[modus - 1], -1, order="F")


def khatri_rao(P, Q):
    """P ⊙ Q: kolonne s er p_s ⊗ q_s (kolonnevis Kronecker)"""
    rader_P, r = P.shape
    rader_Q, _ = Q.shape
    return (P[:, None, :] * Q[None, :, :]).reshape(rader_P * rader_Q, r)


def pseudoinvers(A):
    """A† = V Σ⁻¹ Uᵀ, steg for steg som i regresjonsnotatet"""

    # 1. AᵀA
    ATA = A.T @ A

    # 2. egenverdier λ og egenvektorer v_k til AᵀA (største først)
    λ, V = np.linalg.eigh(ATA)
    rekkefølge = np.argsort(λ)[::-1]
    λ = λ[rekkefølge]
    V = V[:, rekkefølge]

    # 3. σ_k = √λ_k og Σ⁻¹   (λ kan bli bitte litt negativ av avrunding, derfor max 0)
    σ = np.sqrt(np.maximum(λ, 0))
    Σ_inv = np.diag(1 / σ)

    # 4. u_k = (1/σ_k) · A · v_k
    U = np.zeros((A.shape[0], len(σ)))
    for k in range(len(σ)):
        U[:, k] = (1 / σ[k]) * A @ V[:, k]

    # 5. A† = V Σ⁻¹ Uᵀ
    return V @ Σ_inv @ U.T


def feil(X1, A, B, C):
    """‖X − Σ a⊚b⊚c‖, målt på X₍₁₎ = A (C ⊙ B)ᵀ (Lemma 2.10)"""
    return np.linalg.norm(X1 - A @ khatri_rao(C, B).T)


def als(X, rang, maks_runder, toleranse, seed, vis=True):
    """We find A, B and C with the ALS algoritm — steg 1 til 5 fra notatet"""
    t, hb, kanaler = X.shape
    X1, X2, X3 = brett_ut(X, 1), brett_ut(X, 2), brett_ut(X, 3)

    # 1. Start with random values in A, B and C  (tilfeldige tall mellom 0 og 1)
    tilfeldig = np.random.default_rng(seed)
    A = tilfeldig.random(size=(t, rang))
    B = tilfeldig.random(size=(hb, rang))
    C = tilfeldig.random(size=(kanaler, rang))

    forrige = np.inf
    for runde in range(1, maks_runder + 1):
        # 2. Find the best A weights for B and C, save these
        A = (pseudoinvers(khatri_rao(C, B)) @ X1.T).T

        # 3. Find the best values in B for A and C, save these
        B = (pseudoinvers(khatri_rao(C, A)) @ X2.T).T

        # 4. Find the best values in C for A and B, save these
        C = (pseudoinvers(khatri_rao(B, A)) @ X3.T).T

        # 5. Repeat 2, 3 and 4 until the error converges
        nå = feil(X1, A, B, C)
        if vis and (runde == 1 or runde % 10 == 0):
            print(f"  runde {runde:>4}:  feil = {nå:,.1f}")
        if runde > 1 and abs(forrige - nå) <= toleranse * forrige:
            if vis:
                print(f"  runde {runde:>4}:  feil = {nå:,.1f}   ← konvergert")
            break
        forrige = nå

    return A, B, C


def grunnbilde(B, C, s):
    """grunnbilde s = b_s ⊚ c_s  (h·b × 3)"""
    return np.outer(B[:, s], C[:, s])


def bilde(A, B, C, i):
    """bilde_i ≈ A[i,1]·G_1 + A[i,2]·G_2 + ... + A[i,r]·G_r"""
    return sum(A[i, s] * grunnbilde(B, C, s) for s in range(A.shape[1]))


def lagrange_basisfunksjon(i, x, noder):
    """ℓ_i(x) = ∏ (x - x_j) / (x_i - x_j) for alle j ≠ i
    Pensum kaller dette «Lagrangian basis functions» φ_i (forelesning 19–20, side 5).
    ℓ_i(x_i) = 1 og ℓ_i(x_j) = 0 for j ≠ i."""
    produkt = 1.0
    for j in range(len(noder)):
        if j != i:
            produkt = produkt * (x - noder[j]) / (noder[i] - noder[j])
    return produkt


def ny_rad_i_A(t_ny, A):
    """«we let time go»: en ny rad av A for en t vi velger selv, p_n(t) = Σ ℓ_i(t) · A[i,:]"""
    noder = range(A.shape[0])
    return sum(lagrange_basisfunksjon(i, t_ny, noder) * A[i] for i in noder)


# =====================================================================
# 2. SJEKK MOT EKSEMPELET (fra «Tensoren bak bildene»)
#    3 bilder, 4 pixler, R G B, laget av ÉN byggekloss:
#    a = [0.5, 1.0, 1.5]   b = [1.0, 0.6, 0.9, 0.3]   c = [1.0, 0.7, 0.4]
#    Med rang 1 skal ALS finne den igjen, og feilen skal bli ~0.
# =====================================================================

np.set_printoptions(precision=3, suppress=True)

a_eks = np.array([0.5, 1.0, 1.5])
b_eks = np.array([1.0, 0.6, 0.9, 0.3])
c_eks = np.array([1.0, 0.7, 0.4])
X_eks = 180 * np.einsum("i,j,k->ijk", a_eks, b_eks, c_eks)    # a ⊚ b ⊚ c, ganget med 180

print("Eksempelet: X = 180 · a ⊚ b ⊚ c")
print("bilde t=1 (pixel nedover, R G B bortover):\n", X_eks[1])  # [[180 126 72] ...]

A_e, B_e, C_e = np.array([a_eks]).T, np.array([b_eks]).T, np.array([c_eks]).T
print("Lemma 2.10, X₍₁₎ = A (C ⊙ B)ᵀ stemmer:",
      np.allclose(brett_ut(X_eks, 1), 180 * A_e @ khatri_rao(C_e, B_e).T))

A_e, B_e, C_e = als(X_eks, rang=1, maks_runder=50, toleranse=1e-12, seed=0, vis=False)
print(f"ALS med rang 1: feil = {feil(brett_ut(X_eks, 1), A_e, B_e, C_e):.2e}")
print("rekonstruert bilde t=1:\n", bilde(A_e, B_e, C_e, 1))
print()


# =====================================================================
# 3. LAST INN ALLE BILDENE I MAPPEN  →  tensoren X
#    Bildene sorteres etter filnavn, så rekkefølgen i mappen blir rekkefølgen i tid.
# =====================================================================

ALLE_BILDEFILER = [f for f in sorted(os.listdir(BILDEMAPPE))
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
slutt = None if TIL_BILDE is None else TIL_BILDE + 1
BILDEFILER = ALLE_BILDEFILER[FRA_BILDE:slutt:BRUK_HVERT]
print(f"Bruker {len(BILDEFILER)} av {len(ALLE_BILDEFILER)} bilder "
      f"(fra nr. {FRA_BILDE}, til {'slutten' if TIL_BILDE is None else f'nr. {TIL_BILDE}'}, "
      f"hvert {BRUK_HVERT}.)")


størrelser = {}   # for hver størrelse: (høyde, bredde, 3) til det første fotoet som ble åpnet


def last_foto(filnavn, størrelse=MAKS_STØRRELSE):
    """Åpner ett foto som en tabell høyde × bredde × 3 (R, G, B), tall fra 0 til 255.
    størrelse = lengste side i pixler. None = full størrelse.
    Det første fotoet bestemmer hvor stort det blir, og alle de andre må være like store."""
    foto = ImageOps.exif_transpose(Image.open(os.path.join(BILDEMAPPE, filnavn))).convert("RGB")
    if størrelse is not None:
        foto.thumbnail((størrelse, størrelse))
    foto = np.asarray(foto, dtype=float)
    if størrelse not in størrelser:
        størrelser[størrelse] = foto.shape
    elif foto.shape != størrelser[størrelse]:
        raise SystemExit(f"{filnavn} har størrelse {foto.shape}, men {BILDEFILER[0]} har "
                         f"{størrelser[størrelse]}. Alle bildene må være like store.")
    return foto


høyde, bredde, _ = last_foto(BILDEFILER[0]).shape

bilder = [last_foto(filnavn) for filnavn in BILDEFILER]

# the tensor X ∈ ℝ^(time × space × channel): hvert foto h × b × 3 flates ut til (h·b) × 3
X = np.stack([foto.reshape(høyde * bredde, 3) for foto in bilder])
t, hb, kanaler = X.shape
n = t - 1

print(f"Fant {t} bilder i {BILDEMAPPE}:")
for i, filnavn in enumerate(BILDEFILER):
    print(f"  foto {i:>2} (t = {i})  →  {filnavn}")
print(f"X er {t} × {hb:,} × {kanaler}   (time × space × channel, space = {høyde}·{bredde})")
print(f"fiber x_(t=0, pixel 1) = (R, G, B) = {X[0, 0]}")
print()


# =====================================================================
# 4. ALS — finn A, B og C
# =====================================================================

print(f"ALS med rang r = {RANG}:")
A, B, C = als(X, RANG, MAKS_RUNDER, TOLERANSE, SEED)

# Gjør grunnbildene lettere å lese: største verdi i hver b_s og c_s blir 1,
# og skalaen flyttes over i A. Det endrer IKKE X ≈ Σ a⊚b⊚c, bare hvor tallene står.
for s in range(RANG):
    for M in (B, C):
        skala = M[np.argmax(np.abs(M[:, s])), s]
        M[:, s] /= skala
        A[:, s] *= skala

print()
print(f"A er {A.shape[0]} × {A.shape[1]}, B er {B.shape[0]:,} × {B.shape[1]}, "
      f"C er {C.shape[0]} × {C.shape[1]}")
print("C (fargen til hvert grunnbilde, R G B nedover, c1 ... cr bortover):\n", C)
print("A (vektene, ett foto per rad, ett grunnbilde per kolonne):\n", A)
print()


# =====================================================================
# 5. STEG 3 ÉN GANG TIL, I UT_STØRRELSE
#    A (tid) og C (farge) avhenger ikke av hvor store bildene er, bare B (hvor i bildet).
#    Så vi beholder A og C fra ALS og regner bare ut B på nytt, fra de store fotoene.
# =====================================================================

# Hvor mye bommer modellen på de ekte fotoene? (målt på de små bildene, før B byttes ut)
avvik = [np.mean((bilde(A, B, C, i) - X[i]) ** 2) for i in range(t)]
print(f"Gjennomsnittlig bom på de ekte fotoene: {np.sqrt(np.mean(avvik)):.1f} (av 0–255)")


def steg_3_på_nytt(A, C):
    """Steg 3 i ALS én gang, med A og C låst, på fotoene i UT_STØRRELSE:

        B = brett_ut(X_stor, 2) @ pseudoinvers(khatri_rao(C, A)).T

    Det er Bᵀ = (C ⊙ A)† X₍₂₎ᵀ uten transponering:  B = X₍₂₎ · M,  der M = ((C ⊙ A)†)ᵀ.
    Hele X₍₂₎ for store bilder får ikke plass i minnet, så vi tar ett foto om gangen:

        B = Σ_i  X_i · M_i

    X_i : foto i, én rad per pixel, R G B bortover   (de tre kolonnene i X₍₂₎ som hører til foto i)
    M_i : rad i, t + i og 2t + i i M                 (de tre radene i M som hører til foto i)
    """
    t = A.shape[0]                                   # antall fotoer
    M = pseudoinvers(khatri_rao(C, A)).T             # (t·3) × r, rad nr. k·t + i = farge k i foto i
    B = 0
    for i, filnavn in enumerate(BILDEFILER):
        X_i = last_foto(filnavn, UT_STØRRELSE).reshape(-1, 3)   # én rad per pixel, R G B bortover
        M_i = M[[i, t + i, 2 * t + i], :]                       # radene for R, G og B i foto i
        B = B + X_i @ M_i                                       # legg til bidraget fra foto i
        if (i + 1) % 20 == 0 or i == t - 1:
            print(f"  {i + 1} av {t} fotoer")
    return B


ut_høyde, ut_bredde = høyde, bredde
if UT_STØRRELSE is not None and UT_STØRRELSE != MAKS_STØRRELSE:
    print(f"\nSteg 3 én gang til, i størrelse {UT_STØRRELSE}:")
    B = steg_3_på_nytt(A, C)
    ut_høyde, ut_bredde, _ = størrelser[UT_STØRRELSE]
    print(f"B er nå {B.shape[0]:,} × {B.shape[1]}   (bildene er {ut_høyde} × {ut_bredde})")


# =====================================================================
# 6. LAGRE RESULTATET
#    Pixelverdier må ligge mellom 0 og 255 for å kunne lagres,
#    så verdier utenfor klippes til nærmeste kant.
# =====================================================================

def lagre(verdier, filnavn):
    lagres = np.clip(np.round(verdier), 0, 255).astype(np.uint8)
    Image.fromarray(lagres.reshape(ut_høyde, ut_bredde, 3)).save(os.path.join(UTMAPPE, filnavn),
                                                                quality=98)

os.makedirs(UTMAPPE, exist_ok=True)

# grunnbildene, med strukket kontrast så mønsteret synes (laveste verdi svart, høyeste hvit)
for s in range(RANG):
    G = grunnbilde(B, C, s)
    lagre(255 * (G - G.min()) / (G.max() - G.min()), f"grunnbilde_{s + 1}.jpg")

# hvert foto bygd opp igjen: bilde_i ≈ Σ A[i,s]·G_s
for i in range(t):
    lagre(bilde(A, B, C, i), f"foto{i:04d}_rang{RANG}.jpg")


# =====================================================================
# 7. «WE LET TIME GO» — nye frames (bare hvis NYE_FRAMES = True)
#    B og C er låst. En ny frame er en ny rad i A, laget med Lagrange,
#    og bygd av de SAMME grunnbildene.
# =====================================================================

if NYE_FRAMES:
    tider = np.arange(-EKSTRA, n + EKSTRA + STEG / 2, STEG)
    print(f"\nLager {len(tider)} frames, fra t = {-EKSTRA} til t = {n + EKSTRA}:")
    for nr, t_ny in enumerate(tider):
        rad = ny_rad_i_A(t_ny, A)
        ny = sum(rad[s] * grunnbilde(B, C, s) for s in range(RANG))
        type_ = "ekte foto" if t_ny == int(t_ny) and 0 <= t_ny <= n else (
                "mellom" if 0 <= t_ny <= n else "utenfor")
        print(f"  t = {t_ny:>6.2f}  ({type_:9})  vekter {rad}")
        lagre(ny, f"ny_{nr:04d}_t{t_ny:+06.2f}.jpg")

print()
print(f"Ferdig. Bildene ligger i {UTMAPPE}")
