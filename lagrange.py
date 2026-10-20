"""
LAGRANGE-INTERPOLASJON PÅ BILDER — ÉN KODE, ALT STYRES MED INNSTILLINGENE
Følger notatet «Lagrange interpolasjon» i «Kodene mine».

TEORIEN (boksen i notatet)
    ℓ_i(x) = ∏ (x - x_j) / (x_i - x_j)      j = 0..n,  j ≠ i   («de andre enn x_i»)
    p_n(x) = Σ ℓ_i(x) · y_i                   i = 0..n
    x_0, ..., x_n er interpolasjonsnodene, og de må være distinkte.

HVORDAN JEG BRUKER DET («how I use it»)
    - bildene i mappen ligger på tidspunktene t = 0, 1, 2, ...  (t = bildets nummer i mappen)
    - nodene x_i er noen av disse bildene
    - for hver pixel(hb) har vi R, G og B
    - y_i er fargeverdien til pixel(hb) i bildet ved t = x_i
    - så vi trenger p_n^R, p_n^G, p_n^B for hver pixel
    - ℓ_0, ..., ℓ_n er de SAMME for alle pixler og alle farger

HVILKE BILDER SOM BLIR NODER (det eneste som skiller de forskjellige variantene)
    ANTALL_NODER = None  →  TANKEKART: hvert AVSTAND-te bilde i hele mappen er noder,
                            og det er ett polynom for hele videoen.
                            AVSTAND = 1 betyr at alle bildene er noder.
    ANTALL_NODER = 8     →  GLIDENDE: for hvert nye bilde brukes bare 8 bilder rundt x,
                            med AVSTAND mellom. Et lite tankekart for hvert bilde.
"""

import os
import numpy as np
from PIL import Image, ImageOps


# =====================================================================
# INNSTILLINGER — det eneste du trenger å endre
# =====================================================================

BILDEMAPPE_NAVN = "konsert"   # mappen med bildene (finnes ved siden av scriptet eller lenger opp)

# --- hvilke bilder som blir noder ---
AVSTAND = 380                 # hvor mange bilder det er mellom to noder (1 = nabobilder)
ANTALL_NODER = None           # None = TANKEKART (hvert AVSTAND-te bilde i hele mappen)
                              # et tall = GLIDENDE (bare så mange noder om gangen)
PLASSERING = 0.5              # bare for GLIDENDE: hvor x ligger, i node-steg fra første node
                              #   0.5 = mellom de to første nodene     (dobbelteksponering)
                              #   ANTALL_NODER = x etter siste node     (gjettes ut fra fortiden)

# --- hvilke x vi lager nye bilder for ---
SYNK_MED_LYD = True           # True: ett nytt bilde per bilde i mappen, x = 0, 1, 2, ..., siste.
                              #   Da blir videoen like lang som originalen, og nytt bilde k hører
                              #   til originalbilde k, altså samme øyeblikk i musikken.
                              #   Lag videoen med 30 bilder per sekund og lyden fra originalen:
                              #   python3 ../lag_video.py "lagrange_resultat/<mappen>" 30 "../../videoene/100_2495.MOV"
EKSTRA = 1                    # bare når SYNK_MED_LYD = False: hvor mange node-steg før første
                              #   og etter siste node vi går (ekstrapolasjon, der det glitcher)
STEG = 0.05                   # bare når SYNK_MED_LYD = False: hvor langt det er mellom to nye
                              #   bilder, i node-steg

# Hvilke bilder i mappen som brukes (nummer = plassen i mappen, sortert etter filnavn, første = 0)
FRA_BILDE = 0       # første bilde som tas med
TIL_BILDE = None    # siste bilde som tas med. None = helt til slutten av mappen
BRUK_HVERT = 1      # 1 = alle bildene, 3 = hvert 3., 30 = hvert 30., ...
                    # NB: med SYNK_MED_LYD må BRUK_HVERT være 1, ellers går bildet fortere enn lyden

MAKS_STØRRELSE = None         # lengste side i pixler. None = full størrelse. F.eks. 600 for en rask test.

# OPPSKRIFTER
#   Tankekart med testbildene:      "testbilder", AVSTAND = 1,   ANTALL_NODER = None, SYNK_MED_LYD = False
#   Konsert, synk med lyd (15 noder, smerteterskelen):
#                                   "konsert",    AVSTAND = 380, ANTALL_NODER = None, SYNK_MED_LYD = True
#   Konsert, veldig crazy (50 noder):
#                                   "konsert",    AVSTAND = 114, ANTALL_NODER = None
#   Glidende dobbelteksponering:    "konsert",    AVSTAND = 50,  ANTALL_NODER = 8, PLASSERING = 0.5
#   Glidende, gjettet fra fortiden: "konsert",    AVSTAND = 50,  ANTALL_NODER = 4, PLASSERING = 4

MAPPE = os.path.dirname(os.path.abspath(__file__))   # mappen dette scriptet ligger i
BILDEMAPPE = os.path.join(MAPPE, BILDEMAPPE_NAVN)
for _ in range(3):                                   # finnes den ikke ved siden av scriptet,
    if os.path.isdir(BILDEMAPPE):                    # leter vi oppover i mappene
        break
    BILDEMAPPE = os.path.join(os.path.dirname(os.path.dirname(BILDEMAPPE)), BILDEMAPPE_NAVN)


# =====================================================================
# 1. TEORIEN — rett fra boksen
# =====================================================================

def lagrange_basisfunksjon(i, x, noder):
    """ℓ_i(x) = ∏ (x - x_j) / (x_i - x_j) for alle j ≠ i
    Pensum kaller dette «Lagrangian basis functions» φ_i (forelesning 19–20, side 5).
    ℓ_i(x_i) = 1 og ℓ_i(x_j) = 0 for j ≠ i."""
    produkt = 1.0
    for j in range(len(noder)):
        if j != i:                                   # «de andre enn x_i»
            produkt = produkt * (x - noder[j]) / (noder[i] - noder[j])
    return produkt


def p(x, noder, y):
    """p_n(x) = Σ ℓ_i(x) · y_i"""
    total = 0
    for i in range(len(noder)):
        total = total + lagrange_basisfunksjon(i, x, noder) * y[i]
    return total


# =====================================================================
# 2. SJEKK MOT EKSEMPELET I NOTATET
#    pixel(hb): R = 10 ved t=0, R = 20 ved t=1, R = 15 ved t=2
# =====================================================================

print("Eksempelet fra notatet, x = 0.5:")
print(f"  ℓ_0(0.5) = {lagrange_basisfunksjon(0, 0.5, [0, 1, 2])}")             # 0.375
print(f"  ℓ_1(0.5) = {lagrange_basisfunksjon(1, 0.5, [0, 1, 2])}")             # 0.75
print(f"  ℓ_2(0.5) = {lagrange_basisfunksjon(2, 0.5, [0, 1, 2])}")             # -0.125
print(f"  p_2^R(0.5) = {p(0.5, [0, 1, 2], [10, 20, 15])}")  # 16.875
print()


# =====================================================================
# 3. FINN BILDENE I MAPPEN
#    De sorteres etter filnavn, så rekkefølgen i mappen blir rekkefølgen i tid.
#    Bilde nummer k (blant de valgte) ligger på t = k. Bildene lastes inn først når de trengs,
#    så en lang video ikke fyller opp minnet.
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
h, b = høyde, bredde
ANTALL_BILDER = len(BILDEFILER)
print(f"{ANTALL_BILDER} bilder i {BILDEMAPPE}, {h} × {b} pixler")


# =====================================================================
# 4. HVILKE BILDER ER NODER FOR TIDSPUNKTET x?
# =====================================================================

ALLE_NODER = list(range(0, ANTALL_BILDER, AVSTAND))      # t = 0, AVSTAND, 2·AVSTAND, ...

if ANTALL_NODER is None:
    # TANKEKART: de samme nodene for alle x
    if len(ALLE_NODER) < 2:
        raise SystemExit(f"AVSTAND = {AVSTAND} gir bare {len(ALLE_NODER)} node. Gjør AVSTAND mindre.")
    if len(ALLE_NODER) > 60:
        raise SystemExit(f"AVSTAND = {AVSTAND} gir {len(ALLE_NODER)} noder. Da blir ℓ-ene så store at "
                         f"bildet bare blir støy. Gjør AVSTAND større (prøv {ANTALL_BILDER // 15}).")
    print(f"TANKEKART: {len(ALLE_NODER)} noder ved t = {ALLE_NODER[0]}, {ALLE_NODER[1]}, ..., {ALLE_NODER[-1]}"
          f"  → grad n = {len(ALLE_NODER) - 1}")
else:
    # GLIDENDE: ANTALL_NODER bilder med AVSTAND mellom, plassert slik at x ligger
    # PLASSERING node-steg etter den første noden
    if (ANTALL_NODER - 1) * AVSTAND > ANTALL_BILDER - 1:
        raise SystemExit(f"{ANTALL_NODER} noder med {AVSTAND} bilder mellom trenger "
                         f"{(ANTALL_NODER - 1) * AVSTAND + 1} bilder, men mappen har {ANTALL_BILDER}.")
    print(f"GLIDENDE: {ANTALL_NODER} noder om gangen med {AVSTAND} bilder mellom, "
          f"x ligger {PLASSERING} node-steg etter første node  → grad n = {ANTALL_NODER - 1}")

def noder_for(x):
    if ANTALL_NODER is None:
        return ALLE_NODER
    # Eksempel: x = 2050, AVSTAND = 50, PLASSERING = 4  →  noder = [1850, 1900, 1950, 2000]
    # Ved begynnelsen og slutten skyves nodene inn, så vi aldri ber om et bilde som ikke finnes.
    lengde = (ANTALL_NODER - 1) * AVSTAND
    forste = int(np.floor(x - PLASSERING * AVSTAND))
    forste = max(0, min(forste, ANTALL_BILDER - 1 - lengde))
    return [forste + j * AVSTAND for j in range(ANTALL_NODER)]


# =====================================================================
# 5A. NYTT BILDE, PIXEL FOR PIXEL — nøyaktig slik det står i notatet
#     For hver pixel(hb) og hver farge: y_i = fargen i bilde i, så p_n(x).
#     Riktig, men tregt, så vi bruker den bare til å sjekke 5B.
# =====================================================================

def nytt_bilde_pixel_for_pixel(x, noder, bilder):
    h, b, _ = bilder[0].shape
    resultat = np.zeros((h, b, 3))
    for rad in range(h):
        for kol in range(b):
            for farge in range(3):                                   # 0 = R, 1 = G, 2 = B
                y = [bilde[rad, kol, farge] for bilde in bilder]     # y_0, ..., y_n for denne pixelen
                resultat[rad, kol, farge] = p(x, noder, y)
    return resultat


# =====================================================================
# 5B. NYTT BILDE, ALLE PIXLER PÅ ÉN GANG
#     Siden ℓ_i(x) er lik for alle pixler og farger, kan vi gi p() hele bildene som y_i:
#         nytt bilde = ℓ_0(x)·bilde_0 + ℓ_1(x)·bilde_1 + ... + ℓ_n(x)·bilde_n
#     Det er samme p() som over, bare raskere.
#     Bildene som er noder huskes, så de ikke åpnes på nytt for hvert nye bilde.
# =====================================================================

minne = {}

def hent_bilde(nr):
    if nr not in minne:
        minne[nr] = last_foto(BILDEFILER[nr])
    return minne[nr]

def nytt_bilde(x, noder):
    for nr in list(minne):                           # glem bilder som ikke er noder lenger
        if nr not in noder:
            del minne[nr]
    bilder = [hent_bilde(nr) for nr in noder]        # y_0, ..., y_n = bildene på nodene
    return p(x, noder, bilder)


# Sjekk at 5A og 5B gir samme svar (på et lite utsnitt, så det går fort)
sjekk_noder = noder_for(0)[:4]
sjekk_x = (sjekk_noder[0] + sjekk_noder[1]) / 2
utsnitt = [hent_bilde(nr)[:20, :20] for nr in sjekk_noder]
forskjell = np.abs(nytt_bilde_pixel_for_pixel(sjekk_x, sjekk_noder, utsnitt)
                   - p(sjekk_x, sjekk_noder, utsnitt)).max()
print(f"Sjekk: største forskjell mellom pixel-for-pixel og alle-på-én-gang = {forskjell}")
print()


# =====================================================================
# 6. HVILKE x VI LAGER NYE BILDER FOR
# =====================================================================

if SYNK_MED_LYD:
    # Ett nytt bilde per bilde i mappen: x = 0, 1, 2, ..., ANTALL_BILDER - 1.
    # Nytt bilde k hører til originalbilde k, altså samme øyeblikk i musikken.
    nye_tidspunkter = list(range(ANTALL_BILDER))
    print(f"SYNK MED LYD: {ANTALL_BILDER} nye bilder, x = 0, 1, ..., {ANTALL_BILDER - 1}")
    if ANTALL_NODER is None and ALLE_NODER[-1] < ANTALL_BILDER - 1:
        print(f"  ved hver node (hvert {AVSTAND}. bilde) er bildet helt likt originalen")
        print(f"  etter siste node (bilde {ALLE_NODER[-1]}) er det ekstrapolasjon: glitch-slutt på "
              f"{ANTALL_BILDER - 1 - ALLE_NODER[-1]} bilder (ca. {(ANTALL_BILDER - 1 - ALLE_NODER[-1]) / 30:.0f} sek)")
    navn_x = "synk_med_lyd"
else:
    siste = ALLE_NODER[-1] if ANTALL_NODER is None else ANTALL_BILDER - 1
    fra = -EKSTRA * AVSTAND
    til = siste + EKSTRA * AVSTAND
    nye_tidspunkter = np.round(np.arange(fra, til + STEG * AVSTAND / 2, STEG * AVSTAND), 6)
    print(f"{len(nye_tidspunkter)} nye bilder fra t = {fra:g} til t = {til:g}")
    navn_x = f"ekstra{EKSTRA}_steg{STEG}"

navn_noder = "tankekart" if ANTALL_NODER is None else f"glidende{ANTALL_NODER}_plassering{PLASSERING}"
navn_storrelse = "" if MAKS_STØRRELSE is None else f"_maks{MAKS_STØRRELSE}px"
UTMAPPE = os.path.join(MAPPE, "lagrange_resultat",
                       f"{BILDEMAPPE_NAVN}_avstand{AVSTAND}_{navn_noder}_{navn_x}{navn_storrelse}")


# =====================================================================
# 7. LAG DE NYE BILDENE
#    Pixelverdier må ligge mellom 0 og 255 for å kunne lagres,
#    så verdier utenfor klippes til nærmeste kant.
#    Filene nummereres (00000, 00001, ...) så de ligger i riktig rekkefølge.
#    Ferdige bilder hoppes over, så du kan stoppe (ctrl + C) og starte igjen.
# =====================================================================

os.makedirs(UTMAPPE, exist_ok=True)
print(f"Lagres i {UTMAPPE}")
vis_hver = 1 if len(nye_tidspunkter) <= 400 else 100     # ikke skriv ut tusenvis av linjer

for nr, x in enumerate(nye_tidspunkter):
    filnavn = os.path.join(UTMAPPE, f"{nr:05d}_t{x:+.2f}.jpg")
    if os.path.exists(filnavn):
        continue
    noder = noder_for(x)
    nytt = nytt_bilde(x, noder)

    if nr % vis_hver == 0:
        vekter = [lagrange_basisfunksjon(i, x, noder) for i in range(len(noder))]
        type_ = "interpolasjon" if noder[0] <= x <= noder[-1] else "EKSTRAPOLASJON"
        print(f"  bilde {nr:>5}  t = {x:>8.2f}  ({type_:14})  største |ℓ_i| = {max(abs(v) for v in vekter):>10.2f}")

    lagres = np.clip(np.round(nytt), 0, 255).astype(np.uint8)
    # Lagres først under et midlertidig navn og døpes om når filen er ferdig skrevet,
    # så det aldri blir liggende et halvferdig bilde hvis du stopper scriptet.
    Image.fromarray(lagres).save(filnavn + ".tmp", format="JPEG", quality=98)
    os.replace(filnavn + ".tmp", filnavn)

print()
print(f"Ferdig. Bildene ligger i {UTMAPPE}")
