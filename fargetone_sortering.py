"""
FARGETONE-SORTERING
Følger notatet «Counting-Sort» linje for linje.

NOTATET SIER
    vi sorterer pixelene x(t,h,b) etter farge (ikke lysstyrke)
    for alle t, for samme h,b per sortering

    x(t,h,b) har R, G og B  =>  vi regner ut fargetonen

        h(t,h,b) = R·u_R + G·u_G + B·u_B,     u_θ = (cos θ, sin θ)

        fargetone = atan2(h_y, h_x) · 180/π   mod 360

    og sorteringen gjøres med Counting-Sort.

HVORFOR DET IKKE BLIR EN GRÅAKSE
    De tre enhetsvektorene står 120 grader fra hverandre og summerer derfor
    til nullvektoren. Legger du like mye lys i alle tre kanaler, flytter h seg
    langs den summen — altså ingen vei. Fargetonen er blind for lysstyrke.
    Tre tall går inn og to kommer ut, og det ene som kastes bort er nettopp
    gråaksen. Det er grunnen til at denne nøkkelen ikke gir en lysstyrkerampe.

HVA KODEN GJØR
    Et vindu på VINDU frames glir gjennom videoen. For hver posisjon:
        - regn ut fargetonen i hver pixel, i hver av de VINDU framene
        - Counting-Sort de VINDU fargene i hver pixelposisjon (h,b) for seg
        - lagre ÉN av dem: rang RANG
    Så flytter vinduet seg STEG frames fram, og vi gjør det igjen.

    Bevegelsen i originalen beholdes, fordi vinduet følger med i tid.
    Minnebruken er uavhengig av hvor lang videoen er.
"""

import os, math, time
import numpy as np
from PIL import Image, ImageOps


# =====================================================================
# INNSTILLINGER — det eneste du trenger å endre
# =====================================================================

BILDER = "mamma video"

VINDU   = 12   # hvor mange frames som sorteres sammen i hver posisjon
AVSTAND = 2    # avstand mellom framene INNE i vinduet. 2 = annenhver
RANG    = 6    # hvilken av de sorterte fargene som lagres, 1 til VINDU.
               # 1 = først på fargesirkelen fra STARTFARGE, VINDU = sist.
               # Midt på (VINDU//2) er roligst.

START = 0      # første posisjon vinduet står i
SLUTT = None   # siste posisjon. None = så langt det er frames igjen
STEG  = 1      # hvor mange frames vinduet flytter seg per utframe.
               # 1 = like mange utframes som originalen (samme hastighet)

STARTFARGE = 60      # hvor på fargesirkelen rekkefølgen begynner (grader)
MINSTE_LENGDE = 0.0  # 0 = sorter alt. Piksler med kortere |h| enn dette er
                     # nesten grå og lar seg ikke sortere meningsfullt; de
                     # blir stående som de er. Prøv 10 hvis det grynes.

MAKS_STØRRELSE = 960 # lengste side i utbildet

K = 63         # k i Counting-Sort(A, n, k): antall fargetone-bøtter minus én.
               # 360/(K+1) grader per bøtte. Målt på mamma video er k=63,
               # k=127 og k=255 visuelt like (74–88 % av pikslene helt
               # identiske, snittavvik 5 av 255), men k=63 er 3,3 ganger
               # raskere enn k=255. Sett den til 255 hvis du vil ha finest
               # mulig oppløsning og har tid.

PROSESSER = 4  # hvor mange kjerner som jobber samtidig. 1 = enkel og lett
               # å feilsøke. Hver prosess får sin egen sammenhengende bit
               # av videoen, så bildene blir nøyaktig de samme uansett.

STRIPE = 60000 # hvor mange piksler som sorteres om gangen (minnebruk)

MAKS_LAGER = 36 # hvor mange frames som holdes i minnet om gangen, per prosess.
                # Uavhengig av AVSTAND, se seksjon 4. 36 frames i 960 x 540
                # er ca. 150 MB per prosess.

NULLGRENSE = 1e-6   # kortere h enn dette regnes som grå


# Finn mappene (trenger ikke endres)
MAPPE = os.path.dirname(os.path.abspath(__file__))
BILDEMAPPE = None
sted = MAPPE
for _ in range(4):
    if os.path.isdir(os.path.join(sted, BILDER)):
        BILDEMAPPE = os.path.join(sted, BILDER)
        break
    sted = os.path.dirname(sted)
if BILDEMAPPE is None:
    raise SystemExit(f"Fant ikke mappen «{BILDER}» i nærheten av {MAPPE}.")
UTMAPPE = os.path.join(MAPPE, f"verk_{BILDER}_v{VINDU}a{AVSTAND}_rang{RANG}_fra{STARTFARGE}_k{K}")

assert VINDU <= 255, "C teller opptil VINDU, og lagres som uint8"
assert K <= 255, "nøkkelen lagres som uint8"


# =====================================================================
# 1. TEORIEN — rett fra notatet
# =====================================================================

# De tre kanalene som enhetsvektorer. 120 grader mellom dem fordi kanalene
# er likeverdige — og det er nettopp den jevne fordelingen som gjør at de
# summerer til nullvektoren.
U_R = (math.cos(math.radians(0)),   math.sin(math.radians(0)))     # ( 1   ,  0    )
U_G = (math.cos(math.radians(120)), math.sin(math.radians(120)))   # (-1/2 , +√3/2)
U_B = (math.cos(math.radians(240)), math.sin(math.radians(240)))   # (-1/2 , -√3/2)


def fargetone(farge):
    """Fargetonen til én farge, i grader 0-360. None hvis fargen er helt grå.

        h = R·u_R + G·u_G + B·u_B
        fargetone = atan2(h_y, h_x) · 180/π   mod 360

    Hver kanal er LENGDEN på en pil; vinkelen er retningen. atan2 og ikke
    atan, fordi atan bare dekker høyre halvdel av sirkelen — atan2 bruker
    fortegnene til x og y til å velge riktig kvadrant, og dekker hele.

    MERK rekkefølgen: vektoren skrives (x, y), men funksjonen heter
    atan2(y, x) — y først. De samme to tallene, motsatt rekkefølge.

    Er R = G = B opphever de tre pilene hverandre og h blir nullvektoren,
    som ikke har noen retning. Flyttall gir ikke eksakt null, bare noe
    forsvinnende kort, så vi tester lengden mot NULLGRENSE.
    """
    R, G, B = [float(v) for v in farge]
    hx = R * U_R[0] + G * U_G[0] + B * U_B[0]
    hy = R * U_R[1] + G * U_G[1] + B * U_B[1]
    if math.hypot(hx, hy) < NULLGRENSE:
        return None
    return math.degrees(math.atan2(hy, hx)) % 360


def counting_sort(A, nøkkel, k=K):
    """Counting-Sort for ÉN pixel, ordrett fra notatet.

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

    A er fargene og nøkkel[j] er fargetone-bøtta til A[j]. Hele fargen
    flyttes; bare oppslaget bruker nøkkelen. Vi skaper ingen nye farger.

    Notatet er 1-indeksert, så der plasseres først (linje 9) og trekkes fra
    etterpå (linje 10). Python er 0-indeksert, så vi trekker fra først og
    plasserer etterpå. Samme plass, bare forskjøvet indeksering.

    Linje 8 går BAKFRA. Det er det som gir stabilitet: to piksler med samme
    fargetone beholder rekkefølgen de hadde i tid.
    """
    n = len(A)
    Output = [None] * n
    C = [0] * (k + 1)
    for j in range(n):                      # linje 4-5: tell
        C[nøkkel[j]] += 1
    for i in range(1, k + 1):               # linje 6-7: "antall <= i"
        C[i] += C[i - 1]
    for j in range(n - 1, -1, -1):          # linje 8-10: plasser, bakfra
        C[nøkkel[j]] -= 1
        Output[C[nøkkel[j]]] = A[j]
    return Output


def grader_til_nøkkel(grader):
    """0-360 grader -> heltall 0..K, så Counting-Sort kan brukes.
    Vi roterer samtidig sirkelen så STARTFARGE havner på 0."""
    bøtte = np.floor(np.asarray(grader) / 360 * (K + 1)).astype(np.int64)
    return (bøtte - round(STARTFARGE / 360 * (K + 1))) % (K + 1)


# =====================================================================
# 2. NØKKELEN FOR ET HELT BILDE
#    Nøyaktig samme regnestykke som fargetone(), bare gjort på alle
#    pikslene samtidig i stedet for én om gangen.
# =====================================================================

def nøkkel_for(bilde):
    f = bilde.astype(np.float32)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]

    hx = R * U_R[0] + G * U_G[0] + B * U_B[0]
    hy = R * U_R[1] + G * U_G[1] + B * U_B[1]

    grader = np.degrees(np.arctan2(hy, hx)) % 360.0
    lengde = np.hypot(hx, hy)
    grader[lengde < NULLGRENSE] = 0.0        # grå: ingen fargetone

    return grader_til_nøkkel(grader).astype(np.uint8), lengde.astype(np.float32)


# =====================================================================
# 3. FINN FRAMENE
# =====================================================================

ALLE = [f for f in sorted(os.listdir(BILDEMAPPE))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))]
if len(ALLE) == 0:
    raise SystemExit(f"Fant ingen bilder i {BILDEMAPPE}.")

BREDDE_I_FRAMES = (VINDU - 1) * AVSTAND
SISTE_MULIGE = len(ALLE) - 1 - BREDDE_I_FRAMES
if SLUTT is None:
    SLUTT = SISTE_MULIGE
SLUTT = min(SLUTT, SISTE_MULIGE)
POSISJONER = list(range(START, SLUTT + 1, STEG))

# REKKEFØLGEN VI JOBBER I — ikke den utbildene lagres i.
#
# Vindu nr. q bruker framene q, q+AVSTAND, q+2·AVSTAND ... q+11·AVSTAND.
# Går vi fram én posisjon om gangen, deler vindu q og q+1 ingen frames
# når AVSTAND > 1 — de ligger forskjøvet. For å beholde alle framene
# imellom måtte lageret romme 11·AVSTAND frames, og med AVSTAND = 25 i
# 960 x 540 er det 1,1 GB per prosess.
#
# Men vindu q og vindu q+AVSTAND deler 11 av 12! Sorterer vi posisjonene
# etter (q mod AVSTAND, q) hopper vi AVSTAND fram hver gang, og lageret
# trenger aldri mer enn vinduet selv. Hvert utbilde er uavhengig av de
# andre, så hvilken rekkefølge vi regner dem ut i spiller ingen rolle —
# filnavnet følger posisjonen, ikke rekkefølgen.
REKKEFØLGE = sorted(range(len(POSISJONER)),
                    key=lambda j: (POSISJONER[j] % max(AVSTAND, 1), POSISJONER[j]))


# =====================================================================
# 4. LES BILDER — med et lager som følger vinduet
# =====================================================================

lager = {}          # frame-nummer -> (bildet, nøkkelen, lengden |h|)


def les(indeks):
    """Les frame nr. indeks, skaler ned, og regn ut nøkkelen én gang.

    Lageret har et fast tak på MAKS_LAGER frames. Blir det fullt, ryker den
    som ble brukt for lengst siden. Minnebruken er dermed den samme uansett
    hvor stor AVSTAND er — det er REKKEFØLGEN vi jobber i som sørger for at
    taket holder (se under).
    """
    if indeks in lager:
        lager[indeks] = lager.pop(indeks)      # flytt bakerst: nylig brukt
        return lager[indeks]
    bilde = ImageOps.exif_transpose(
        Image.open(os.path.join(BILDEMAPPE, ALLE[indeks]))).convert("RGB")
    if max(bilde.size) > MAKS_STØRRELSE:
        skala = MAKS_STØRRELSE / max(bilde.size)
        bilde = bilde.resize((int(bilde.width * skala) // 2 * 2,
                              int(bilde.height * skala) // 2 * 2), Image.LANCZOS)
    X = np.asarray(bilde, dtype=np.uint8)
    lager[indeks] = (X, *nøkkel_for(X))
    while len(lager) > MAKS_LAGER:
        lager.pop(next(iter(lager)))           # eldste ut
    return lager[indeks]


# =====================================================================
# 5. SORTER ÉN POSISJON — Counting-Sort i striper
#
#    Dette er counting_sort() over, kjørt på mange piksler samtidig.
#    C har én rad per piksel i stripen og K+1 kolonner — altså ett helt
#    C-array per piksel, akkurat som i notatet, bare mange ved siden av
#    hverandre. Derfor er STRIPE bare et minnevalg, ikke en algoritme.
#
#    C teller aldri høyere enn VINDU, så uint8 holder, og det er fire
#    ganger mindre å flytte på enn med heltall i full bredde.
# =====================================================================

def sorter_posisjon(X, N, rang):
    n, høyde, bredde, _ = X.shape
    P = høyde * bredde
    F = X.reshape(n, P, 3)
    Nf = N.reshape(n, P)
    ut = np.empty((P, 3), dtype=np.uint8)

    for s in range(0, P, STRIPE):
        e = min(s + STRIPE, P)
        m = e - s
        rad = np.arange(m)
        farger = F[:, s:e, :]
        nøkler = Nf[:, s:e]

        C = np.zeros((m, K + 1), dtype=np.uint8)      # linje 1-3
        for j in range(n):                            # linje 4-5: tell
            C[rad, nøkler[j]] += 1
        np.cumsum(C, axis=1, out=C)                   # linje 6-7

        Output = np.empty((n, m, 3), dtype=np.uint8)
        for j in range(n - 1, -1, -1):                # linje 8-10: bakfra
            kj = nøkler[j]
            C[rad, kj] -= 1
            Output[C[rad, kj], rad] = farger[j]

        ut[s:e] = Output[rang - 1]                    # vi trenger bare én rang
    return ut.reshape(høyde, bredde, 3)


# =====================================================================
# 6. ÉN UTFRAME
# =====================================================================

def lag_frame(teller, pos):
    filnavn = os.path.join(UTMAPPE, f"{teller:06d}.jpg")
    if os.path.exists(filnavn):
        return False

    deler = [les(pos + i * AVSTAND) for i in range(VINDU)]
    X = np.stack([d[0] for d in deler])
    N = np.stack([d[1] for d in deler])

    ut = sorter_posisjon(X, N, RANG)

    if MINSTE_LENGDE > 0:
        L = np.stack([d[2] for d in deler])
        grå = L.mean(axis=0) < MINSTE_LENGDE
        ut[grå] = X[RANG - 1][grå]

    Image.fromarray(ut).save(filnavn, quality=98)
    return True


def kjør_blokk(blokk):
    """Én sammenhengende bit av videoen. Hver prosess får sin egen, så
    lageret i les() fortsatt treffer på nabovinduene."""
    i0, i1 = blokk
    gjort = 0
    for teller in REKKEFØLGE[i0:i1]:
        if lag_frame(teller, POSISJONER[teller]):
            gjort += 1
    return gjort


# =====================================================================
# 7. KJØR
#    Ferdige bilder hoppes over, så du kan stoppe (ctrl + C) og starte igjen.
# =====================================================================

if __name__ == "__main__":

    # --- kontroll mot notatet: de tre enhetsvektorene ---
    print("u_R =", tuple(round(v, 4) for v in U_R))
    print("u_G =", tuple(round(v, 4) for v in U_G))
    print("u_B =", tuple(round(v, 4) for v in U_B))
    sx = U_R[0] + U_G[0] + U_B[0]
    sy = U_R[1] + U_G[1] + U_B[1]
    print(f"u_R + u_G + u_B = ({sx:+.1e}, {sy:+.1e})  -> nullvektoren, altså ingen gråakse")
    print()

    # --- kontroll mot eksempelet i notatet: (150, 171, 163) ---
    R, G, B = 150, 171, 163
    hx = R * U_R[0] + G * U_G[0] + B * U_B[0]
    hy = R * U_R[1] + G * U_G[1] + B * U_B[1]
    print(f"Eksempelet i notatet, rgb = ({R}, {G}, {B}):")
    print(f"  h = ({hx:.2f}, {hy:.2f})        notatet: (-17,00 ; 6,93)")
    print(f"  fargetone = {fargetone((R,G,B)):.1f} grader   notatet: 157,8")
    print()

    # --- kontroll mot landemerkene på sirkelen ---
    for navn, farge, fasit in [("rød",(255,0,0),0), ("gul",(255,255,0),60),
                               ("grønn",(0,255,0),120), ("cyan",(0,255,255),180),
                               ("blå",(0,0,255),240), ("magenta",(255,0,255),300)]:
        v = fargetone(farge)
        assert abs(v - fasit) < 1e-6, f"{navn} ga {v}, skulle vært {fasit}"
    assert fargetone((120,120,120)) is None, "grå skal ikke ha fargetone"
    print("Landemerkene stemmer: rød 0, gul 60, grønn 120, cyan 180, blå 240, magenta 300")
    print("Grå, svart og hvitt har ingen fargetone")
    print()

    # --- kontroll av Counting-Sort mot eksempelet i notatet ---
    A = [2, 5, 3, 0, 2, 3, 0, 3]
    assert counting_sort(A, A, k=5) == [0, 0, 2, 2, 3, 3, 3, 5]
    print(f"Counting-Sort({A}) = {counting_sort(A, A, k=5)}")
    print()

    print(f"{len(ALLE)} frames i «{BILDER}».")
    print(f"Vindu: {VINDU} frames, hver {AVSTAND}. — strekker seg over {BREDDE_I_FRAMES} frames")
    print(f"Posisjoner: {START} til {SLUTT}, steg {STEG}  ->  {len(POSISJONER)} utframes")
    print(f"Lagrer rang {RANG} av {VINDU}.   k = {K}  ({360/(K+1):.1f} grader per bøtte)")
    print()

    os.makedirs(UTMAPPE, exist_ok=True)

    # --- kontroll: er stripe-versjonen lik counting_sort()? ---
    deler = [les(POSISJONER[0] + i * AVSTAND) for i in range(VINDU)]
    Xk = np.stack([d[0] for d in deler]); Nk = np.stack([d[1] for d in deler])
    utk = sorter_posisjon(Xk, Nk, RANG)
    rng = np.random.default_rng(0)
    for y, x in zip(rng.integers(0, utk.shape[0], 30), rng.integers(0, utk.shape[1], 30)):
        fasit = counting_sort([tuple(int(v) for v in Xk[t, y, x]) for t in range(VINDU)],
                              [int(Nk[t, y, x]) for t in range(VINDU)])
        if fasit[RANG - 1] != tuple(int(v) for v in utk[y, x]):
            raise SystemExit(f"Stripe-versjonen er IKKE lik counting_sort() i pixel ({y},{x}).")
    print(f"Kontrollert mot counting_sort() på 30 piksler: likt")
    print(f"Bildestørrelse: {utk.shape[1]} × {utk.shape[0]}")
    print()

    lager.clear()
    t_start = time.time()

    if PROSESSER <= 1:
        for nr, teller in enumerate(REKKEFØLGE):
            lag_frame(teller, POSISJONER[teller])
            if (nr + 1) % 10 == 0 or nr == 0:
                per = (time.time() - t_start) / (nr + 1)
                igjen = (len(POSISJONER) - nr - 1) * per
                print(f"\r  {nr+1:>6} av {len(POSISJONER)}   {per:.2f} s per frame"
                      f"   ca {igjen/3600:.1f} t igjen   ", end="", flush=True)
    else:
        import multiprocessing as mp
        grense = np.linspace(0, len(REKKEFØLGE), PROSESSER + 1).astype(int)
        blokker = [(int(grense[i]), int(grense[i+1])) for i in range(PROSESSER)
                   if grense[i+1] > grense[i]]
        print(f"Kjører på {len(blokker)} kjerner, én sammenhengende blokk hver.")
        print("(ingen løpende teller når flere prosesser skriver samtidig —")
        print(" se hvor mange filer som ligger i utmappen underveis)")
        with mp.Pool(len(blokker)) as pool:
            pool.map(kjør_blokk, blokker)

    brukt = time.time() - t_start
    print()
    print(f"Ferdig på {brukt/60:.1f} min. {len(POSISJONER)} bilder i:")
    print(f"  {UTMAPPE}")
    print("Lag video med lag_video.py, eller:")
    print(f"  ffmpeg -framerate 25 -i '{UTMAPPE}/%06d.jpg' -c:v libx264 -pix_fmt yuv420p ut.mp4")
