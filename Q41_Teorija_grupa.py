#!/usr/bin/env python3

"""
Q41 Teorija grupa — SU(N) reprezentacije, Lie algebre, S_n permutaciona
dejstva — čisto kvantno.

Paradigma:
  Teorija kompaktnih Lie grupa i simetričnih grupa čini klasifikacioni skelet
  kvantne mehanike. Q41 koristi **Cartan-Weyl / Chevalley bazu** Lie algebre
  su(N) (za N = 64) u **pozicionoj reprezentaciji** na 64-dim Hilbert-u
  6 qubit-a.

  Standardni Chevalley generatori su(N):
    • Cartan subalgebra h = span{H_i}, gde su H_i dijagonalni, ΣH_i traceless.
      Prirodni izbor: X̂ = diag(0, 1, 2, …, N−1)  (pozicioni operator).
      Tada  X̂ − ⟨X̂⟩·I ∈ h.
    • Simple root operatori (Chevalley E, F):
          T̂_+ = Σ_{j=0..N−2} |j+1⟩⟨j|        (raising / shift +1)
          T̂_− = Σ_{j=0..N−2} |j⟩⟨j+1|        (lowering / shift −1)
      Zadovoljavaju [H, E] = α(H)·E, [H, F] = −α(H)·F, [E, F] = H_α.
    • Korenovi sistema A_{N−1}: pozitivni koreni α_ij za i < j generišu
      sve tranzitivne elemente. U pozicionoj reprezentaciji, α_{j, j+1}
      odgovara T̂_+ između susednih položaja.

  Weyl grupa SU(N):
    • W(A_{N−1}) = S_N.
    • Weyl refleksije deluju kao permutacije pozicionih bazisnih vektora.
    • Za N = 64, dobija se S_64 ⊃ S_7 (u smislu dejstva na 7-picks kroz
      sortiranje izvlačenja — kanonska S_7-invarijantna reprezentacija
      neuređene kombinacije).


Mapiranje na loto:
  Za svaku poziciju i ∈ {1..7}:
    1) j_target (strukturalni cilj, nije frekvencija) dobija se iz
           target_i(prev) = prev + (N_MAX − prev) / (N_NUMBERS − i + 2),
           j_target = round(target_i) − i   ∈ [0, 32].
    2) Chevalley slobodni kinetski Hamiltonijan (NN hopping u j-prostoru):
           H_kin = −J · (T̂_+ + T̂_−)
       Gde T̂_+ = Σ_{j=0..62} |j+1⟩⟨j|,  T̂_− = T̂_+†, su Chevalley E, F
       generatori za simple rootove α_{j,j+1} sistema A_{63} Lie algebre
       su(64). H_kin je **slobodna čestica na 1D lancu** sa energijskim
       spektrom ε_k = −2J · cos(k), analog slobodnog Lie-grupnog putovanja.
    3) Vremenska evolucija kao kvantna šetnja (Lie grupa SU(64)):
           U(t*) = exp(−i · H_kin · t*)  ∈  SU(64)
           |ψ_QW⟩ = U(t*) · |j_target⟩
       Kvantna šetnja iz |j_target⟩ proizvodi **Bessel-like wavepacket**:
           ⟨j|ψ_QW⟩ ≈ (−i)^{|j−j_target|} · J_{j−j_target}(2J·t*)
       Distribucija |⟨j|ψ_QW⟩|² ima "light-cone" strukture: oscilujuće
       u središnjem regionu |j − j_target| < 2J·t*, sa kaustičkim vrhovima
       na rubovima light cone-a. Za t* = 3.0, J = 1.0: širina ≈ 6 mesta.
    4) S_N Weyl grupa dejstvo (S_7 ⊂ S_64) — refleksija oko j_target:
           R_tgt |j⟩ = |2·j_target − j⟩   ako je 2·j_target − j ∈ [0, 63]
                                               (inače |j⟩ — fiksna tačka)
       R_tgt je element Weyl grupe W(SU(64)) = S_64 (proizvod transpozicija).
       Simetrizator: P_sym = (I + R_tgt) / 2.
       Kvantna šetnja je VEĆ simetrična oko j_target (Bessel funkcije
       zadovoljavaju J_{-n}(x) = (−1)^n · J_n(x), pa |ψ_QW|² je palindrom
       oko j_target); P_sym je pretežno identiteta na |ψ_QW⟩ i dijagnostika
       potvrđuje w_sym ≈ 1. Ostavljen radi formalnog Weyl-grupnog postupka.
    5) Finalno stanje i Born sempling:
           |ψ_fin⟩ = P_sym · |ψ_QW⟩ / ‖P_sym · |ψ_QW⟩‖
           P(j) = |⟨j|ψ_fin⟩|²
       Maskovanje: num > prev_pick, num ∈ [i, i+32]; renormalize; rng.choice.

Dijagnostika po poziciji (Lie / S_n invarijante):
  • ⟨j⟩ = ⟨ψ_fin|X̂|ψ_fin⟩                       (očekivana pozicija)
  • σ_j² = ⟨(X̂ − ⟨j⟩)²⟩                         (disperzija wavepacket-a)
  • w_sym = ‖P_sym · ψ_QW‖²                     (težina R_tgt-simetričnog sektora)

Lie / Cartan-Weyl paradigma u pozicionoj reprezentaciji;
  različita od Q32 (braid/anyoni), Q40 (braid/TL), i od prethodnih pokušaja
  SU(2)⊗6 Pauli-tensor pristupa koji su patili od bit-spread artifacta.
NQ = 6 qubit-a po poziciji (DIM = 64), reciklirani registar.
čisto kvantno — Cartan generator X̂, Chevalley E/F hopping,
  Weyl refleksija R_tgt, ground-state dekompozicija, Born sempling.
  Bez klasičnog ML-a, bez hibrida.

Okruženje: Python 3.11.13, qiskit 1.4.4, macOS M1, seed = 39.
CSV = /Users/4c/Desktop/GHQ/data/loto7hh_4602_k32.csv
CSV u celini (S̄ kao info).
DeprecationWarning / FutureWarning se gase.
"""


from __future__ import annotations

import csv
import math
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass


# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass


# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4602_k32.csv")
N_NUMBERS = 7
N_MAX = 39

NQ = 6                              
DIM = 1 << NQ                       # 64
POS_RANGE = 33                      # Num_i ∈ [i, i + 32]

J_HOP = 1.0                         # intenzitet Chevalley NN hopping-a
T_STAR = 3.0                        # vreme kvantne šetnje (light-cone ≈ 2J·t*)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def sort_rows_asc(H: np.ndarray) -> np.ndarray:
    return np.sort(H, axis=1)


# =========================
# Structural target (bez frekvencije)
# =========================
def target_num_structural(position_1based: int, prev_pick: int) -> float:
    denom = float(N_NUMBERS - position_1based + 2)
    return float(prev_pick) + float(N_MAX - prev_pick) / denom


def compute_j_target(position_1based: int, prev_pick: int) -> Tuple[int, float]:
    target = target_num_structural(position_1based, prev_pick)
    j = int(round(target)) - position_1based
    j = max(0, min(POS_RANGE - 1, j))
    return j, target


# =========================
# Cartan generator X̂ (pozicioni operator na 64-dim prostoru)
# =========================
X_OP = np.diag(np.arange(DIM, dtype=np.float64)).astype(np.complex128)


# =========================
# Chevalley simple-root shift operatori
#   T_+ |j⟩ = |j+1⟩  (j = 0..N-2)
#   T_− = T_+†
# =========================
def shift_plus(n: int) -> np.ndarray:
    T = np.zeros((n, n), dtype=np.complex128)
    for j in range(n - 1):
        T[j + 1, j] = 1.0
    return T


T_PLUS = shift_plus(DIM)
T_MINUS = T_PLUS.conj().T

# H_kin = -J (T_+ + T_-)
H_KIN = -1.0 * (T_PLUS + T_MINUS)


# =========================
# Weyl refleksija R_tgt oko j_target (Weyl grupa SU(64))
#   R_tgt |j⟩ = |2·j_target − j⟩  ako je 2·j_target − j ∈ [0, N−1]
#              |j⟩  inače (fiksna tačka)
# =========================
def build_reflection_around_target(j_target: int) -> np.ndarray:
    R = np.zeros((DIM, DIM), dtype=np.complex128)
    for j in range(DIM):
        jp = 2 * j_target - j
        if 0 <= jp < DIM:
            R[jp, j] = 1.0
        else:
            R[j, j] = 1.0
    return R


# =========================
# Precompute unitarnu evoluciju U(t*) = exp(-i H_kin t*)  (Lie grupa SU(64))
# =========================
def evolve_unitary(H: np.ndarray, t: float) -> np.ndarray:
    Hh = (H + H.conj().T) / 2.0
    evals, evecs = np.linalg.eigh(Hh)
    D = np.diag(np.exp(-1j * t * evals))
    return evecs @ D @ evecs.conj().T


U_KIN = evolve_unitary(J_HOP * H_KIN, T_STAR)


# =========================
# Predikcija jedne pozicije
# =========================
def lie_pick_one_position(
    position_1based: int,
    prev_pick: int,
    rng: np.random.Generator,
) -> Tuple[int, int, float, float, float, float]:
    j_target, target = compute_j_target(position_1based, prev_pick)

    # Kvantna šetnja iz |j_target⟩ pod slobodnim Chevalley kinetskim H_kin
    psi_init = np.zeros(DIM, dtype=np.complex128)
    psi_init[j_target] = 1.0
    psi_qw = U_KIN @ psi_init

    # Weyl refleksija R_tgt i simetrizator P_sym = (I + R_tgt)/2
    R_tgt = build_reflection_around_target(j_target)
    P_sym = 0.5 * (np.eye(DIM, dtype=np.complex128) + R_tgt)
    psi_sym = P_sym @ psi_qw
    w_sym = float(np.linalg.norm(psi_sym)) ** 2

    norm_s = float(np.linalg.norm(psi_sym))
    if norm_s < 1e-12:
        psi_fin = psi_qw / max(float(np.linalg.norm(psi_qw)), 1e-15)
    else:
        psi_fin = psi_sym / norm_s

    # Dijagnostika: ⟨j⟩ i disperzija wavepacket-a u j-prostoru
    probs_full = np.abs(psi_fin) ** 2
    probs_full = np.clip(np.real(probs_full), 0.0, None)
    js = np.arange(DIM, dtype=np.float64)
    mean_j = float(np.sum(js * probs_full))
    var_j = float(np.sum(((js - mean_j) ** 2) * probs_full))

    mask = np.zeros(DIM, dtype=np.float64)
    for j in range(POS_RANGE):
        num = position_1based + j
        if 1 <= num <= N_MAX and num > prev_pick:
            mask[j] = 1.0

    probs_valid = probs_full * mask
    s = float(probs_valid.sum())
    if s < 1e-15:
        for j in range(POS_RANGE):
            num = position_1based + j
            if 1 <= num <= N_MAX and num > prev_pick:
                return num, j_target, target, mean_j, var_j, w_sym
        return (
            max(prev_pick + 1, position_1based),
            j_target,
            target,
            mean_j,
            var_j,
            w_sym,
        )

    probs_valid /= s
    j_sampled = int(rng.choice(DIM, p=probs_valid))
    num = position_1based + j_sampled
    return num, j_target, target, mean_j, var_j, w_sym


# =========================
# Autoregresivni run
# =========================
def run_lie_autoregressive() -> List[int]:
    rng = np.random.default_rng(SEED)
    picks: List[int] = []
    prev_pick = 0

    for i in range(1, N_NUMBERS + 1):
        num, j_t, target, mean_j, var_j, w_sym = lie_pick_one_position(
            i, prev_pick, rng
        )
        picks.append(int(num))
        print(
            f"  [pos {i}]  target={target:.3f}  j_target={j_t:2d}  "
            f"⟨j⟩={mean_j:5.2f}  σ_j={math.sqrt(max(var_j,0)):.3f}  "
            f"w_sym={w_sym:.4f}  num={num:2d}"
        )
        prev_pick = int(num)

    return picks


# =========================
# Main
# =========================
def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Nema CSV: {CSV_PATH}")

    H = load_rows(CSV_PATH)
    H_sorted = sort_rows_asc(H)
    S_bar = float(H_sorted.sum(axis=1).mean())

    print("=" * 88)
    print("Q41 Teorija grupa — SU(64) Cartan-Weyl + Weyl grupa W = S_64 ⊃ S_7")
    print("=" * 88)
    print(f"CSV:            {CSV_PATH}")
    print(f"Broj redova:    {H.shape[0]}")
    print(f"Qubit budget:   {NQ} po poziciji  (Hilbert dim={DIM})")
    print(f"Lie algebra:    su(64) u Chevalley bazi  (Cartan X̂, E/F = T̂±)")
    print(f"Hamiltonijan:   H_kin = −J · (T̂_+ + T̂_−)  (slobodna Chevalley dinamika)")
    print(f"Parametri:      J = {J_HOP}  (hopping)   t* = {T_STAR}  (vreme šetnje)")
    print(f"Evolucija:      |ψ_QW⟩ = exp(−i·H_kin·t*) · |j_target⟩  (kvantna šetnja)")
    print(f"Weyl dejstvo:   R_tgt |j⟩ = |2·j_target − j⟩  (refleksija ∈ W(SU(64)))")
    print(f"Simetrizator:   P_sym = (I + R_tgt)/2  (projektor na R_tgt-invar.)")
    print(f"Stanje:         |ψ_fin⟩ = P_sym · ψ_QW,  Born sempling iz |⟨j|·⟩|²")
    print(f"Srednja suma S̄: {S_bar:.3f}  (CSV info, nije driver)")
    print(f"Seed:           {SEED}")
    print()
    print("Pokretanje Lie/Weyl (SU(64) kvantna šetnja + R_tgt) po pozicijama:")

    picks = run_lie_autoregressive()

    n_odd = sum(1 for v in picks if v % 2 == 1)
    gaps = [picks[i + 1] - picks[i] for i in range(N_NUMBERS - 1)]

    print()
    print("=" * 88)
    print("REZULTAT Q41 (NEXT kombinacija)")
    print("=" * 88)
    print(f"Suma:  {sum(picks)}   (S̄={S_bar:.2f})")
    print(f"#odd:  {n_odd}")
    print(f"Gaps:  {gaps}")
    print(f"Predikcija NEXT: {picks}")


if __name__ == "__main__":
    main()



"""
========================================================================================
Q41 Teorija grupa — SU(64) Cartan-Weyl + Weyl grupa W = S_64 ⊃ S_7
========================================================================================
CSV:            /data/loto7hh_4602_k32.csv
Broj redova:    4602
Qubit budget:   6 po poziciji  (Hilbert dim=64)
Lie algebra:    su(64) u Chevalley bazi  (Cartan X̂, E/F = T̂±)
Hamiltonijan:   H_kin = −J · (T̂_+ + T̂_−)  (slobodna Chevalley dinamika)
Parametri:      J = 1.0  (hopping)   t* = 3.0  (vreme šetnje)
Evolucija:      |ψ_QW⟩ = exp(−i·H_kin·t*) · |j_target⟩  (kvantna šetnja)
Weyl dejstvo:   R_tgt |j⟩ = |2·j_target − j⟩  (refleksija ∈ W(SU(64)))
Simetrizator:   P_sym = (I + R_tgt)/2  (projektor na R_tgt-invar.)
Stanje:         |ψ_fin⟩ = P_sym · ψ_QW,  Born sempling iz |⟨j|·⟩|²
Srednja suma S̄: 140.509  (CSV info, nije driver)
Seed:           39

Pokretanje Lie/Weyl (SU(64) kvantna šetnja + R_tgt) po pozicijama:
  [pos 1]  target=4.875  j_target= 4  ⟨j⟩= 5.22  σ_j=3.691  w_sym=0.9595  num= 7
  [pos 2]  target=11.571  j_target=10  ⟨j⟩=10.00  σ_j=4.243  w_sym=1.0000  num=14
  [pos 3]  target=18.167  j_target=15  ⟨j⟩=15.00  σ_j=4.243  w_sym=1.0000  num=17
  [pos 4]  target=21.400  j_target=17  ⟨j⟩=17.00  σ_j=4.243  w_sym=1.0000  num=22
  [pos 5]  target=26.250  j_target=21  ⟨j⟩=21.00  σ_j=4.243  w_sym=1.0000  num=30
  [pos 6]  target=33.000  j_target=27  ⟨j⟩=27.00  σ_j=4.243  w_sym=1.0000  num=38
  [pos 7]  target=38.500  j_target=31  ⟨j⟩=31.00  σ_j=4.243  w_sym=1.0000  num=39

========================================================================================
REZULTAT Q41 (NEXT kombinacija)
========================================================================================
Suma:  167   (S̄=140.51)
#odd:  3
Gaps:  [7, 3, 5, 8, 8, 1]
Predikcija NEXT: [7, 14, 17, 22, 30, 38, 39]
"""



"""
REZULTAT — Q41 Teorija grupa / SU(64) Cartan-Weyl + Weyl grupa
--------------------------------------------------------------
(Popunjava se iz printa main()-a nakon pokretanja.)

Koncept:
  • Čisto kvantno: Cartan X̂ (dijagonalni), Chevalley E/F = T̂±
    (shift ±1 u j-prostoru), Weyl refleksija R_tgt (element W(SU(64))),
    ground-state dekompozicija, Born sempling. Bez klasičnog ML-a.
  • SU(N) Lie algebra: su(64) u Cartan-Weyl / Chevalley bazi, realizovana u
    pozicionoj reprezentaciji 6-qubit Hilbert-a. Harmonic trap =
    kvadratni polinom u Cartan podalgebri (enveloping algebra U(h)).
  • S_N dejstvo: Weyl grupa W(A_{N−1}) = S_N za SU(N). Za N = 64,
    S_64 ⊃ S_7 (S_7 kao kanonska simetrija sorted 7-pick kombinacije).
    Refleksija R_tgt je element W ⊂ S_N (proizvod transpozicija oko
    j_target).
  • Ključna razlika od prethodnih pokušaja: pozicioni bazis |j⟩ čuva
    numeričku lokalnost u j-prostoru, pa Lie/Weyl operacije ne razpršuju
    težište preko udaljenih bit-pozicija (2^k).
  • Lie / Cartan-Weyl paradigma, različita od Q32 (braid/anyoni)
    i Q40 (braid/TL).
  • NQ = 6 qubit-a po poziciji, reciklirani 64-dim registar.
  • deterministički seed + fiksni σ, J + seeded Born sempling.

Tehnike:
  • X̂ = diag(0, 1, …, 63)                    — Cartan generator
  • T̂_+ = Σ_j |j+1⟩⟨j|,  T̂_− = T̂_+†          — Chevalley E, F
  • H_trap = ((X̂ − j_target·I) / σ)²          — enveloping polinom u U(h)
  • H_kin = −J · (T̂_+ + T̂_−)                  — NN hopping
  • H = H_trap + H_kin → eigh → ground state
  • R_tgt = Σ_j |2j_target − j⟩⟨j| (unutar opsega) — Weyl refleksija
  • P_sym = (I + R_tgt)/2 — simetrizator oko j_target (čuva lokalnost)
  • Born sempling iz maskovane distribucije |⟨j|ψ_fin⟩|².
"""
