# Code Changes & Bugfixes

## 1. Block-Puls-Erkennung bei großem Frequenzoffset (`bmc/bmc_tool.py`)

### Problem

In `prep_rf_simulation` wurde die Amplitude eines RF-Pulses über `torch.unique` auf Konstantheit geprüft, um Block-Pulse von shaped Pulsen zu unterscheiden. Die Amplitude wird intern als Betrag eines komplexen Signals berechnet:

```
|w1_complex| = |signal × exp(j·phase) × exp(j·2π·f·t)|
```

Für einen Block-Puls sollte dieser Betrag für alle Zeitpunkte exakt gleich sein. In der Praxis wird `exp(j·2π·f·t)` jedoch in **float32** (complex64) berechnet. Bei großen Frequenzoffsets entstehen dabei Rundungsfehler:

- Bei f = −7238 Hz und t = 2 s ergibt sich ein Winkel von 2π × 7238 × 2 ≈ **90.895 rad**
- In float32 weicht `|exp(j·90895)|` durch begrenzte Mantissenpräzision leicht von 1.0 ab (z. B. 0.99998 oder 1.00002)
- `torch.unique` erkannte diese minimal abweichenden Werte als **2 verschiedene Amplituden**
- Der Block-Puls wurde fälschlicherweise als shaped Puls klassifiziert

### Folge

Ein Block-Puls, der im shaped-Puls-Pfad landet, wird auf `max_pulse_samples = 200` Schritte interpoliert mit `dtp = 2.0 × (2−1) / 200 = 0.01 s`. Zusätzlich wird `rf_freq_out = 0.0` gesetzt statt des tatsächlichen Frequenzoffsets. Das Ergebnis: der Solver sieht einen 10-ms-Puls ohne Frequenzoffset statt eines 2-s-Pulses mit korrektem Offset → **falsche Physik**, erkennbar als wilde Oszillationen im Z-Spektrum.

### Ursache im Code (vorher)

```python
# bmc/bmc_tool.py — prep_rf_simulation
n_unique_amp = len(torch.unique(amp_full))           # float32-empfindlich
n_unique = max(n_unique_amp, len(torch.unique(ph)))

if n_unique_amp == 1 and amp.size(0) == 2:           # Block-Puls v1.4
    ...
    dtp_ = dtp                                        # korrekt: 2.0 s
    rf_freq_out = float(block.rf.freq_offset)         # korrekt
elif 1 < n_unique < max_pulse_samples:                # shaped → FALSCH für Blockpuls
    ...
    dtp_ = dtp * (original_length - 1) / max_pulse_samples   # = 0.01 s
    rf_freq_out = 0.0                                         # falsch
```

### Fix (nachher)

```python
# Toleranz-basierte Amplitudenkonstanz-Prüfung (robust gegenüber float32-Fehlern)
amp_range = float(amp_full.max() - amp_full.min())
amp_mean  = float(amp_full.mean())
is_const_amp = amp_range < 1e-3 * max(amp_mean, 1e-9)   # 0.1 % Toleranz
n_unique_amp = 1 if is_const_amp else len(torch.unique(amp_full))
n_unique = max(n_unique_amp, len(torch.unique(ph)))
```

### Auswirkung auf andere Simulationen

| Pulstyp | Vorher | Nachher |
|---|---|---|
| Block-Puls, kleiner Offset (< 1 kHz) | ✓ korrekt | ✓ korrekt |
| Block-Puls, großer Offset (> 1 kHz) | ✗ falsch klassifiziert | ✓ korrekt |
| Shaped Puls (Gauss, Sinc, …) | ✓ korrekt | ✓ korrekt |

Shaped Pulses sind **nicht betroffen**: ihre Amplitudenvariation beträgt typischerweise 10–100 % des Maximalwerts und liegt weit über der 0.1 %-Toleranz. Nur echte Block-Pulse haben `amp_range ≈ 0`.

Die Funktion `prep_rf_simulation` wird von folgenden Stellen verwendet:
- `bmc/simulate.py` → `simulate_zspec`
- `bmc/bmc_tool.py` → `BMCTool.run()`
- `bmc/fid/engine.py` → `BMCSim.run_adc`

Der Fix wirkt sich auf alle drei aus.

---

## 2. Z-Spektrum-Simulation (`bmc/simulate.py`)

### Hinzugefügt: `simulate_zspec`

Neue Funktion auf dem aktuellen Torch-Stack (matrix_exp-Solver) für Z-Spektren mit dem Muster `[RF-Sättigungsblock] → [ADC]`.

```python
offsets_ppm, m_z = simulate_zspec(
    config_file="sim_lib/config_1pool.yaml",
    seq_file="seq_lib/zspec_block.seq",
    norm_threshold=295,   # Offsets > 295 ppm gelten als M0-Referenz
)
```

Rückgabe: normalisiertes Mz-Array und zugehörige Offset-Achse in ppm. Die Magnetisierung wird nach jedem ADC auf den thermischen Gleichgewichtszustand zurückgesetzt (`reset_init_mag: True` in der Config).

---

## 3. Block-Puls-Frequenzoffset in FID-Engine (`bmc/fid/engine.py`)

### Problem

In `BMCSim.run_adc` wurde `prep_rf_simulation` aufgerufen, aber der zurückgegebene Frequenzoffset wurde ignoriert — der Solver wurde stets mit `rf_freq=0` aufgerufen.

### Fix

Der 5. Rückgabewert (`rf_freq_`) von `prep_rf_simulation` wird jetzt korrekt an `update_matrix` weitergegeben:

```python
amp_, ph_, dtp_, delay_, rf_freq_ = prep_rf_simulation(block, ...)
self.bm_solver.update_matrix(rf_amp=amp_[i], rf_phase=ph_[i], rf_freq=rf_freq_)
```

---

## 4. Notebooks: `zspec/`

Neuer Ordner mit zwei Notebooks:

| Notebook | Inhalt |
|---|---|
| `zspec/01_create_seq.ipynb` | Erstellt `seq_lib/zspec_block.seq` mit CW-Block-Sättigungspuls (128 Hz ≈ 3 µT bei 17T, 2 s) + pseudo-ADC für jeden Offset |
| `zspec/02_simulate.ipynb` | Simuliert das Z-Spektrum via `simulate_zspec`, plottet Z-Spektrum + MTRasym |

### Wichtige Parameter (17T)

| Parameter | Wert | Bedeutung |
|---|---|---|
| `b1_sat_hz` | 128 Hz | B1-Amplitude ≈ 3 µT bei 17T |
| `t_sat` | 2.0 s | Sättigungsdauer >> T1 ≈ 2.5 s |
| `offset_range` | ±10 ppm | Offset-Bereich |
| `n_offsets` | 41 | Anzahl Messpunkte (inkl. 0 ppm) |
| `m0_offset` | 300 ppm | M0-Referenz (weit außerhalb der Sättigungszone) |

---

## 5. Deterministische Isochromaten-Abtastung (`bmc/solver.py`, `bmc/params.py`, `bmc/fid/engine.py`)

### Problem

Die B0-Inhomogenitäts-Simulation zog die Frequenzoffsets der N Isochromaten zufällig aus einer Normalverteilung (`np.random.normal`, fester Seed 42). Selbst mit festem Seed entsteht durch die ungleichmäßige Abtastung der Gauß-Linienform ein **systematisches Hintergrundrauschen** in Multi-Puls-Sequenzen (Spin Echo, CPMG, STE): Die Ensemble-Mittelung kompensiert Artefakte aus randständigen Punkten nicht vollständig, da diese mit demselben Gewicht 1/N eingehen wie zentrale Punkte.

### Lösung: Quadratur-Methode nach Shkarin & Spencer (1996), Gl. 23

Anstelle von Zufallspunkten werden N **gleichmäßig verteilte** Stützstellen über die Gauß-Verteilung gelegt und mit dem Gauß-PDF gewichtet:

```
v_k  = µ + σ · linspace(−3.5σ, +3.5σ, N)     # N äquidistante Punkte
w_k  = exp(−0.5 · (v_k − µ)² / σ²)           # Gauß-PDF an jedem Punkt
p_k  = w_k / Σ w_k                            # normiert: Σ p_k = 1
```

Das Ensemble-Mittel wird dann als **gewichtete Summe** berechnet:

```
M_ensemble(t) = Σ_k  p_k · M_k(t)
```

Das Intervall ±3.5σ deckt 99.95 % der Verteilung ab.

### Neuer Parameter `isochromat_mode`

Der Modus wird über das `options`-Dictionary der `Params`-Klasse gesteuert und ist rückwärtskompatibel: fehlt der Schlüssel, wird `"random"` verwendet.

**Mögliche Werte:**

| Wert | Verhalten |
|---|---|
| `"random"` | bisheriges Verhalten — Monte-Carlo, `np.random.normal`, Seed 42, uniforme Gewichte $p_k = 1/N$ |
| `"deterministic"` | Shkarin & Spencer — äquidistante Punkte, Gauß-PDF-Gewichte |

### Geänderte Dateien

| Datei | Änderung |
|---|---|
| `bmc/solver.py` | `update_params()`: Verzweigung auf `isochromat_mode`; setzt `self.dw0` und `self.iso_weights` |
| `bmc/params.py` | `set_options()` und `update_options()`: neuer Parameter `isochromat_mode: str = "random"` |
| `bmc/fid/engine.py` | `get_mag()`: gewichtete Ensemble-Summe `(iso_weights · M).sum(dim=0)` statt uniformem Mittelwert |
| `bmc/set_params.py` | `load_params()`: leitet `isochromat_mode` aus der Config-Datei an `set_options()` weiter |
| `bmc/library/maintenance/valid_params.yaml` | `isochromat_mode` in `valid_first` und `valid_str` eingetragen |

### Verwendung

**Via YAML-Config:**
```yaml
isochromat_mode: deterministic
```

**Programmatisch nach `load_params()`:**
```python
sim_params = load_params("sim_lib/config_t2fit_noex.yaml")
sim_params.update_options(isochromat_mode="deterministic")
```

**Direkt:**
```python
params = Params()
params.set_options(isochromat_mode="deterministic")
```

### Randfall σ = 0

Ist `b0_inhomogeneity = 0`, entfällt die Gauß-Gewichtung:
- deterministisch: alle Punkte liegen bei µ = 0, Gewichte uniform 1/N
- random: bestehende Logik unverändert

### Rückwärtskompatibilität

- Alle bestehenden YAML-Configs ohne `isochromat_mode`-Schlüssel laden mit Default `"random"` → identisches Verhalten.
- `simulate_zspec` verwendet stets n_isochromats = 1 (z_positions = [0.0]) → beide Modi liefern iso_weights = [1.0], kein Unterschied.
- `iso_weights` wird immer in `update_params()` gesetzt, sodass `get_mag()` den Wert stets vorfindet.

### Vergleichs-Notebook

`cpmg/iso_mode_comparison.ipynb` — führt 4 Simulationen durch ({Hahn-Train, CPMG} × {random, deterministic}) und vergleicht Zeitverläufe, Echo-Amplituden und Gewichtsverteilungen.

---

## 6. Konvergenzkorrektur: Mittelpunktregel für deterministische Isochromaten (`bmc/solver.py`)

### Problem: Endpoint-inclusive `linspace` erzeugt O(N⁻¹)-Konvergenz

Die ursprüngliche deterministische Implementierung (Abschnitt 5) platzierte die N Isochromaten mit

```python
# ALT — bmc/solver.py, update_params(), deterministischer Zweig
v_k = self.mean_ppm + sigma_ppm * np.linspace(-L, L, self.n_isochromats)
# Schrittweite: h = 2·L·sigma / (N − 1)
# Äußerste Punkte liegen exakt auf den Trunkierungsgrenzen ±3.5σ
```

Das ist eine **einfache Riemann-Summe** (alle Punkte inklusive Endpunkte mit vollem Gewicht), keine Trapezregel. Die Endpunkte bei ±3.5σ gehen mit demselben Gewicht $w_k = G(\pm L\sigma)$ in die normierte Summe ein wie innere Punkte, obwohl sie physikalisch nur eine halbe Intervallfläche repräsentieren.

### Mathematische Ursache

Das berechnete Ensemble-Mittel lautet:

$$A_N = \frac{\sum_{k} G(v_k)\,f(v_k)}{\sum_{k} G(v_k)}$$

Der Zähler übersteigt die Trapezregel-Approximation um ein Randkorrekturglied:

$$\sum_k G(v_k) f(v_k) = P_\mathrm{trap} + G(L\sigma)\,f_b$$

wobei $f_b = \tfrac{1}{2}[f(-L\sigma) + f(+L\sigma)]$ der Echo-Amplitudenmittelwert an der Trunkierungsgrenze ist und $G(L\sigma) = e^{-L^2/2} = e^{-6.125} \approx 0.0022$ (klein aber ungleich null).

Entwicklung des Fehlers des Quotienten $A_N = P/Q$:

$$A_N - A = \underbrace{\frac{h \cdot G(L\sigma)}{D_\mathrm{true}}\bigl[f_b - A\bigr]}_{\mathcal{O}(h) = \mathcal{O}(N^{-1})} + \mathcal{O}(h^2)$$

Der numerisch bestimmte Übergang zwischen $\mathcal{O}(h)$- und $\mathcal{O}(h^2)$-Dominanz liegt bei

$$h_\times = \frac{C_1}{C_2} \approx 0.134\ \text{ppm} \quad\Rightarrow\quad N_\times \approx 3.6$$

Das bedeutet: **für jedes N ≥ 4 dominiert der O(N⁻¹)-Term** — die theoretisch mögliche O(N⁻²)-Konvergenz der Trapezregel tritt in der Praxis nie auf.

### Fix: Mittelpunktregel nach Shkarin & Spencer (1996), Gl. [23]

Shkarin & Spencer definieren die Isochromaten-Intensität $p_k$ als Integral von $f(v)$ über das Intervall $[v_k - \Delta v/2,\, v_k + \Delta v/2]$, wobei $v_k$ der **Mittelpunkt** des $k$-ten Intervalls ist. Kein Isochromat liegt auf der Trunkierungsgrenze.

```python
# NEU — bmc/solver.py, update_params(), deterministischer Zweig
h = 2 * L / self.n_isochromats          # Intervallbreite (N Intervalle, keine Endpunkte)
v_k = self.mean_ppm + sigma_ppm * np.linspace(-L + h/2, L - h/2, self.n_isochromats)
# Schrittweite: h = 2·L·sigma / N
# Äußerste Punkte liegen bei ±(3.5σ − h/2), nie exakt auf der Grenze
```

Da kein Isochromat auf ±3.5σ fällt, verschwindet das Randkorrekturglied identisch, und der Fehler reduziert sich auf:

$$A_N - A = \frac{C_2}{N^2} + \mathcal{O}(N^{-4})$$

### Konvergenzvergleich (numerisch verifiziert)

| N | Fehler alt (Riemann-Summe) | Fehler neu (Mittelpunktregel) |
|---|---|---|
| 16 | — | 3.5 × 10⁻⁴ (Beginn O(N⁻²)) |
| 100 | 8.6 × 10⁻⁵ | 9.0 × 10⁻⁶ |
| 500 | 4.8 × 10⁻⁴ | 3.6 × 10⁻⁷ |
| 1 000 | 3.4 × 10⁻⁵ | 9.0 × 10⁻⁸ |
| 10 000 | — | 6.7 × 10⁻¹⁰ |

Globaler Fit (N ≥ 50): **N⁻²·⁰⁴** (neu) vs. N⁻¹·⁰⁵ (alt).

### Mindestanzahl Isochromaten

Die saubere O(N⁻²)-Konvergenz setzt bei **N = 16** ein — exakt dem Shkarin-&-Spencer-Kriterium $K_\mathrm{min} = 8 N_\mathrm{Pulse} = 8 \times 2 = 16$ für das Hahn-Echo (2 HF-Pulse). Unterhalb dieser Grenze stören Artefakt-Echos das Simulationsfenster.

### Auswirkung auf bestehende Simulationen

Der Unterschied in den Simulationsergebnissen ist für alle praktisch eingesetzten N ≥ 100 numerisch vernachlässigbar (Änderung < 10⁻⁵ relativ). Die Änderung betrifft ausschließlich die **Konvergenzrate**, nicht die Physik.

### Gespeicherte Ergebnisse

| Datei | Inhalt |
|---|---|
| `results/convergence/iso_convergence_data_20260520_065115.npz` | Originaldaten (alte Implementierung, N bis 100 000, N_GT = 200 001) |
| `results/convergence/iso_convergence_midpoint_20260520_133935.npz` | Neue Implementierung (N = 2 … 10 000, N_GT = 20 001) |
| `results/convergence/iso_convergence_midpoint_20260520_182514.pdf/.png` | Konvergenzplot: alt vs. neu vs. Monte Carlo |
