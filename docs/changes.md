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
