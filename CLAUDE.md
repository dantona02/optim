# CLAUDE.md — Bloch-McConnell Simulation Framework

## Projekt-Überblick

Bloch-McConnell (BMC) Simulator für CEST-MRI-Sequenzen bei hohem Feld (primär 17T).
Kern: Lösung der BMC-Differentialgleichungen via `torch.linalg.matrix_exp` auf GPU/CPU.

---

## Projektstruktur

```
optim/
├── bmc/                    # Haupt-Simulationspaket
│   ├── bmc_tool.py         # prep_rf_simulation — RF-Puls-Aufbereitung
│   ├── solver.py           # BlochMcConnellSolver (matrix_exp, aktueller Stack)
│   ├── bmc_solver.py       # alter Padé-Solver (nicht mehr aktiv verwendet)
│   ├── simulate.py         # simulate_zspec, simulate_, simulate
│   ├── set_params.py       # load_params — YAML → Params-Objekt
│   ├── params.py           # Params-Datenklasse
│   ├── fid/
│   │   └── engine.py       # BMCSim — FID/zeitaufgelöste Simulation
│   └── utils/
│       ├── eval.py         # plot_z, calc_mtr_asym
│       ├── seq/write.py    # write_seq — PyPulseq → .seq-Datei
│       └── global_device.py  # GLOBAL_DEVICE (cuda/cpu)
├── sim_lib/
│   └── config_1pool.yaml   # Simulationsparameter (Pools, B0, Optionen)
├── seq_lib/                # .seq-Dateien (PyPulseq-Format v1.4)
├── zspec/                  # Z-Spektrum Notebooks
│   ├── 01_create_seq.ipynb # Sequenz erstellen
│   ├── 02_simulate.ipynb   # Z-Spektrum simulieren und plotten
│   └── 03_rabi_demo.ipynb  # Rabi-Oszillationen vs. Steady-State Demo
└── docs/
    └── changes.md          # Bugfix-Dokumentation
```

---

## Simulationsstack

### Aktueller Stack (verwenden)

```
BlochMcConnellSolver  (bmc/solver.py)
  → update_matrix(rf_amp, rf_phase, rf_freq)
  → solve_equation(mag, dtp)   # torch.linalg.matrix_exp(A * dtp)
```

Magnetisierungsvektor-Layout für N CEST-Pools:

```
m_vec = [Mx_w, My_w, Mz_w, ..., Mx_cest_N, My_cest_N, Mz_cest_N, M0_w, M0_cest_N]
         ^idx 0,1,2          ^idx 2*(N+1)-1                        ^mz_loc=2*(N+1)
```

Für 1 CEST-Pool: `mz_loc = 4` (Index von Mz_water in m_vec).

Tensor-Shape der Magnetisierung: `[n_iso, 1, vec_size, 1]` — bei Single-Slice `[1, 1, 6, 1]`.

### Alter Stack (nicht mehr verwenden)

`bmc/bmc_solver.py` — Padé-Approximation, nicht mehr aktiv. `run_1_4_0` / `run_1_3_0` in `bmc_tool.py` sind dead code.

---

## RF-Puls-Aufbereitung (`bmc/bmc_tool.py`)

### `prep_rf_simulation(block, max_pulse_samples)`

Bereitet einen PyPulseq-Block für den Solver auf. Gibt 5 Werte zurück:

```python
amp_, ph_, dtp_, delay_after_pulse, rf_freq_out = prep_rf_simulation(block, max_pulse_samples)
```

**Block-Puls-Erkennung** (kritisch):
- Amplitudenkonstanz wird mit **0.1%-Toleranz** geprüft, nicht via `torch.unique`
- Grund: `exp(j·2π·f·t)` in float32 liefert bei großen Frequenzoffsets (> ~1 kHz) Amplitude ≠ 1.0 durch Rundungsfehler
- `rf_freq_out` = `block.rf.freq_offset` für Block-Pulse (korrekte Physik)
- `rf_freq_out = 0.0` für shaped Pulse (Frequenzoffset bereits in der Phase kodiert)

**Seq-File-Versionen:**
- v1.4: `rf.t = [0., duration]` → 2 Samples → `dtp_ = duration`
- v1.3: N Samples mit konstantem dtp → `dtp_ = dtp * N`

---

## Z-Spektrum-Simulation (`bmc/simulate.py`)

### `simulate_zspec(config_file, seq_file, norm_threshold=295)`

Sequenz-Schema: `[RF-Block] → [pseudo-ADC]` pro Offset.

```python
offsets_ppm, m_z = simulate_zspec(
    config_file="sim_lib/config_1pool.yaml",
    seq_file="seq_lib/zspec_block.seq",
    norm_threshold=295,   # Offsets > 295 ppm = M0-Referenz
)
```

- Magnetisierung wird nach jedem ADC auf thermisches Gleichgewicht zurückgesetzt (`reset_init_mag: True`)
- Erster Offset > `norm_threshold` ppm wird als M0-Referenz verwendet → normiertes Mz

---

## Sequenzerstellung

### PyPulseq-Systemparameter (Standard für 17T)

```python
sys = pp.Opts(
    max_grad=500, grad_unit="mT/m",
    max_slew=1e9, slew_unit="T/m/s",
    rf_ringdown_time=0, rf_dead_time=0,
    rf_raster_time=1e-6,
    gamma=42576400,        # Hz/T (= 42.576 MHz/T)
    grad_raster_time=1e-6,
)
```

### Frequenzoffset-Berechnung

```python
GAMMA_HZ = sys.gamma * 1e-6   # = 42.576  [MHz/T]
LARMOR   = B0 * GAMMA_HZ      # = 723.8   bei 17T  [MHz] = [Hz/ppm]
offset_hz = offset_ppm * LARMOR  # z.B. 10 ppm → 7238 Hz
```

Einheiten: `LARMOR` ist numerisch in MHz, aber da ppm = 10⁻⁶ gilt, ergibt `ppm × MHz = Hz`. Kein Fehler.

### Block-Sättigungspuls (CEST)

```python
rf_sat = pp.make_block_pulse(
    flip_angle=b1_sat_hz * t_sat * 2 * np.pi,
    system=sys,
    duration=t_sat,
    freq_offset=offset_hz,
    phase_offset=0,
)
```

---

## Config-Datei (`sim_lib/config_1pool.yaml`)

```yaml
water_pool: {f: 1, t1: 2.5, t2: 0.071}   # T2=71ms, R2=14.1 Hz

cest_pool:
  amide: {f: 0.00064865, t1: 1.3, t2: 0.1, k: 150, dw: 8}
  # f: Pool-Größe (relativ zu Wasser)
  # k: Austauschrate [Hz] → Wasser
  # dw: chemische Verschiebung [ppm]

b0: 17
gamma: 267.5153   # rad/(µT·s) — NICHT Hz/T!
reset_init_mag: True
max_pulse_samples: 200
```

**Gamma-Konvention:** In der Config ist `gamma = 267.5153 rad/(µT·s)` (gyromagnetisches Verhältnis, nicht geteilt durch 2π). In PyPulseq ist `sys.gamma = 42576400 Hz/T`. Beides beschreibt dasselbe, unterschiedliche Einheiten.

---

## Physikalische Richtgrößen (17T)

| Größe | Wert |
|---|---|
| Larmorfrequenz | 723.8 MHz |
| 1 µT B1 | ≈ 42.6 Hz |
| Typische CEST-Sättigung | 1–5 µT = 43–213 Hz |
| Sättigungsdauer (Steady-State) | t_sat >> T1 ≈ 2.5 s |
| T2 Wasser | 71 ms |
| Amid-Pool | dw = 8 ppm, k = 150 Hz |

---

## Rabi-Oszillationen vs. Steady-State

Das Regime hängt von zwei Zeitskalen ab:

| Bedingung | Effekt |
|---|---|
| `t_sat << 1/k` (~7 ms) | kein CEST-Dip (zu wenig Austauschzyklen) |
| `t_sat << T2` (71 ms) | Rabi-Oszillationen sichtbar (kein Steady-State) |
| `t_sat > 1/k` | CEST-Dip entwickelt sich |
| `t_sat >> T2` | glatter Lorentzian (Steady-State) |

Analytische Mz ohne Relaxation:

```
Ω_eff = √(ω₁² + Δω²)
θ = arctan(ω₁/Δω)
Mz = cos²θ + sin²θ · cos(Ω_eff · t_sat)
```

---

## Bekannte Fallstricke

### 1. Notebook-Kernel-Zustand
Immer **Kernel → Restart & Run All** verwenden, nie einzelne Zellen nachträglich ausführen. Alte Variablenwerte (z.B. `t_sat = 2e-3`) überleben sonst im Kernel.

### 2. Seq-Datei nach Parameter-Änderung neu generieren
`02_simulate.ipynb` liest die `.seq`-Datei von Disk — wenn `01_create_seq.ipynb` nicht neu ausgeführt wurde, simuliert man alte Parameter.

### 3. `\n` in f-strings in Notebooks
Niemals echte Zeilenumbrüche in f-strings in Notebook-Cells schreiben. Stattdessen Titel als separate Variablen vorberechnen.

### 4. Float32-Präzision in `prep_rf_simulation`
Amplitude-Uniqueness-Check verwendet 0.1%-Toleranz. Nicht auf `torch.unique` zurückwechseln — das bricht Block-Puls-Erkennung bei Offsets > ~1 kHz.

### 5. `k: 0` in Config
Mit `k: 0` gibt es keinen chemischen Austausch → Z-Spektrum zeigt nur Wasser-Lorentzian, keinen CEST-Peak.
