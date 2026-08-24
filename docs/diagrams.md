# Mermaid diagrams — Sutram SRM

Paste any block into <https://mermaid.live>, a GitHub `.md` file, Notion, or Obsidian.
For a PPT: render at mermaid.live → Actions → PNG/SVG → drop into the slide.

---

## 1. System architecture (component view)

```mermaid
flowchart TB
    subgraph EXT["External data sources"]
        CDS[("Copernicus Data Space<br/>Sentinel-2 L2A · 10 m")]
        ZEN[("Zenodo<br/>SEN2VENuS v2.0.0")]
        HF[("HuggingFace<br/>pretrained weights")]
    end

    subgraph ENTRY["Entry points"]
        CLI["scripts/<br/>run_inference · evaluate<br/>train · build_dataset"]
        APP["app/demo.py<br/>Streamlit viewer"]
    end

    subgraph LIB["src/srm — core library"]
        PRE["preprocess/<br/>SCL mask · scale · tile"]
        MOD["models/<br/>4 branches + tiling"]
        TRU["trust/<br/>confidence fusion"]
        VAL["validate/<br/>Wald · calibration"]
        MET["metrics/<br/>PSNR SSIM SAM ERGAS"]
        TRN["train/<br/>losses · dataset · model"]
        IO["io/<br/>COG write · CRS preserve"]
        PIPE["pipeline.py<br/>orchestrator"]
    end

    subgraph OUT["Outputs"]
        COG[("6-band COG @ 2.5 m<br/>4 SR + sigma + confidence")]
        JSON[("metrics.json<br/>consistency · footprint")]
        CKPT[("checkpoints/best.pt")]
    end

    CDS --> CLI
    ZEN --> CLI
    HF --> MOD
    CLI --> PIPE
    PIPE --> PRE --> MOD --> TRU --> IO
    MOD --> VAL --> MET
    TRU --> IO
    IO --> COG
    VAL --> JSON
    CLI --> TRN
    TRN --> CKPT
    CKPT -.loads.-> MOD
    COG --> APP
    JSON --> APP

    classDef ext fill:#e8eef2,stroke:#5a8ca6,color:#16506b
    classDef lib fill:#16506b,stroke:#0d3547,color:#fff
    classDef out fill:#e6f2e9,stroke:#1d7a3e,color:#14532d
    class CDS,ZEN,HF ext
    class PRE,MOD,TRU,VAL,MET,TRN,IO,PIPE lib
    class COG,JSON,CKPT out
```

---

## 2. UML class diagram (the branch abstraction)

This is the design decision that makes the system extensible: every model implements one
interface, so the pipeline, trust layer and evaluation harness need no special-casing.

Note the `<<module>>` stereotypes: `pipeline` and `trust.layer` are Python modules of
functions, not classes. Drawing them as classes would send a reviewer looking for types that
do not exist in the source.

```mermaid
classDiagram
    class SRBranch {
        <<abstract>>
        +str name
        +int scale
        +predict(lr: ndarray) Prediction*
    }
    class Prediction {
        <<dataclass>>
        +ndarray sr
        +ndarray~None~ sigma
        +str name
        +float seconds
        +/has_uncertainty bool
    }
    class BicubicBranch {
        +str name = "bicubic"
        +int scale
        +predict(lr) Prediction
    }
    class Sen2SRBranch {
        -Module model
        -ModelLoader _loader
        +str device
        +str name = "SEN2SR"
        -_infer_tile(patch) ndarray
        +predict(lr) Prediction
    }
    class LdsrBranch {
        -SRLatentDiffusion model
        +str device
        +int n_samples
        +int steps
        +str name = "LDSR-S2"
        +predict(lr) Prediction
    }
    class OursBranch {
        -Module model
        +str device
        +int epoch
        +float val_psnr
        +str name = "Ours"
        -_infer_tile(patch) ndarray
        +predict(lr) Prediction
    }
    class pipeline {
        <<module>>
        +run(lr, branches, scale, fidelity, generative) dict
        +write_product(out_path, result, src_profile, scale) dict
        +write_metrics(out_path, result) Path
    }
    class trust_layer {
        <<module>>
        +psf_downsample(sr, scale, sigma) ndarray
        +lr_consistency(sr, lr, scale) ndarray
        +spectral_angle_map(a, b) ndarray
        +sam_consistency(sr, lr, scale) ndarray
        +ndvi(arr, red_idx, nir_idx) ndarray
        +delta_ndvi(sr, lr, scale) ndarray
        +branch_disagreement(sr_a, sr_b) ndarray
        +confidence_map(sr, lr, scale, sigma, other_branch) dict
    }
    class tiling {
        <<module>>
        +tiled_predict(lr, fn, tile, overlap, scale) ndarray
    }

    SRBranch <|-- BicubicBranch
    SRBranch <|-- Sen2SRBranch
    SRBranch <|-- LdsrBranch
    SRBranch <|-- OursBranch
    SRBranch ..> Prediction : returns
    LdsrBranch ..> Prediction : populates sigma
    pipeline ..> SRBranch : invokes
    pipeline ..> trust_layer : fuses via
    Sen2SRBranch ..> tiling : uses
    OursBranch ..> tiling : uses
```

---

## 3. Data flow (inference)

```mermaid
flowchart LR
    A[/"S2 L2A GeoTIFF<br/>4 bands @ 10 m"/]
    B["SCL cloud mask<br/>classes 0,1,3,8,9,10"]
    C["Scale to reflectance<br/>÷10000, clip 0-1"]
    D["Tile 128×128<br/>32 px overlap"]

    E1["Bicubic"]
    E2["SEN2SR<br/>hard constraint"]
    E3["LDSR-S2<br/>N samples"]
    E4["Ours"]

    F["Hann-feathered<br/>reassembly"]
    G{"Trust layer"}
    H["Confidence map<br/>[0,1]"]
    I[/"6-band COG @ 2.5 m"/]
    J["Footprint check<br/>drift = 0.00 m"]

    A --> B --> C --> D
    D --> E1 & E2 & E3 & E4
    E1 & E2 & E3 & E4 --> F --> G
    E3 -. "sigma" .-> G
    G --> H --> I
    I --> J

    classDef io fill:#e6f2e9,stroke:#1d7a3e
    classDef trust fill:#fdf6ec,stroke:#96650d
    class A,I io
    class G,H trust
```

---

## 4. Trust layer detail

The four independent signals and how they fuse. LR-consistency is the only one that is
*physical* rather than learned — it asks whether the output survives being re-observed by
the sensor.

```mermaid
flowchart TB
    SR[/"SR output<br/>2.5 m"/]
    LR[/"Original input<br/>10 m"/]
    OTHER[/"Second branch<br/>output"/]
    SIG[/"Diffusion sigma"/]

    PSF["PSF downsample<br/>Gaussian ≈ S2 MTF"]

    S1["LR-consistency<br/>|down(SR) − LR|"]
    S2["Spectral angle<br/>SAM per pixel"]
    S3["Branch disagreement<br/>normalised difference"]
    S4["Sampling variance"]

    FUSE{{"Weighted fusion<br/>0.4 / 0.2 / 0.2 / 0.2"}}
    CONF[/"Confidence [0,1]"/]
    CAL["Calibration check<br/>corr = −0.38"]

    SR --> PSF
    PSF --> S1
    PSF --> S2
    LR --> S1
    LR --> S2
    SR --> S3
    OTHER --> S3
    SIG --> S4

    S1 & S2 & S3 & S4 --> FUSE --> CONF --> CAL

    classDef phys fill:#16506b,stroke:#0d3547,color:#fff
    classDef out fill:#fdf6ec,stroke:#96650d
    class PSF,S1 phys
    class CONF,CAL out
```

---

## 5. Training pipeline

```mermaid
flowchart TB
    Z[("Zenodo<br/>SEN2VENuS 139 GB")]
    K["Select sites<br/>KUDALIAR = Telangana, India"]
    DL["Resumable download<br/>7.88 GB"]
    IDX["Read index.csv<br/>7269 patches"]
    PAIR["Load pair<br/>S2 10 m ↔ VENuS 5 m"]

    SCALE{"Scale mode"}
    X2["×2 native<br/>both real"]
    X4["×4: degrade input<br/>to 20 m via PSF"]

    FILT["Filter on contrast<br/>not zero-fraction"]
    SHARD["Stream float16 shards<br/>1000 patches each"]

    WARM["Warm start<br/>SEN2SRLite weights"]
    LOSS["L1 + LR-consistency<br/>+ SAM + gradient"]
    TRAIN["Train 40 epochs<br/>MPS · 33 s/epoch"]
    CK[("best.pt<br/>val PSNR 39.23")]

    Z --> K --> DL --> IDX --> PAIR --> SCALE
    SCALE -->|"scale=2"| X2
    SCALE -->|"scale=4"| X4
    X2 & X4 --> FILT --> SHARD
    SHARD --> TRAIN
    WARM --> TRAIN
    LOSS --> TRAIN
    TRAIN --> CK
    TRAIN -.->|"checkpoint each epoch"| CK

    classDef india fill:#16506b,stroke:#0d3547,color:#fff
    classDef out fill:#e6f2e9,stroke:#1d7a3e
    class K india
    class CK out
```

---

## 6. Sequence — a single inference run

```mermaid
sequenceDiagram
    actor U as Analyst
    participant CLI as run_inference.py
    participant P as pipeline.run()
    participant B as SRBranch(es)
    participant T as trust.confidence_map()
    participant IO as io.raster

    U->>CLI: --input scene.tif --branches sen2sr,ldsr
    CLI->>IO: read_bands()
    IO-->>CLI: array + profile (CRS, transform)
    CLI->>P: run(lr, branches)

    loop each branch
        P->>B: predict(lr)
        B-->>P: Prediction(sr, sigma?)
    end

    P->>T: confidence_map(sr, lr, sigma, other)
    T->>T: PSF downsample → compare to input
    T-->>P: {consistency, sam, disagreement, confidence}
    P-->>CLI: predictions + trust + consistency

    CLI->>IO: write_product() — 4 SR + sigma + confidence
    IO->>IO: rescale transform, keep bounds
    IO-->>CLI: COG path
    CLI->>IO: check_footprint()
    IO-->>CLI: ok = true, drift = 0.00 m
    CLI-->>U: product.tif + metrics.json
```

---

## 7. Development roadmap (Gantt)

```mermaid
gantt
    title Sutram SRM — delivery plan
    dateFormat YYYY-MM-DD
    axisFormat %d %b

    section Done
    M0 Foundation           :done, m0, 2026-08-23, 1d
    M1 Pipeline MVP         :done, m1, 2026-08-23, 2d
    M2 Own model trained    :done, m2, 2026-08-24, 1d

    section Next
    Downstream task eval    :active, t1, 2026-08-25, 2d
    Loss ablation           :t2, after t1, 1d
    LDSR full benchmark     :t3, after t1, 1d
    Multi-site training     :t4, after t2, 2d
    India ref validation    :crit, t5, after t3, 2d

    section Hardening
    Full-tile inference     :t6, after t4, 2d
    Test suite              :t7, after t4, 2d
    Final demo + PPT        :crit, t8, after t5, 2d
```
