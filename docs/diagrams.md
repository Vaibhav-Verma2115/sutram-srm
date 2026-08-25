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
    class preprocess {
        <<module>>
        +scl_mask(scl) ndarray
        +apply_cloud_mask(arr, scl) tuple
        +tile_positions(h, w) list
        +hann_window(size, overlap) ndarray
    }
    class io_loaders {
        <<module>>
        +probe(payload) dict
        +overview(payload, size) ndarray
        +load_band_files(files, window) tuple
        +load_scene(uploads) tuple
    }
    class io_raster {
        <<module>>
        +read_bands(path) tuple
        +sr_profile(profile, scale) dict
        +write_cog(path, arr, profile) Path
        +check_footprint(src, dst) dict
    }
    class validate {
        <<module>>
        +degrade(hr, scale) ndarray
        +wald_protocol(branch, image) dict
        +consistency_check(sr, lr) dict
        +calibration_curve(conf, err) dict
        +evaluate_tasks(truth, cands) dict
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
    pipeline ..> io_raster : writes through
    pipeline ..> validate : checks with
    io_loaders ..> preprocess : feeds
    validate ..> trust_layer : reuses PSF
```

---

## 3. Data flow (inference) — as actually implemented

Verified against source. Two things a generic diagram gets wrong here: tiling happens
*inside* each branch that needs it (not as a separate stage), and the trust layer consumes
the fidelity and generative branches specifically, not all four.

```mermaid
flowchart TB
    A[/"S2 L2A GeoTIFF<br/>4 bands · B04 B03 B02 B08 @ 10 m"/]
    S[/"SCL band (optional)<br/>20 m"/]
    B["read_bands + to_reflectance<br/>÷10000, clip 0-1"]
    SCL["apply_cloud_mask<br/>classes 0,1,3,8,9,10<br/>nearest-neighbour to 10 m"]

    subgraph BR["Branches — each tiles internally"]
        E1["Bicubic<br/>whole-array interpolate"]
        E2["SEN2SR<br/>tiled_predict 128px + Hann"]
        E4["Ours<br/>tiled_predict 128px + Hann"]
        E3["LDSR-S2<br/>N stochastic samples"]
    end

    P["pipeline.run()"]
    G{{"confidence_map()<br/>fidelity.sr + generative.sr/sigma"}}
    C["consistency_check()<br/>per branch, vs input"]
    H["confidence [0,1]<br/>+ sigma"]
    I[/"6-band COG @ 2.5 m<br/>4 SR + sigma + confidence"/]
    J["check_footprint()<br/>drift = 0.00 m"]
    M[/"metrics.json<br/>+ cloud_fraction"/]

    A --> B --> SCL --> P
    S --> SCL
    P --> E1 & E2 & E4 & E3
    E2 -->|"fidelity"| G
    E3 -->|"sigma + sr"| G
    E1 & E2 & E3 & E4 --> C
    G --> H --> I --> J --> M
    C --> M
    SCL --> M

    classDef io fill:#e6f2e9,stroke:#1d7a3e
    classDef trust fill:#fdf6ec,stroke:#96650d
    class A,S,I,M io
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

## 6. Activity diagram — end-to-end processing

Every decision point the pipeline actually makes, including the guards that
turn silent failures into explicit messages.

```mermaid
flowchart TD
    Start([User has Sentinel-2 data]) --> Mode{Input type?}

    Mode -->|4 JP2 band files| Probe[probe metadata<br/>no pixel decode]
    Mode -->|4-band GeoTIFF| ReadTif[read_bands]
    Mode -->|Sample scene| ReadTif

    Probe --> Big{Larger than<br/>640 px?}
    Big -->|Yes| Ovw[overview thumbnail<br/>show window box]
    Ovw --> Pick[User positions window]
    Pick --> WinRead[decode window only]
    Big -->|No| FullRead[decode whole raster]
    WinRead --> Empty
    FullRead --> Empty
    ReadTif --> Empty

    Empty{Window has<br/>data?}
    Empty -->|under 2% non-zero| Err[/Refuse:<br/>move the window/]
    Err --> Pick
    Empty -->|Yes| DN

    DN{Integer L2A<br/>values?}
    DN -->|max > 1.5| Scale[divide by 10000<br/>clip to 0-1]
    DN -->|No| Cloud
    Scale --> Cloud

    Cloud{SCL band<br/>supplied?}
    Cloud -->|Yes| Mask[apply_cloud_mask<br/>resample 20m to 10m<br/>zero classes 0,1,3,8,9,10]
    Cloud -->|No| Branch
    Mask --> Branch

    Branch[/For each selected branch/]
    Branch --> Tile{Model needs<br/>128px tiles?}
    Tile -->|Yes| TP[tiled_predict<br/>Hann-feathered blend]
    Tile -->|No| Direct[whole-array inference]
    TP --> Collect
    Direct --> Collect

    Collect[Collect Predictions]
    Collect --> Trust[confidence_map<br/>consistency · SAM<br/>disagreement · sigma]
    Trust --> Cons[consistency_check<br/>per branch]
    Cons --> Write[write_cog<br/>4 SR + sigma + confidence]
    Write --> Verify{check_footprint<br/>drift = 0?}
    Verify -->|No| Fail[/Report geolocation error/]
    Verify -->|Yes| Out([2.5 m COG + metrics.json])

    classDef guard fill:#fdf6ec,stroke:#96650d
    classDef bad fill:#fbeaea,stroke:#b3341f
    classDef good fill:#e6f2e9,stroke:#1d7a3e
    class Empty,Verify,DN,Cloud,Big,Tile guard
    class Err,Fail bad
    class Out good
```

---

## 7. Sequence — upload to download

```mermaid
sequenceDiagram
    actor U as User
    participant App as superresolve.py
    participant L as io.loaders
    participant M as OursBranch
    participant T as trust.layer
    participant V as validate.wald
    participant IO as io.raster

    U->>App: upload B04, B03, B02, B08 (.jp2)
    App->>L: probe(B04)
    L-->>App: 10980×10980, EPSG:32644
    Note over App: no pixels decoded yet

    App->>L: overview(B04, 320)
    L-->>App: thumbnail from reduced JP2 level
    App-->>U: show granule + window box
    U->>App: position window

    App->>L: load_band_files(files, window)
    L->>L: verify all four same size
    L->>L: decode window per band
    L->>L: offset affine transform
    L-->>App: (4, 384, 384) + profile

    alt window is empty
        App-->>U: refuse — move the window
    else has data
        App->>App: detect DN, scale ÷10000
        App->>M: predict(lr)
        M->>M: tiled_predict, 128px + Hann
        M-->>App: Prediction(sr 1536×1536)

        App->>T: confidence_map(sr, lr)
        T->>T: PSF downsample → compare to input
        T->>T: spectral angle · ΔNDVI
        T-->>App: confidence map [0,1]

        App->>V: consistency_check(sr, lr)
        V-->>App: MAE 0.00150, SAM 0.576°

        App-->>U: swipe slider + metrics

        U->>App: Download
        App->>IO: write_cog(4 SR + sigma + confidence)
        IO->>IO: rescale transform, keep bounds
        IO-->>App: COG path
        App-->>U: 2.5 m GeoTIFF
    end
```

---

## 8. Development roadmap (Gantt)

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
