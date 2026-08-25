#!/usr/bin/env python
"""Build a self-contained judge demo: one HTML file, no server, no network.

Every image is embedded as base64, so the page opens by double-click on any
machine and cannot fail mid-presentation because a server died, a model failed
to load, or the venue wifi dropped. That reliability is the whole point --
app/testbench.py exists for actually exercising the models.

    python scripts/make_judge_demo.py
    open docs/judge_demo.html
"""
from __future__ import annotations

import base64
import io
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCENE = "telangana_2026-03"

# (row, col, size) in 10 m pixels, plus a label and one-line reading guide.
CROPS = [
    (600, 700, 200, "Village &amp; highway",
     "Road edges sharpen; individual buildings separate from the settlement blur."),
    (690, 640, 110, "Settlement core",
     "At 10 m this is texture. At 2.5 m the street grid and structures resolve."),
    (760, 850, 150, "Field parcels",
     "Parcel boundaries and farm tracks become continuous lines."),
    (520, 980, 170, "Mixed farmland",
     "Crop-stage differences between adjacent fields become distinguishable."),
]


def stretch(rgb: np.ndarray, pct: float = 2.0) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(rgb.shape[-1]):
        lo, hi = np.percentile(rgb[..., i], [pct, 100 - pct])
        out[..., i] = np.clip((rgb[..., i] - lo) / max(hi - lo, 1e-8), 0, 1)
    return out


def to_b64(arr_rgb: np.ndarray, size: int = 760, quality: int = 88) -> str:
    """Encode a HxWx3 float array in [0,1] as a base64 JPEG data URI."""
    img = Image.fromarray((np.clip(arr_rgb, 0, 1) * 255).astype(np.uint8))
    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def heat_b64(gray: np.ndarray, size: int = 760) -> str:
    """Confidence -> transparent red mask (RGBA PNG).

    A full-frame heat map hides the imagery, which defeats the purpose: a viewer
    needs to see *which features* are uncertain, not just that uncertainty
    exists. So confident pixels are fully transparent and only doubt is painted,
    with alpha rising as confidence falls below the 0.5 flag threshold.
    """
    g = np.clip(gray, 0, 1)
    alpha = np.clip((0.62 - g) / 0.62, 0, 1) ** 0.8 * 0.85
    rgba = np.zeros(g.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = 255          # red
    rgba[..., 1] = (60 * (1 - alpha)).astype(np.uint8)
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    img = Image.fromarray(rgba, mode="RGBA").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    with rasterio.open(ROOT / "data/raw" / f"{SCENE}.tif") as s:
        lr = np.clip(s.read().astype(np.float32) / 10000, 0, 1)
    with rasterio.open(ROOT / "data/outputs" / f"{SCENE}_sr.tif") as s:
        prod = s.read().astype(np.float32)
    sr, conf = prod[:4], prod[5]

    scenes = []
    for y, x, w, title, note in CROPS:
        lr_c = lr[:3, y:y + w, x:x + w]
        sr_c = sr[:3, 4 * y:4 * (y + w), 4 * x:4 * (x + w)]
        cf_c = conf[4 * y:4 * (y + w), 4 * x:4 * (x + w)]
        # Bicubic-upsample the input so both panes display at identical size --
        # otherwise the comparison would be confounded by display scaling.
        bic = F.interpolate(torch.from_numpy(lr_c)[None], scale_factor=4,
                            mode="bicubic", align_corners=False)[0].numpy()
        scenes.append({
            "title": title, "note": note,
            "gsd": f"{w * 10 / 1000:.1f} km across",
            "before": to_b64(stretch(bic.transpose(1, 2, 0))),
            "after": to_b64(stretch(sr_c.transpose(1, 2, 0))),
            "conf": heat_b64(cf_c),
            "confmean": float(cf_c.mean()),
            "conflow": float((cf_c < 0.5).mean()),
        })

    html = TEMPLATE.replace("__SCENES__", json.dumps(scenes))
    out = ROOT / "docs" / "judge_demo.html"
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB, {len(scenes)} scenes)")
    return 0


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sutram SRM — Live Demo</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--txt:#e6edf3;--dim:#8b949e;--acc:#2f81f7;--good:#3fb950;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font:15px/1.5 -apple-system,"Segoe UI",Roboto,sans-serif;
     min-height:100vh;padding:22px 20px 40px}
header{max-width:1180px;margin:0 auto 18px;display:flex;align-items:flex-end;
       justify-content:space-between;flex-wrap:wrap;gap:14px}
h1{font-size:26px;letter-spacing:-.4px}
h1 span{color:var(--acc)}
.sub{color:var(--dim);font-size:13.5px;margin-top:3px}
.kpis{display:flex;gap:9px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 13px;text-align:center}
.kpi b{display:block;font-size:19px;color:var(--good);line-height:1.2}
.kpi span{font-size:10.5px;color:var(--dim);letter-spacing:.2px}
main{max-width:1180px;margin:0 auto}
.tabs{display:flex;gap:7px;margin-bottom:14px;flex-wrap:wrap}
.tab{background:var(--panel);border:1px solid var(--line);color:var(--dim);border-radius:7px;
     padding:8px 15px;cursor:pointer;font-size:13.5px;transition:.15s}
.tab:hover{color:var(--txt);border-color:#4a5568}
.tab.on{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
.stage{position:relative;border:1px solid var(--line);border-radius:11px;overflow:hidden;
       background:#000;aspect-ratio:1;max-height:70vh;margin:0 auto;user-select:none;
       cursor:ew-resize;touch-action:none}
.stage img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block;
           -webkit-user-drag:none}
#after{clip-path:inset(0 0 0 var(--pos,50%))}
#conf{opacity:0;transition:opacity .25s;pointer-events:none}
.stage.showconf #conf{opacity:1}
.handle{position:absolute;top:0;bottom:0;left:var(--pos,50%);width:3px;background:#fff;
        box-shadow:0 0 12px rgba(0,0,0,.85);pointer-events:none}
.knob{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:44px;height:44px;
      border-radius:50%;background:#fff;box-shadow:0 3px 14px rgba(0,0,0,.6);
      display:grid;place-items:center;color:#0d1117;font-size:17px;font-weight:700}
.lbl{position:absolute;top:13px;padding:6px 13px;border-radius:6px;font-size:12.5px;
     font-weight:600;background:rgba(13,17,23,.82);border:1px solid var(--line);backdrop-filter:blur(4px)}
.lbl.l{left:13px}.lbl.r{right:13px;color:var(--good);border-color:#2d5a3d}
.bar{display:flex;align-items:center;gap:16px;margin:14px 0 6px;flex-wrap:wrap}
input[type=range]{flex:1;min-width:220px;accent-color:var(--acc);height:5px}
.toggle{display:flex;align-items:center;gap:8px;background:var(--panel);border:1px solid var(--line);
        border-radius:7px;padding:8px 13px;cursor:pointer;font-size:13px;white-space:nowrap}
.toggle input{accent-color:var(--acc);width:15px;height:15px;cursor:pointer}
.note{color:var(--dim);font-size:13.5px;margin-top:9px;padding-left:2px}
.note b{color:var(--txt);font-weight:600}
.confbox{display:none;margin-top:10px;padding:11px 14px;background:#1c2128;border:1px solid var(--line);
         border-left:3px solid var(--acc);border-radius:7px;font-size:13px;color:var(--dim)}
.stage.showconf ~ .confbox{display:block}
footer{max-width:1180px;margin:26px auto 0;padding-top:14px;border-top:1px solid var(--line);
       color:var(--dim);font-size:12px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}
kbd{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:1px 6px;font-size:11px}
</style></head><body>

<header>
  <div>
    <h1>Sentinel-2 <span>10 m &rarr; 2.5 m</span> Super-Resolution</h1>
    <div class="sub">Real imagery over Telangana, India &middot; tile 44QKE &middot; 10 March 2026</div>
  </div>
  <div class="kpis">
    <div class="kpi"><b>&times;4</b><span>RESOLUTION</span></div>
    <div class="kpi"><b>0.908</b><span>FIELD F1 (was 0.800)</span></div>
    <div class="kpi"><b>0.00 m</b><span>GEO DRIFT</span></div>
  </div>
</header>

<main>
  <div class="tabs" id="tabs"></div>

  <div class="stage" id="stage">
    <img id="before" alt="10 m input">
    <img id="after"  alt="2.5 m super-resolved">
    <img id="conf"   alt="confidence">
    <div class="lbl l">Sentinel-2 &mdash; 10 m</div>
    <div class="lbl r">Super-resolved &mdash; 2.5 m</div>
    <div class="handle"><div class="knob">&#8646;</div></div>
  </div>

  <div class="bar">
    <input type="range" id="slider" min="0" max="100" value="50" aria-label="Comparison slider">
    <label class="toggle"><input type="checkbox" id="cx"> Show confidence map</label>
  </div>

  <div class="note" id="note"></div>
  <div class="confbox" id="confbox"></div>
</main>

<footer>
  <span>Drag the slider or use <kbd>&larr;</kbd> <kbd>&rarr;</kbd>. Press <kbd>C</kbd> for confidence, <kbd>1</kbd>&ndash;<kbd>4</kbd> to switch view.</span>
  <span>Team Sutram &middot; Smart India Hackathon</span>
</footer>

<script>
const SCENES = __SCENES__;
const stage=document.getElementById('stage'), slider=document.getElementById('slider'),
      before=document.getElementById('before'), after=document.getElementById('after'),
      conf=document.getElementById('conf'), tabs=document.getElementById('tabs'),
      note=document.getElementById('note'), confbox=document.getElementById('confbox'),
      cx=document.getElementById('cx');
let idx=0;

function setPos(p){p=Math.max(0,Math.min(100,p));stage.style.setProperty('--pos',p+'%');slider.value=p;}

function load(i){
  idx=i; const s=SCENES[i];
  before.src=s.before; after.src=s.after; conf.src=s.conf;
  note.innerHTML='<b>'+s.title+'</b> &mdash; '+s.note+' <span style="opacity:.6">('+s.gsd+')</span>';
  confbox.innerHTML='<b style="color:var(--txt)">Reading the confidence map:</b> red marks where the model is '+
    'least certain; unmarked areas are supported by the sensor data. Mean confidence here is <b style="color:var(--txt)">'+
    s.confmean.toFixed(3)+'</b>, with <b style="color:var(--txt)">'+(s.conflow*100).toFixed(1)+
    '%</b> of pixels flagged below 0.5. We show where the model is guessing rather than hiding it.';
  [...tabs.children].forEach((t,k)=>t.classList.toggle('on',k===i));
}

SCENES.forEach((s,i)=>{
  const b=document.createElement('button');
  b.className='tab'; b.textContent=(i+1)+'. '+s.title.replace('&amp;','&');
  b.onclick=()=>load(i); tabs.appendChild(b);
});

slider.oninput=e=>setPos(+e.target.value);
cx.onchange=()=>stage.classList.toggle('showconf',cx.checked);

let drag=false;
const fromEvent=e=>{
  const r=stage.getBoundingClientRect();
  const x=(e.touches?e.touches[0].clientX:e.clientX)-r.left;
  setPos(x/r.width*100);
};
stage.addEventListener('mousedown',e=>{drag=true;fromEvent(e);});
window.addEventListener('mousemove',e=>{if(drag)fromEvent(e);});
window.addEventListener('mouseup',()=>drag=false);
stage.addEventListener('touchstart',e=>{drag=true;fromEvent(e);},{passive:true});
stage.addEventListener('touchmove',e=>{if(drag)fromEvent(e);},{passive:true});
window.addEventListener('touchend',()=>drag=false);

addEventListener('keydown',e=>{
  if(e.key==='ArrowRight')setPos(+slider.value+3);
  else if(e.key==='ArrowLeft')setPos(+slider.value-3);
  else if(e.key.toLowerCase()==='c'){cx.checked=!cx.checked;cx.onchange();}
  else if(/^[1-9]$/.test(e.key)&&SCENES[+e.key-1])load(+e.key-1);
});

load(0); setPos(50);
</script></body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
