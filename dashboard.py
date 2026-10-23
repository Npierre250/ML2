#!/usr/bin/env python3
"""
dashboard.py - live web dashboard (stdlib only, no Flask needed).
Reads everything from the storage layer, so it works on HDFS and in local mode.

  python dashboard.py            ->  http://localhost:5000
"""
import json
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from anomaly import config
from anomaly.model import read_meta
from anomaly.storage import get_storage

store = get_storage()
state = {"mode": config.STORAGE, "pipeline": {}, "stats": {}, "scores": {}, "by_fault_type": {},
         "records": [], "yarn_apps": [], "model": None, "paths": {},
         "by_machine": {}, "history": [], "root_causes": {}, "alert_counts": {}, "alerts": []}


def run(cmd, timeout=8):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""


def prf(c):
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), **c}


def refresh():
    while True:
        try:
            raw = store.read_text(config.D_METRICS, "metrics.json")
            m = json.loads(raw) if raw else None
            age = None
            if m and m.get("updated_at"):
                age = (datetime.now() - datetime.fromisoformat(m["updated_at"])).total_seconds()
            fresh = age is not None and age < config.BATCH_SECONDS * 3 + 10
            total = m["total"] if m else 0

            recs = []
            for line in store.tail(config.D_ALL, 12):
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass

            apps = []
            if config.STORAGE == "hdfs":
                for line in run("yarn application -list 2>/dev/null").split("\n"):
                    p = line.split()
                    if ("IoT" in line or "Flink" in line) and len(p) >= 8:
                        apps.append({"id": p[0], "name": p[1], "state": p[7]})

            where = "HDFS" if config.STORAGE == "hdfs" else "local folder"
            state["pipeline"] = {
                "generator": {"ok": fresh, "msg": "15 machines streaming on port %d" % config.SOCKET_PORT if fresh else "No data flowing"},
                "processor": {"ok": fresh, "msg": "Rules + Autoencoder scoring" if fresh else "Not running"},
                "storage":   {"ok": total > 0, "msg": f"{total:,} records in {where}" if total else "Waiting for data..."},
                "dashboard": {"ok": True, "msg": "Dashboard is live!"},
            }
            state["stats"] = {"total": total, "critical": m["rule_flags"] if m else 0,
                              "ae": m["ae_anomalies"] if m else 0, "faults": m["true_faults"] if m else 0, "labelled": m["labelled"] if m else 0,
                              "batches": m["batches"] if m else 0, "age": None if age is None else round(age)}
            state["scores"] = {k: prf(m[k]) for k in ("rules", "ae")} if m and m.get("labelled") else {}
            state["by_fault_type"] = m["by_fault_type"] if m else {}
            state["by_machine"] = m["by_machine"] if m else {}
            state["history"] = m["history"] if m else []
            state["root_causes"] = m.get("root_causes", {}) if m else {}
            state["alert_counts"] = m.get("alerts", {}) if m else {}
            alerts = []
            for line in store.tail(config.D_ALERTS, 8):
                try:
                    alerts.append(json.loads(line))
                except ValueError:
                    pass
            state["alerts"] = alerts
            state["records"] = recs
            state["yarn_apps"] = apps
            meta = read_meta(config.MODEL_PATH)
            state["model"] = ({"threshold": meta["threshold"], "sizes": meta["sizes"], **meta.get("extra", {})}
                              if meta else None)
            state["paths"] = {d: store.count_files(d) for d in (config.D_ALL, config.D_CRIT, config.D_AE, config.D_ALERTS, config.D_CKPT)}
        except Exception as e:                      # never let the dashboard thread die
            state["error"] = str(e)
        time.sleep(4)


HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Industrial IoT — Flink + Autoencoder</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg:#0A0F1E;--card:#1A2235;--border:#1E3A5F;--blue:#0EA5E9;--teal:#14B8A6;--green:#10B981;--orange:#F59E0B;--red:#EF4444;--text:#F1F5F9;--muted:#64748B;}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(14,165,233,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(14,165,233,.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;}
.w{position:relative;z-index:1;}
header{display:flex;align-items:center;justify-content:space-between;padding:1rem 2rem;background:rgba(17,24,39,.95);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100;}
.logo{width:40px;height:40px;background:linear-gradient(135deg,var(--blue),var(--teal));border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.3rem;}
.ht{font-size:1.05rem;font-weight:700;}
.hs{font-size:.72rem;color:var(--muted);font-family:'JetBrains Mono',monospace;}
.pill{display:flex;align-items:center;gap:.5rem;background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);border-radius:20px;padding:.3rem .8rem;font-size:.75rem;font-weight:600;color:var(--green);}
.dot{width:8px;height:8px;background:var(--green);border-radius:50%;animation:p 2s infinite;}
@keyframes p{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(1.3)}}
main{padding:1.5rem 2rem;display:flex;flex-direction:column;gap:1.5rem;}
.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;}
.step{background:var(--card);border:2px solid var(--border);border-radius:14px;padding:1.5rem;text-align:center;transition:all .5s;position:relative;overflow:hidden;}
.step::after{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--border);transition:all .5s;}
.step.running{border-color:var(--green);}
.step.running::after{background:linear-gradient(90deg,var(--green),var(--teal));}
.step.waiting{opacity:.6;}
.si{font-size:2.5rem;margin-bottom:.75rem;}
.sn{font-size:.9rem;font-weight:700;margin-bottom:.3rem;}
.sm{font-size:.7rem;color:var(--muted);min-height:2.5rem;display:flex;align-items:center;justify-content:center;}
.badge{display:inline-block;padding:.25rem .75rem;border-radius:20px;font-size:.65rem;font-weight:700;margin-top:.75rem;}
.br{background:rgba(16,185,129,.15);color:var(--green);border:1px solid rgba(16,185,129,.3);}
.bw{background:rgba(100,116,139,.15);color:var(--muted);border:1px solid rgba(100,116,139,.3);}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;}
.stat{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.25rem;position:relative;overflow:hidden;}
.stat::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.stat.bl::after{background:linear-gradient(90deg,var(--blue),transparent);}
.stat.gr::after{background:linear-gradient(90deg,var(--green),transparent);}
.stat.rd::after{background:linear-gradient(90deg,var(--red),transparent);}
.stat.tl::after{background:linear-gradient(90deg,var(--teal),transparent);}
.sl{font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:.5rem;}
.sv{font-size:2.2rem;font-weight:700;font-family:'JetBrains Mono',monospace;transition:all .3s;}
.sv.bl{color:var(--blue)}.sv.gr{color:var(--green)}.sv.rd{color:var(--red)}.sv.tl{color:var(--teal)}
.ss{font-size:.7rem;color:var(--muted);margin-top:.4rem;}
.panel{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden;}
.ph{display:flex;align-items:center;justify-content:space-between;padding:.85rem 1.25rem;border-bottom:1px solid var(--border);background:rgba(0,0,0,.2);}
.pt{font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);}
.pb{padding:1rem 1.25rem;}
.flow{display:flex;align-items:center;gap:.5rem;padding:1rem;}
.fs{flex:1;background:rgba(0,0,0,.3);border:1px solid var(--teal);border-radius:8px;padding:.75rem .5rem;text-align:center;}
.fi{font-size:1.4rem;margin-bottom:.3rem;}
.fn{font-size:.7rem;font-weight:600;}
.fsb{font-size:.6rem;color:var(--muted);margin-top:.2rem;}
.arr{color:var(--teal);font-size:1.2rem;flex-shrink:0;animation:fl 1.5s infinite;}
@keyframes fl{0%,100%{opacity:.4;transform:translateX(0)}50%{opacity:1;transform:translateX(3px)}}
.mid{display:grid;grid-template-columns:1fr 2fr;gap:1rem;}
.nodes{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;}
.node{background:rgba(0,0,0,.3);border:1px solid var(--border);border-radius:8px;padding:.75rem;text-align:center;}
.ni{font-size:1.5rem;margin-bottom:.3rem;}
.nn{font-size:.75rem;font-weight:600;color:var(--blue);font-family:'JetBrains Mono',monospace;}
.nr{font-size:.6rem;color:var(--muted);margin-top:.2rem;}
.ns{font-size:.65rem;color:var(--green);margin-top:.3rem;font-weight:600;}
.ar{display:flex;align-items:center;justify-content:space-between;padding:.75rem;border-radius:8px;border:1px solid var(--border);margin-bottom:.5rem;background:rgba(0,0,0,.2);}
.an{font-size:.85rem;font-weight:600;}
.ai{font-size:.65rem;color:var(--muted);font-family:'JetBrains Mono',monospace;}
.bg{padding:.25rem .65rem;border-radius:20px;font-size:.7rem;font-weight:700;background:rgba(16,185,129,.15);color:var(--green);border:1px solid rgba(16,185,129,.3);}
.tbl{width:100%;border-collapse:collapse;font-size:.75rem;}
.tbl th{text-align:left;padding:.5rem .75rem;font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);border-bottom:1px solid var(--border);}
.tbl td{padding:.6rem .75rem;border-bottom:1px solid rgba(30,58,95,.4);font-family:'JetBrains Mono',monospace;font-size:.7rem;}
.s0{color:var(--green)}.s1{color:var(--orange)}.s2,.s3{color:var(--red);font-weight:700}
.cy{background:rgba(239,68,68,.1);color:var(--red);border:1px solid rgba(239,68,68,.3);border-radius:4px;padding:.1rem .4rem;font-size:.6rem;}
.hg{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;}
.hr{display:flex;flex-direction:column;gap:.5rem;padding:.75rem;border-radius:8px;border:1px solid var(--border);background:rgba(0,0,0,.2);}
.hn{font-size:.7rem;color:var(--blue);font-family:'JetBrains Mono',monospace;}
.hd{font-size:.65rem;color:var(--muted);margin-top:.1rem;}
.hc{font-size:.85rem;font-weight:700;color:var(--teal);font-family:'JetBrains Mono',monospace;}
.empty{text-align:center;padding:2rem;color:var(--muted);}
.spin{display:inline-block;animation:sp 1s linear infinite;}
@keyframes sp{to{transform:rotate(360deg)}}
footer{text-align:center;padding:1rem;font-size:.7rem;color:var(--muted);border-top:1px solid var(--border);margin-top:.5rem;}
.cmp{display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;padding:1rem 1.25rem}
.cm{background:rgba(0,0,0,.25);border:1px solid var(--border);border-radius:10px;padding:1rem}
.cm.best{border-color:var(--green)}
.cm h4{font-size:.8rem;margin-bottom:.15rem}.cm .sub{font-size:.62rem;color:var(--muted);margin-bottom:.8rem}
.m{margin:.45rem 0}.m .l{display:flex;justify-content:space-between;font-size:.68rem;color:var(--muted)}
.m .l b{color:var(--text);font-family:'JetBrains Mono',monospace}
.bar{height:6px;background:rgba(100,116,139,.25);border-radius:4px;margin-top:.2rem;overflow:hidden}
.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--teal));transition:width .6s}
.cm.best .bar i{background:linear-gradient(90deg,var(--green),var(--teal))}
.hint{font-size:.68rem;color:var(--muted);padding:0 1.25rem 1rem}
.tyrow{display:flex;align-items:center;gap:.6rem;font-size:.7rem;margin:.45rem 0}
.tyrow .n{width:5.5rem;color:var(--muted);font-family:'JetBrains Mono',monospace}
.tyrow .bar{flex:1;margin:0}.tyrow .v{width:3rem;text-align:right;font-family:'JetBrains Mono',monospace}
.kv{display:grid;grid-template-columns:1fr 1fr;gap:.4rem .8rem;font-size:.72rem}
.kv span{color:var(--muted)}.kv b{font-family:'JetBrains Mono',monospace;text-align:right}
tr.flag td{background:rgba(245,158,11,.07)}
.tag{border-radius:4px;padding:.1rem .4rem;font-size:.6rem;font-weight:700}
.tg-ae{background:rgba(245,158,11,.15);color:var(--orange);border:1px solid rgba(245,158,11,.3)}
.tg-ok{color:var(--muted)}
.legend{display:flex;gap:1.2rem;font-size:.65rem;color:var(--muted);margin-top:.4rem}.legend i{display:inline-block;width:14px;height:3px;margin-right:.35rem;vertical-align:middle}
.mid2{display:grid;grid-template-columns:2fr 1fr;gap:1rem}
.mg{display:grid;grid-template-columns:repeat(3,1fr);gap:.35rem}
.cell{border-radius:6px;padding:.45rem .2rem;text-align:center;font-size:.62rem;font-family:'JetBrains Mono',monospace;color:#04121a;font-weight:600}
.lv{border-radius:4px;padding:.1rem .45rem;font-size:.6rem;font-weight:700}.lv-c{background:rgba(239,68,68,.15);color:var(--red);border:1px solid rgba(239,68,68,.35)}.lv-w{background:rgba(245,158,11,.15);color:var(--orange);border:1px solid rgba(245,158,11,.35)}
.tg-f{background:rgba(239,68,68,.1);color:var(--red);border:1px solid rgba(239,68,68,.3)}
@media(max-width:900px){.steps,.stats,.cmp,.mid,.hg{grid-template-columns:1fr 1fr}}
</style></head>
<body><div class="w">
<header>
  <div style="display:flex;align-items:center;gap:1rem">
    <div class="logo">⚡</div>
    <div><div class="ht">Industrial IoT — Real-time Autoencoder Anomaly Detection</div>
    <div class="hs">Ndagijimana Pierre · 0120000396 · Machine Learning II &amp; Artificial Neural Networks · Università degli Studi di Napoli Parthenope</div></div>
  </div>
  <div style="display:flex;align-items:center;gap:1rem">
    <div class="pill"><div class="dot"></div><span id="mode">LIVE</span></div>
    <div id="clk" style="font-family:'JetBrains Mono',monospace;font-size:.85rem;color:var(--muted)"></div>
  </div>
</header>
<main>
  <div><div style="font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:1rem">🔄 Pipeline status</div>
  <div class="steps" id="steps"></div></div>

  <div class="stats">
    <div class="stat bl"><div class="sl">Records processed</div><div class="sv bl" id="s1">0</div><div class="ss" id="s1s">stored in batches</div></div>
    <div class="stat gr"><div class="sl">True faults (ground truth)</div><div class="sv gr" id="s2">0</div><div class="ss">labels sent by the simulator (evaluation only)</div></div>
    <div class="stat rd"><div class="sl">Rule flags (any threshold)</div><div class="sv rd" id="s3">0</div><div class="ss">fixed limits: 75 °C · 20 mm/s · 80 dB</div></div>
    <div class="stat tl"><div class="sl">Autoencoder anomalies</div><div class="sv tl" id="s4">0</div><div class="ss">reconstruction error &gt; learned threshold</div></div>
  </div>

  <div class="panel">
    <div class="ph"><div class="pt">🧠 Which detector finds the faults? (scored against ground truth)</div></div>
    <div class="cmp" id="cmp"></div>
    <div class="hint">Precision = of the alerts raised, how many were real faults. Recall = of the real faults, how many were caught. The autoencoder learns normal sensor <i>relationships</i>; fixed thresholds only look at each sensor alone.</div>
  </div>

  <div class="panel"><div class="ph"><div class="pt">🎯 Detection rate by fault type</div></div>
    <div class="pb" id="ftypes"></div>
    <div class="hint">Obvious faults cross a safe limit, so fixed rules see them. Subtle faults keep every sensor inside its normal range but break the relationships between sensors: only a model that learned normal behaviour can notice. (Needs the labelled demo stream.)</div></div>

  <div class="mid2">
    <div class="panel"><div class="ph"><div class="pt">📈 Share of records flagged, per batch</div></div>
      <div class="pb"><svg id="trend" viewBox="0 0 600 150" preserveAspectRatio="none" style="width:100%;height:150px"></svg>
      <div class="legend"><span><i style="background:#64748B"></i>True faults</span><span><i style="background:#EF4444"></i>Rules (≥1 threshold)</span><span><i style="background:#14B8A6"></i>Autoencoder</span></div></div></div>
    <div class="panel"><div class="ph"><div class="pt">🏭 Machine health (AE flag rate)</div></div><div class="pb"><div class="mg" id="grid"></div></div></div>
  </div>

  <div class="mid">
    <div style="display:flex;flex-direction:column;gap:1rem">
      <div class="panel"><div class="ph"><div class="pt">🤖 Autoencoder model</div></div><div class="pb" id="model"></div></div>
      <div class="panel"><div class="ph"><div class="pt">🔎 Likely root-cause sensor (AE anomalies)</div></div><div class="pb" id="causes"></div></div>
      <div class="panel" id="yarnp" style="display:none"><div class="ph"><div class="pt">🐘 YARN applications</div></div><div class="pb" id="yarn"></div></div>
    </div>
    <div class="panel">
      <div class="ph"><div class="pt">📡 Latest sensor records</div></div>
      <div style="overflow-x:auto"><table class="tbl"><thead><tr>
        <th>Machine</th><th>Temp °C</th><th>Vib mm/s</th><th>Sound dB</th><th>Rules</th><th>AE score</th><th>AE verdict</th><th>Likely cause</th><th>Truth</th>
      </tr></thead><tbody id="tbl"></tbody></table></div>
    </div>
  </div>

  <div class="panel"><div class="ph"><div class="pt">🚨 Alerts (de-duplicated, with root cause)</div><div class="hint" style="padding:0" id="acount"></div></div>
    <div style="overflow-x:auto"><table class="tbl"><thead><tr><th>Time (UTC)</th><th>Level</th><th>Machine</th><th>Likely cause</th><th>Error vs threshold</th><th>Rule severity</th></tr></thead><tbody id="alerts"></tbody></table></div></div>

  <div class="panel"><div class="ph"><div class="pt">🗄️ Storage layout</div></div>
  <div class="pb"><div class="hg" id="paths"></div></div></div>
</main>
<footer>Autoencoder (NumPy) · real-time scoring · Machine Learning II &amp; Artificial Neural Networks</footer>
</div>
<script>
const $=id=>document.getElementById(id), fmt=n=>(n||0).toLocaleString();
setInterval(()=>{$('clk').textContent=new Date().toLocaleTimeString()},1000);
const CAUSE={temperature:'Temperature',vibration:'Vibration',sound_level:'Sound',power_consumption:'Power'};
const STEPS=[['generator','🏭','Sensor stream'],['processor','🧠','Autoencoder scoring'],['storage','🗄️','Storage'],['dashboard','🌐','Dashboard']];
function bar(label,v){return `<div class="m"><div class="l">${label}<b>${(v*100).toFixed(1)}%</b></div><div class="bar"><i style="width:${v*100}%"></i></div></div>`}
function bar2(label,v,c){return `<div class="m"><div class="l">${label}<b>${(v*100).toFixed(1)}%</b></div><div class="bar"><i style="width:${v*100}%;background:${c}"></i></div></div>`}
function ftypes(bt){
  const lab={threshold:'Obvious: a sensor over its safe limit',decoupling:'Subtle: sensors inconsistent with each other',mild_offset:'Subtle: mild offset inside the normal range',none:'Healthy readings (false-alarm rate)'};
  const rows=['threshold','decoupling','mild_offset','none'].filter(k=>bt&&bt[k]&&bt[k].n).map(k=>{const t=bt[k];
    return `<div style="margin-bottom:.9rem"><div style="font-size:.74rem;font-weight:600;margin-bottom:.2rem">${lab[k]} <span style="color:var(--muted);font-weight:400">(${fmt(t.n)} readings)</span></div>${bar2('Rules',t.rules/t.n,'#EF4444')}${bar2('Autoencoder',t.ae/t.n,'#14B8A6')}</div>`}).join('');
  $('ftypes').innerHTML=rows||'<div class="empty">Waiting for labelled data…</div>';
}
function trend(h){
  if(!h||h.length<2){$('trend').innerHTML='';return}
  const W=600,H=150,P=6,mx=Math.max(.1,...h.flatMap(p=>[p.ae,p.rules,p.truth]));
  const pts=k=>h.map((p,i)=>`${(P+i*(W-2*P)/(h.length-1)).toFixed(1)},${(H-P-p[k]/mx*(H-2*P)).toFixed(1)}`).join(' ');
  $('trend').innerHTML=[['truth','#64748B','5 4'],['rules','#EF4444',''],['ae','#14B8A6','']].map(([k,c,da])=>`<polyline fill="none" vector-effect="non-scaling-stroke" stroke="${c}" stroke-width="2" ${da?`stroke-dasharray="${da}"`:''} points="${pts(k)}"/>`).join('');
}
function grid(bm){
  const ks=Object.keys(bm||{}).sort();
  $('grid').innerHTML=ks.length?ks.map(k=>{const m=bm[k],r=m.n?m.ae/m.n:0,hue=140*(1-Math.min(r/.5,1));
    return `<div class="cell" style="background:hsl(${hue},70%,55%)" title="${k}: ${(r*100).toFixed(1)}% flagged of ${m.n}">${m.type.slice(0,4)} ${k.slice(-3)}</div>`}).join(''):'<div class="empty">Waiting…</div>';
}
function alerts(d){
  const c=d.alert_counts||{};
  $('acount').textContent=`${c.CRITICAL||0} critical · ${c.WARNING||0} warning · ${c.suppressed||0} duplicates suppressed`;
  const a=(d.alerts||[]).slice().reverse();
  $('alerts').innerHTML=a.length?a.map(x=>`<tr><td>${(x.alert_time||'').slice(11,19)}</td><td><span class="lv ${x.level==='CRITICAL'?'lv-c':'lv-w'}">${x.level}</span></td><td>${x.machine_id}</td><td>${CAUSE[x.root_cause]||x.root_cause}</td><td>${x.ae_ratio.toFixed(1)}×</td><td>${x.rule_severity}</td></tr>`).join(''):'<tr><td colspan="6" class="empty">No alerts yet…</td></tr>';
}
function render(d){
  $('mode').textContent=(d.mode==='hdfs'?'HDFS CLUSTER':'LOCAL DEMO')+' · live';
  $('steps').innerHTML=STEPS.map(([k,ic,nm])=>{const s=d.pipeline[k]||{ok:false,msg:'...'};
    return `<div class="step ${s.ok?'running':'waiting'}"><div class="si">${ic}</div><div class="sn">${nm}</div><div class="sm">${s.msg}</div><span class="badge ${s.ok?'br':'bw'}">${s.ok?'✅ RUNNING':'⏳ WAITING'}</span></div>`}).join('');
  const s=d.stats||{};
  $('s1').textContent=fmt(s.total);$('s2').textContent=fmt(s.faults);$('s3').textContent=fmt(s.critical);$('s4').textContent=fmt(s.ae);
  $('s1s').textContent=fmt(s.batches)+' batches written';
  const names={rules:['Rules (original)','fixed thresholds on single sensors'],ae:['Autoencoder','learned normal behaviour']};
  const sc=d.scores||{}, keys=Object.keys(names), best=keys.reduce((a,k)=>(sc[k]&&(!sc[a]||sc[k].f1>sc[a].f1))?k:a,keys[0]);
  $('cmp').innerHTML=sc.ae&&sc.rules?keys.map(k=>{const c=sc[k];return `<div class="cm ${k===best?'best':''}"><h4>${names[k][0]}${k===best?' 🏆':''}</h4><div class="sub">${names[k][1]}</div>${bar('Precision',c.precision)}${bar('Recall',c.recall)}${bar('F1 score',c.f1)}<div class="sub" style="margin-top:.6rem">TP ${fmt(c.tp)} · FP ${fmt(c.fp)} · FN ${fmt(c.fn)}</div></div>`}).join(''):'<div class="empty">Waiting for data…</div>';
  const m=d.model;
  $('model').innerHTML=m?`<div class="kv"><span>Architecture</span><b>${m.sizes.join(' → ')}</b><span>Alert threshold</span><b>${m.threshold.toFixed(4)}</b><span>Test ROC-AUC</span><b>${m.auc??'—'}</b><span>Test precision / recall</span><b>${m.precision??'—'} / ${m.recall??'—'}</b><span>Trained on</span><b>${fmt(m.train_rows)} normal rows</b><span>Trained at</span><b>${(m.trained_at||'—').replace('T',' ')}</b></div>`:'<div class="empty">No model yet — run train_autoencoder.py</div>';
  const rc=d.root_causes||{}, tot=Object.values(rc).reduce((a,v)=>a+v,0);
  $('causes').innerHTML=tot?Object.keys(CAUSE).map(k=>{const v=(rc[k]||0)/tot;return `<div class="tyrow"><div class="n">${CAUSE[k]}</div><div class="bar"><i style="width:${v*100}%"></i></div><div class="v">${(v*100).toFixed(0)}%</div></div>`}).join(''):'<div class="empty">Waiting…</div>';
  trend(d.history);grid(d.by_machine);alerts(d);ftypes(d.by_fault_type);
  $('yarnp').style.display=d.mode==='hdfs'?'block':'none';
  $('yarn').innerHTML=(d.yarn_apps||[]).length?d.yarn_apps.map(a=>`<div class="ar"><div><div class="an">⚡ ${a.name}</div><div class="ai">${a.id}</div></div><span class="bg">${a.state}</span></div>`).join(''):'<div class="empty">No Flink app on YARN</div>';
  const rs=(d.records||[]).slice().reverse();
  $('tbl').innerHTML=rs.length?rs.map(r=>`<tr class="${r.ae_anomaly?'flag':''}"><td>${r.machine_id}</td>
    <td style="${r.temperature>75?'color:var(--red)':''}">${r.temperature.toFixed(1)}</td>
    <td style="${r.vibration>20?'color:var(--orange)':''}">${r.vibration.toFixed(1)}</td>
    <td style="${r.sound_level>80?'color:var(--blue)':''}">${r.sound_level.toFixed(1)}</td>
    <td class="s${Math.min(r.severity,3)}">${r.severity}</td><td>${r.ae_score.toFixed(3)}</td>
    <td>${r.ae_anomaly?'<span class="tag tg-ae">ANOMALY</span>':'<span class="tg-ok">normal</span>'}</td>
    <td>${r.ae_anomaly?CAUSE[r.root_cause]||r.root_cause:'—'}</td>
    <td>${r.is_fault===null?'—':r.is_fault?`<span class="tag tg-f">${r.fault_type}</span>`:'<span class="tg-ok">ok</span>'}</td></tr>`).join(''):'<tr><td colspan="9" class="empty">Waiting for data…</td></tr>';
  const p=d.paths||{}, P=[['output/streaming/all','All enriched records'],['output/streaming/critical','Rule critical alerts'],['output/streaming/ae_anomalies','Autoencoder anomalies'],['output/streaming/alerts','De-duplicated alerts'],['checkpoints','Flink checkpoints']];
  $('paths').innerHTML=P.map(([k,ds])=>`<div class="hr"><div><div class="hn">/flink/${k}/</div><div class="hd">${ds}</div></div><div class="hc">${fmt(p[k])} files</div></div>`).join('');
}
async function go(){try{render(await fetch('/api').then(r=>r.json()))}catch(e){}}
go();setInterval(go,3000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            body, ctype = HTML.encode(), "text/html;charset=utf-8"
        elif self.path == "/api":
            body, ctype = json.dumps(state).encode(), "application/json"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    threading.Thread(target=refresh, daemon=True).start()
    print(f"Dashboard: http://localhost:{config.DASHBOARD_PORT}   (storage: {config.STORAGE})")
    ThreadingHTTPServer(("0.0.0.0", config.DASHBOARD_PORT), Handler).serve_forever()
