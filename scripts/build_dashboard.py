"""Generate a self-contained dashboard_<date>.html from predictions_<date>.json.

Organic, scannable redesign: top-5 players per game shown as probability lanes
(not tables), with +EV players surfaced against the market. Data is embedded as
a JS const (no server, no external fetch). Rerun any time; reads cached JSON.

Usage: python -m scripts.build_dashboard 2026-10-08
"""
import json
import sys

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NHL Goal-Scorer Edge — __DATE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,900&family=Hanken+Grotesk:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
/* design · archetype: dashboard · style: organic-editorial + precise-data · axes: form, encoding
 * brief: audience=bettor scanning tonight's slate · decision=who scores + where's value · tone=organic/digestible
 * palette: warm charcoal oklch(17% .01 70) · ice oklch(80% .12 230) · value oklch(80% .16 155)
 * type: Fraunces (display) + Hanken Grotesk (ui/data) */
:root{
  --bg:oklch(15% .008 256); --bg-2:oklch(19% .009 256); --surface:oklch(21.5% .010 256);
  --surface-2:oklch(25% .013 256);
  --line:oklch(32% .013 256); --line-soft:oklch(26% .010 256);
  --fg:oklch(97% .004 256); --fg-dim:oklch(76% .008 256); --fg-faint:oklch(61% .011 256);
  --ice:oklch(80% .12 232); --ice-2:oklch(87% .13 205);
  --pos:oklch(82% .17 152); --pos-dim:oklch(60% .12 152); --pos-bg:oklch(42% .10 152);
  --neg:oklch(70% .15 28); --warn:oklch(84% .13 85);
  --font-display:"Fraunces",Georgia,serif;
  --font-ui:"Hanken Grotesk",system-ui,sans-serif;
  --r:18px; --r-sm:11px;
  --ease:cubic-bezier(.2,.7,.2,1);
}
*{box-sizing:border-box}
html,body{overflow-x:clip}
body{
  margin:0;background:var(--bg);color:var(--fg);font-family:var(--font-ui);
  font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased;
  padding:clamp(18px,4vw,40px) clamp(14px,4vw,40px) 60px;
  background-image:
    radial-gradient(65% 55% at 90% -12%, oklch(34% .08 232 / .42), transparent 70%),
    radial-gradient(48% 44% at -2% 2%, oklch(32% .07 268 / .28), transparent 66%);
  background-attachment:fixed;
}
.num{font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.wrap{max-width:1180px;margin:0 auto}

/* ---------- header ---------- */
header{display:flex;flex-wrap:wrap;gap:16px 28px;align-items:flex-end;
  justify-content:space-between;margin-bottom:26px}
.brand h1{font-family:var(--font-display);font-optical-sizing:auto;font-weight:900;
  font-size:clamp(26px,5vw,44px);line-height:.95;letter-spacing:-.02em;margin:0;
  color:var(--fg)}
.brand h1 em{font-style:italic;font-weight:500;color:var(--ice)}
.brand p{margin:7px 0 0;color:var(--fg-faint);font-size:13.5px;max-width:46ch}
.meta{display:flex;flex-wrap:wrap;gap:7px 14px;align-items:center;
  font-size:12.5px;color:var(--fg-faint);text-align:right}
.meta .k{color:var(--fg-dim);font-weight:600}
.badge{font-weight:700;font-size:12px;padding:5px 11px;border-radius:999px;
  border:1px solid var(--line);color:var(--fg-dim)}
.badge.good{color:var(--pos);border-color:color-mix(in oklch,var(--pos),transparent 60%);
  background:color-mix(in oklch,var(--pos),transparent 90%)}
.badge.warnb{color:var(--warn);border-color:color-mix(in oklch,var(--warn),transparent 60%)}
.badge.lowb{color:var(--neg);border-color:color-mix(in oklch,var(--neg),transparent 55%)}

/* ---------- controls ---------- */
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:0 0 22px}
.seg{display:inline-flex;background:var(--bg-2);border:1px solid var(--line-soft);
  border-radius:999px;padding:3px}
.seg button{font-family:var(--font-ui);font-size:13px;font-weight:600;color:var(--fg-faint);
  background:none;border:none;padding:7px 15px;border-radius:999px;cursor:pointer;
  min-height:34px;transition:.18s var(--ease)}
.seg button:hover{color:var(--fg-dim)}
.seg button[aria-pressed=true]{background:var(--surface-2);color:var(--fg);
  box-shadow:0 1px 0 oklch(100% 0 0 / .04),0 2px 8px oklch(0% 0 0 / .25)}
.seg button:focus-visible{outline:2px solid var(--ice);outline-offset:2px}
.toggle{margin-left:auto;display:inline-flex;align-items:center;gap:9px;
  color:var(--fg-faint);font-size:13px;font-weight:600}
.toggle input{appearance:none;width:40px;height:23px;border-radius:999px;background:var(--line);
  position:relative;cursor:pointer;transition:.18s var(--ease);flex:none}
.toggle input::after{content:"";position:absolute;top:2px;left:2px;width:19px;height:19px;
  border-radius:50%;background:var(--fg);transition:.18s var(--ease)}
.toggle input:checked{background:var(--pos-bg)}
.toggle input:checked::after{transform:translateX(17px);background:var(--pos)}
.toggle input:focus-visible{outline:2px solid var(--ice);outline-offset:2px}

/* ---------- game cards ---------- */
.games{display:grid;gap:clamp(14px,2vw,22px);
  grid-template-columns:repeat(auto-fill,minmax(min(100%,500px),1fr))}
.games.flat{grid-template-columns:1fr}
.game{display:flex;flex-direction:column;gap:.25rem;background:
    linear-gradient(180deg, oklch(100% 0 0 / .018), transparent 40%),
    var(--surface);
  border:1px solid var(--line-soft);border-radius:var(--r);
  padding:clamp(14px,2.2vw,22px);box-shadow:0 1px 2px oklch(0% 0 0 / .3)}
.match{display:flex;align-items:baseline;justify-content:space-between;gap:12px;padding-bottom:.25rem;border-bottom:1px solid var(--line-soft)}
.match h2{font-family:var(--font-display);font-weight:600;font-size:21px;margin:0;
  letter-spacing:-.01em;color:var(--fg)}
.match h2 .at{color:var(--fg-faint);font-style:italic;font-weight:400;padding:0 5px}
.match .tag{font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
  color:var(--fg-faint)}

/* ---------- player lane (tiered: identity+value / bar / detail) ---------- */
.lane{display:flex;flex-direction:column;gap:10px;
  padding:16px 14px;border-radius:var(--r-sm);
  transition:background .18s var(--ease);
  opacity:0;transform:translateY(7px);animation:rise .5s var(--ease) forwards}
@keyframes rise{to{opacity:1;transform:none}}
.lane + .lane{border-top:1px solid var(--line-soft)}
.lane:hover{background:oklch(100% 0 0 / .022)}
.lane.val{background:
  linear-gradient(90deg, color-mix(in oklch,var(--pos-bg),transparent 82%), transparent 60%);
  box-shadow:inset 3px 0 0 var(--pos)}
.lane.val:hover{background:
  linear-gradient(90deg, color-mix(in oklch,var(--pos-bg),transparent 74%), transparent 60%)}

/* tier 1: who + value chip */
.top{display:flex;align-items:baseline;justify-content:space-between;gap:12px}
.who{display:flex;align-items:baseline;gap:9px;min-width:0;flex:1 1 auto}
.rk{font-family:var(--font-display);font-weight:500;font-size:15px;color:var(--fg-faint);
  width:17px;flex:none;font-feature-settings:"tnum"}
.nm{font-family:var(--font-display);font-weight:600;font-size:17px;color:var(--fg);
  letter-spacing:-.01em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tm{font-size:11px;font-weight:700;letter-spacing:.06em;color:var(--fg-faint);flex:none}
.gm{font-size:11px;font-weight:700;letter-spacing:.03em;color:var(--ice);opacity:.8;flex:none}
.chip{font-weight:800;font-size:13.5px;padding:4px 11px;border-radius:999px;white-space:nowrap;
  flex:none;font-variant-numeric:tabular-nums}
.chip.pos{color:var(--pos);background:color-mix(in oklch,var(--pos),transparent 86%);
  border:1px solid color-mix(in oklch,var(--pos),transparent 65%)}
.chip.neg{color:var(--fg-faint);background:var(--bg-2);border:1px solid var(--line-soft)}
.chip.none{color:var(--fg-faint);font-weight:600;font-size:12px;background:none;
  border:1px dashed var(--line);padding:4px 10px}

/* tier 2: probability bar */
.prob{display:flex;align-items:center;gap:13px}
.track{flex:1;height:12px;border-radius:999px;background:var(--bg-2);
  box-shadow:inset 0 1px 2px oklch(0% 0 0 / .35);overflow:hidden;min-width:60px}
.fill{height:100%;border-radius:999px;transform-origin:left;
  background:linear-gradient(90deg,var(--ice),var(--ice-2));
  box-shadow:0 0 12px color-mix(in oklch,var(--ice),transparent 55%);
  animation:grow .7s var(--ease) both}
@keyframes grow{from{transform:scaleX(0)}}
.pct{font-weight:800;font-size:19px;color:var(--fg);min-width:62px;text-align:right;
  font-variant-numeric:tabular-nums;line-height:1}
.pct small{font-size:10.5px;font-weight:600;color:var(--fg-faint);display:block;line-height:1;
  margin-top:3px;letter-spacing:.02em}

/* tier 3: detail line */
.detail{font-size:12.5px;color:var(--fg-faint);font-variant-numeric:tabular-nums}
.detail b{color:var(--fg-dim);font-weight:700}
.detail .lbl{font-size:10.5px;font-weight:800;text-transform:uppercase;letter-spacing:.06em;
  color:var(--fg-faint);margin-right:3px}
.detail .sep{opacity:.45;padding:0 7px}

/* ---------- legend ---------- */
.legend{max-width:1180px;margin:30px auto 0;color:var(--fg-faint);font-size:12.5px;
  line-height:1.7;border-top:1px solid var(--line-soft);padding-top:16px}
.legend b{color:var(--fg-dim)}
.legend .sw{display:inline-block;width:11px;height:11px;border-radius:3px;
  vertical-align:middle;margin-right:4px;background:linear-gradient(90deg,var(--ice),var(--ice-2))}
.legend .sw.v{background:var(--pos)}

@media (max-width:420px){
  .nm{font-size:15.5px}
  .detail{font-size:12px}
}
@media (prefers-reduced-motion:reduce){
  .lane,.fill{animation:none;opacity:1;transform:none}
}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="brand">
    <h1>Goal-Scorer <em>Edge</em></h1>
    <p id="sub"></p>
  </div>
  <div class="meta" id="meta"></div>
</header>

<div class="controls">
  <div class="seg" role="group" aria-label="view">
    <button data-view="game" aria-pressed="true">By game</button>
    <button data-view="all" aria-pressed="false">All players</button>
  </div>
  <div class="seg" role="group" aria-label="sort order">
    <button data-sort="p_goal" aria-pressed="true">Most likely</button>
    <button data-sort="ev" aria-pressed="false">Best value</button>
  </div>
  <label class="toggle"><input type="checkbox" id="valonly">value bets only</label>
</div>

<main class="games" id="games"></main>

<div class="legend">
  <span class="sw"></span><b>Goal bar</b> our model's chance the player scores (bar scaled to 50%). ·
  <span class="sw v"></span><b>Value</b> positive expected value vs the Vegas price —
  <b>EV</b> = how much each $1 bet returns on average. ·
  <b>pt</b> chance of a point (goal or assist). ·
  <b>fair</b> the odds our probability implies; <b>Veg</b> the actual line.
  <br>Top 5 per game. Roster-based before lineups post; opponent goalie not start-adjusted. Not betting advice.
</div>
</div>

<script>
const DATA = __DATA__;
let sortKey="p_goal", valOnly=false, view="game";
const MAXP=0.5;

const pct=x=>x==null?"—":(x*100).toFixed(0)+"%";
const pct1=x=>x==null?"—":((x>=0?"+":"")+(x*100).toFixed(1)+"%");
const am=o=>o==null?"—":((o>0?"+":"")+Math.round(o));
const fair=p=>p==null?"—":am(p>=0.5?-100*p/(1-p):100*(1-p)/p);
const isVal=p=>p.ev!=null&&p.ev>0;
const sortFn=(a,b)=>{const av=a[sortKey],bv=b[sortKey];
  if(av==null&&bv==null)return b.p_goal-a.p_goal;
  if(av==null)return 1; if(bv==null)return -1; return bv-av;};

function metaBar(){
  const m=DATA, r=m.requests_remaining;
  let badge='<span class="badge">odds not pulled</span>';
  if(r!=null){const n=+r,c=n>100?"good":n>30?"warnb":"lowb";
    badge=`<span class="badge ${c}">${n} / 500 calls left</span>`;}
  document.getElementById("sub").textContent=
    "Most likely goal scorers for tonight's slate, with value vs the market.";
  const parts=[`<span><span class="k">${m.date}</span></span>`];
  if(m.odds_pulled_at) parts.push(`<span>odds <span class="k">${m.odds_pulled_at.slice(11,16)}</span></span>`);
  else if(m.generated) parts.push(`<span>built <span class="k">${m.generated.slice(11,16)}</span></span>`);
  parts.push(badge);
  document.getElementById("meta").innerHTML=parts.join("");
}

function laneHTML(p,i,showGame){
  const val=isVal(p);
  const w=Math.max(4,Math.min(100,(p.p_goal/MAXP)*100));
  const d=Math.min(i,14)*45;
  const gm=showGame?`<span class="gm">${p.game}</span>`:"";
  const ptPart=`<span class="lbl">point</span><b>${pct(p.p_point)}</b> (fair ${fair(p.p_point)})`;
  let chip, detail;
  if(p.vegas_odds==null){
    chip=`<span class="chip none">no line yet</span>`;
    detail=ptPart;
  }else{
    chip=`<span class="chip ${val?'pos':'neg'}">${val?'+EV ':''}${pct1(p.ev)}</span>`;
    detail=`<span class="lbl">goal</span><b>${am(p.vegas_odds)}</b> (fair ${fair(p.p_goal)})`
          +`<span class="sep">·</span>${ptPart}`;
  }
  return `<div class="lane ${val?'val':''}" style="animation-delay:${d}ms">
    <div class="top">
      <div class="who"><span class="rk num">${i+1}</span>
        <span class="nm">${p.name}</span><span class="tm">${p.team}</span>${gm}</div>
      ${chip}
    </div>
    <div class="prob">
      <div class="track"><div class="fill" style="width:${w}%;animation-delay:${d+60}ms"></div></div>
      <span class="pct num">${(p.p_goal*100).toFixed(0)}%<small>P(goal)</small></span>
    </div>
    <div class="detail">${detail}</div>
  </div>`;
}

function render(){
  const host=document.getElementById("games");
  host.className="games"+(view==="all"?" flat":"");
  host.innerHTML="";
  if(view==="all"){
    let all=[];
    for(const g of DATA.games) for(const p of g.players) all.push(Object.assign({game:g.game},p));
    all.sort(sortFn);
    if(valOnly) all=all.filter(isVal);
    if(!all.length){host.innerHTML='<p style="color:var(--fg-faint)">No value bets on the board right now.</p>';return;}
    const label=sortKey==="ev"?"by value":"by goal probability";
    const card=document.createElement("section");card.className="game";
    card.innerHTML=`<div class="match"><h2>All players</h2>`
      +`<span class="tag">${all.length} · ${label}</span></div>`
      +all.map((p,i)=>laneHTML(p,i,true)).join("");
    host.appendChild(card);
    return;
  }
  for(const g of DATA.games){
    let ps=g.players.slice().sort(sortFn);
    if(valOnly) ps=ps.filter(isVal);
    if(valOnly&&!ps.length) continue;
    const card=document.createElement("section");card.className="game";
    card.innerHTML=`<div class="match">
        <h2>${g.away}<span class="at">at</span>${g.home}</h2>
        <span class="tag">top 5</span>
      </div>${ps.map((p,i)=>laneHTML(p,i)).join("")}`;
    host.appendChild(card);
  }
  if(!host.children.length)
    host.innerHTML='<p style="color:var(--fg-faint)">No value bets on the board right now.</p>';
}

document.querySelectorAll("[data-view]").forEach(b=>b.onclick=()=>{
  view=b.dataset.view;
  document.querySelectorAll("[data-view]").forEach(x=>x.setAttribute("aria-pressed",x===b));
  render();
});
document.querySelectorAll("[data-sort]").forEach(b=>b.onclick=()=>{
  sortKey=b.dataset.sort;
  document.querySelectorAll("[data-sort]").forEach(x=>x.setAttribute("aria-pressed",x===b));
  render();
});
document.getElementById("valonly").onchange=e=>{valOnly=e.target.checked;render();};
metaBar();render();
</script>
</body>
</html>
"""


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-10-08"
    doc = json.load(open(f"predictions_{date}.json"))
    html = HTML.replace("__DATE__", date).replace("__DATA__", json.dumps(doc))
    out = f"dashboard_{date}.html"
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
