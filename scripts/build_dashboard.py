"""Generate a self-contained dashboard_<date>.html from predictions_<date>.json.

Design: a broadcast-quant "edge instrument". Each player's chance to score is an
ICE bar; the GOLD overhang shows the edge past Vegas's implied line (the tick),
so value is visible as the bar extending beyond the market. Cold = likelihood,
warm = money. Two views (by-game / leaderboard), sort + value filter. Data is
embedded as a JS const (no server, no external data fetch).

Usage: python -m scripts.build_dashboard 2026-10-08
"""
import json
import sys

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Goal-Scorer Edge — __DATE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
/* design · archetype: dashboard/leaderboard · style: broadcast-quant + instrument · axes: temperature, register
 * brief: audience=sharp bettor · decision=who scores + where's the edge vs Vegas · tone=precise, confident
 * palette: ink oklch(14% .02 255) · ice oklch(82% .14 220) · gold oklch(84% .145 85)
 * type: Archivo (display) + JetBrains Mono (readout) */
:root{
  --ink:oklch(13.5% .018 258); --ink-2:oklch(16.5% .02 258); --ink-3:oklch(20% .022 258);
  --line:oklch(29% .02 258); --line-soft:oklch(23% .016 258);
  --paper:oklch(96% .008 245); --muted:oklch(73% .015 248); --faint:oklch(55% .018 252);
  --ice:oklch(83% .135 218); --ice-deep:oklch(66% .15 228);
  --gold:oklch(85% .145 86); --gold-deep:oklch(74% .15 72);
  --neg:oklch(66% .14 25);
  --d1:"Archivo",system-ui,sans-serif; --mono:"JetBrains Mono",monospace;
  --ease:cubic-bezier(.2,.75,.2,1); --r:16px;
}
*{box-sizing:border-box}
html,body{overflow-x:clip}
body{
  margin:0;background:var(--ink);color:var(--paper);font-family:var(--d1);
  font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased;
  padding:clamp(18px,3.5vw,38px) clamp(14px,3.5vw,38px) 72px;
  background-image:
    radial-gradient(70% 60% at 100% -10%, oklch(30% .09 220 / .40), transparent 60%),
    radial-gradient(55% 50% at -5% 0%, oklch(28% .07 86 / .16), transparent 60%),
    linear-gradient(oklch(100% 0 0 / .015), transparent 300px);
  background-attachment:fixed;
}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
.wrap{max-width:1120px;margin:0 auto}

/* ---------- masthead ---------- */
header{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;
  gap:18px 30px;padding-bottom:18px;border-bottom:2px solid var(--line);margin-bottom:5px}
.mk{display:flex;align-items:center;gap:13px}
.puck{width:16px;height:16px;border-radius:50%;background:var(--gold);flex:none;
  box-shadow:0 0 18px color-mix(in oklch,var(--gold),transparent 40%);margin-bottom:9px}
.mk h1{font-family:var(--d1);font-weight:900;font-size:clamp(27px,5.2vw,46px);
  line-height:.86;letter-spacing:-.035em;margin:0;text-transform:uppercase}
.mk h1 span{color:var(--ice)}
.mk p{margin:6px 0 0;color:var(--faint);font-size:12px;font-family:var(--mono);
  letter-spacing:.02em;text-transform:uppercase}
.meta{display:flex;flex-wrap:wrap;gap:4px 16px;align-items:center;justify-content:flex-end;
  font-family:var(--mono);font-size:11.5px;color:var(--faint);text-transform:uppercase;
  letter-spacing:.03em;text-align:right}
.meta .k{color:var(--muted)}
.badge{font-family:var(--mono);font-weight:700;font-size:11px;padding:4px 10px;border-radius:5px;
  border:1px solid var(--line);color:var(--muted);letter-spacing:.03em}
.badge.good{color:var(--gold);border-color:color-mix(in oklch,var(--gold),transparent 55%)}
.badge.warnb{color:var(--gold-deep);border-color:color-mix(in oklch,var(--gold-deep),transparent 55%)}
.badge.lowb{color:var(--neg);border-color:color-mix(in oklch,var(--neg),transparent 50%)}

/* ---------- slate strip + controls ---------- */
.strip{display:flex;flex-wrap:wrap;align-items:center;gap:8px 20px;margin:18px 0 20px;
  font-family:var(--mono);font-size:12px;color:var(--faint);text-transform:uppercase;
  letter-spacing:.06em}
.strip b{color:var(--paper);font-weight:700}
.strip .v{color:var(--gold)}
.strip .spacer{flex:1 1 40px}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.seg{display:inline-flex;background:var(--ink-2);border:1px solid var(--line-soft);border-radius:8px;padding:3px}
.seg button{font-family:var(--d1);font-size:12.5px;font-weight:700;color:var(--muted);
  background:none;border:none;padding:7px 13px;border-radius:6px;cursor:pointer;min-height:34px;
  text-transform:uppercase;letter-spacing:.03em;transition:.16s var(--ease)}
.seg button:hover{color:var(--paper)}
.seg button[aria-pressed=true]{background:var(--paper);color:var(--ink)}
.seg button:focus-visible{outline:2px solid var(--ice);outline-offset:2px}
.vfilter{display:inline-flex;align-items:center;gap:9px;font-family:var(--d1);font-size:12.5px;
  font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.03em;cursor:pointer}
.vfilter input{appearance:none;width:40px;height:24px;border-radius:999px;background:var(--line);
  position:relative;cursor:pointer;transition:.16s var(--ease);flex:none}
.vfilter input::after{content:"";position:absolute;top:2px;left:2px;width:20px;height:20px;
  border-radius:50%;background:var(--paper);transition:.16s var(--ease)}
.vfilter input:checked{background:color-mix(in oklch,var(--gold),var(--ink) 55%)}
.vfilter input:checked::after{transform:translateX(16px);background:var(--gold)}
.vfilter input:focus-visible{outline:2px solid var(--ice);outline-offset:2px}

/* ---------- by-game sections ---------- */
.board{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(min(100%,480px),1fr))}
.board.flat{grid-template-columns:1fr;gap:0}
.game{background:var(--ink-2);border:1px solid var(--line-soft);border-radius:var(--r);
  padding:clamp(13px,2vw,20px);box-shadow:0 2px 14px oklch(0% 0 0 / .25)}
.match{display:flex;align-items:center;justify-content:space-between;gap:12px;
  padding-bottom:12px;margin-bottom:4px;border-bottom:1px solid var(--line-soft)}
.vs{font-family:var(--d1);font-weight:900;font-size:19px;letter-spacing:-.01em;text-transform:uppercase}
.vs .x{color:var(--faint);font-weight:600;padding:0 7px}
.vs .home{color:var(--muted)}
.mtag{font-family:var(--mono);font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;
  color:var(--faint)}
.mtag .v{color:var(--gold)}
/* leaderboard header (all view) */
.lead-h{display:flex;align-items:baseline;justify-content:space-between;gap:12px;
  padding:4px 2px 16px}
.lead-h h2{font-family:var(--d1);font-weight:900;font-size:24px;letter-spacing:-.02em;margin:0;
  text-transform:uppercase}
.lead-h .by{font-family:var(--mono);font-size:11px;color:var(--faint);text-transform:uppercase;
  letter-spacing:.06em}
.flat .game{background:none;border:none;box-shadow:none;padding:0}
.flat .match{display:none}

/* ---------- player row ---------- */
.p{display:grid;column-gap:16px;row-gap:7px;align-items:center;padding:13px 6px;
  grid-template-columns:30px minmax(0,1fr) minmax(130px,2.1fr) 64px 82px;
  grid-template-areas:"rk who meter pct ev" "rk sub sub sub ev";
  position:relative}
.p + .p{border-top:1px solid var(--line-soft)}
.p.val::before{content:"";position:absolute;left:-6px;top:6px;bottom:6px;width:3px;
  border-radius:3px;background:var(--gold);box-shadow:0 0 10px color-mix(in oklch,var(--gold),transparent 45%)}
.rk{grid-area:rk;font-family:var(--d1);font-weight:800;font-size:17px;color:var(--faint);
  font-variant-numeric:tabular-nums;text-align:center}
.who{grid-area:who;min-width:0}
.nm{font-family:var(--d1);font-weight:700;font-size:16.5px;letter-spacing:-.01em;color:var(--paper);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tag{font-family:var(--mono);font-size:10.5px;letter-spacing:.04em;color:var(--faint);
  text-transform:uppercase;margin-top:1px;display:flex;gap:7px}
.tag .gm{color:var(--ice)}

/* the ice/gold edge meter */
.meter{grid-area:meter;position:relative;height:15px;border-radius:999px;background:var(--ink);
  box-shadow:inset 0 1px 3px oklch(0% 0 0 / .5), inset 0 0 0 1px oklch(100% 0 0 / .03);overflow:hidden}
.meter .ice{position:absolute;left:0;top:0;bottom:0;border-radius:999px;transform-origin:left;
  background:linear-gradient(90deg,var(--ice-deep),var(--ice))}
.meter .gold{position:absolute;top:0;bottom:0;transform-origin:left;
  background:linear-gradient(90deg,var(--gold-deep),var(--gold));
  box-shadow:0 0 14px color-mix(in oklch,var(--gold),transparent 35%)}
.meter .mk-tick{position:absolute;top:0;bottom:0;width:2px;margin-left:-1px;
  background:var(--paper);opacity:.75;box-shadow:0 0 5px oklch(0% 0 0 / .6)}
.pct{grid-area:pct;font-family:var(--d1);font-weight:800;font-size:20px;color:var(--paper);
  font-variant-numeric:tabular-nums;text-align:right;letter-spacing:-.02em;line-height:1}
.pct small{display:block;font-family:var(--mono);font-size:9px;font-weight:500;color:var(--faint);
  text-transform:uppercase;letter-spacing:.08em;margin-top:3px}

/* value cell */
.ev{grid-area:ev;justify-self:end;text-align:right}
.ev .pill{font-family:var(--mono);font-weight:700;font-size:14px;padding:5px 9px;border-radius:7px;
  display:inline-block;letter-spacing:-.01em}
.ev.pos .pill{color:var(--ink);background:var(--gold);
  box-shadow:0 0 16px color-mix(in oklch,var(--gold),transparent 55%)}
.ev.neg .pill{color:var(--faint);background:var(--ink-3)}
.ev.none .pill{color:var(--faint);background:none;border:1px dashed var(--line);font-size:11px}
.ev small{display:block;font-family:var(--mono);font-size:9px;color:var(--faint);
  text-transform:uppercase;letter-spacing:.07em;margin-top:4px}

/* readout sub-line */
.sub{grid-area:sub;font-family:var(--mono);font-size:11.5px;color:var(--faint);
  letter-spacing:.01em;display:flex;flex-wrap:wrap;gap:3px 12px}
.sub b{color:var(--muted);font-weight:500}
.sub .s{color:var(--faint);opacity:.6}

/* ---------- legend ---------- */
.legend{max-width:1120px;margin:34px auto 0;padding-top:18px;border-top:1px solid var(--line-soft);
  font-family:var(--mono);font-size:11.5px;color:var(--faint);line-height:1.9;letter-spacing:.01em}
.legend .key{display:inline-flex;align-items:center;gap:7px;margin-right:22px;white-space:nowrap}
.legend .chip-ice,.legend .chip-gold,.legend .chip-tick{display:inline-block;height:11px;border-radius:3px}
.legend .chip-ice{width:22px;background:linear-gradient(90deg,var(--ice-deep),var(--ice))}
.legend .chip-gold{width:14px;background:var(--gold)}
.legend .chip-tick{width:2px;height:13px;background:var(--paper)}
.legend b{color:var(--muted)}

/* ---------- motion (first paint only) ---------- */
.empty{color:var(--faint);font-family:var(--mono);padding:30px 6px;text-align:center}
@keyframes rise{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:none}}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes fade{from{opacity:0}to{opacity:.75}}
.board.animate .p{animation:rise .5s var(--ease) backwards;animation-delay:var(--d)}
.board.animate .meter .ice,.board.animate .meter .gold{animation:grow .75s var(--ease) backwards;
  animation-delay:calc(var(--d) + 80ms)}
.board.animate .meter .mk-tick{animation:fade .4s var(--ease) backwards;
  animation-delay:calc(var(--d) + .55s)}
@media (prefers-reduced-motion:reduce){
  .board.animate .p,.board.animate .ice,.board.animate .gold,.board.animate .mk-tick{animation:none}
}

@media (max-width:560px){
  .p{grid-template-columns:28px 1fr auto;
     grid-template-areas:"rk who ev" "rk meter pct" "sub sub sub";column-gap:12px}
  .pct{font-size:18px}
  .sub{margin-top:2px}
}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="mk">
    <span class="puck"></span>
    <div>
      <h1>Goal-Scorer <span>Edge</span></h1>
      <p id="sub">model vs market · anytime scorer</p>
    </div>
  </div>
  <div class="meta" id="meta"></div>
</header>

<div class="strip">
  <span id="summary"></span>
  <span class="spacer"></span>
  <div class="controls">
    <div class="seg" role="group" aria-label="view">
      <button data-view="game" aria-pressed="true">By game</button>
      <button data-view="all" aria-pressed="false">Leaderboard</button>
    </div>
    <div class="seg" role="group" aria-label="sort">
      <button data-sort="p_goal" aria-pressed="true">Likely</button>
      <button data-sort="ev" aria-pressed="false">Value</button>
    </div>
    <label class="vfilter"><input type="checkbox" id="valonly">Value only</label>
  </div>
</div>

<main class="board" id="board"></main>

<div class="legend">
  <span class="key"><span class="chip-ice"></span><b>Ice</b> = our chance to score</span>
  <span class="key"><span class="chip-gold"></span><b>Gold</b> = edge past Vegas (value)</span>
  <span class="key"><span class="chip-tick"></span><b>Tick</b> = Vegas implied chance</span>
  <br>Fill reaching past the tick is a +EV bet. EV = return per $1 at the Vegas price.
  PT = point chance. Top 5 per game · roster-based before lineups post · not betting advice.
</div>
</div>

<script>
const DATA = __DATA__;
let sortKey="p_goal", valOnly=false, view="game", mounted=false;
const MAXP=0.5;

const clamp=x=>Math.max(2,Math.min(100,x));
const pct=x=>x==null?"—":Math.round(x*100)+"%";
const pct1=x=>x==null?"—":((x>=0?"+":"")+(x*100).toFixed(1)+"%");
const am=o=>o==null?"—":((o>0?"+":"")+Math.round(o));
const fair=p=>p==null?"—":am(p>=0.5?-100*p/(1-p):100*(1-p)/p);
const isVal=p=>p.ev!=null&&p.ev>0;
const sortFn=(a,b)=>{const av=a[sortKey],bv=b[sortKey];
  if(av==null&&bv==null)return b.p_goal-a.p_goal;
  if(av==null)return 1;if(bv==null)return -1;return bv-av;};

function metaBar(){
  const m=DATA,r=m.requests_remaining;
  let badge='<span class="badge">odds pending</span>';
  if(r!=null){const n=+r,c=n>100?"good":n>30?"warnb":"lowb";
    badge=`<span class="badge ${c}">${n}/500 calls</span>`;}
  const parts=[`<span class="k">${m.date}</span>`];
  if(m.odds_pulled_at)parts.push(`<span>odds <span class="k">${m.odds_pulled_at.slice(11,16)}</span></span>`);
  else if(m.generated)parts.push(`<span>built <span class="k">${m.generated.slice(11,16)}</span></span>`);
  parts.push(badge);
  document.getElementById("meta").innerHTML=parts.join("");
  const all=DATA.games.flatMap(g=>g.players);
  const nv=all.filter(isVal).length;
  document.getElementById("summary").innerHTML=
    `<b>${DATA.games.length}</b> games · <b>${all.length}</b> skaters · `+
    (nv?`<b class="v">${nv}</b> value bet${nv>1?"s":""}`:`no value bets yet`);
}

function meterHTML(p){
  const g=clamp(p.p_goal/MAXP*100);
  const v=p.vegas_prob!=null?clamp(p.vegas_prob/MAXP*100):null;
  let ice=g,gold=0,tick=null;
  if(v!=null){tick=v; if(p.p_goal>p.vegas_prob){ice=v;gold=g-v;}else{ice=g;}}
  return `<div class="meter">
    <div class="ice" style="width:${ice}%"></div>
    ${gold>0.5?`<div class="gold" style="left:${ice}%;width:${gold}%"></div>`:""}
    ${tick!=null?`<div class="mk-tick" style="left:${tick}%"></div>`:""}
  </div>`;
}

function rowHTML(p,i,showGame){
  const val=isVal(p),d=Math.min(i,16)*42;
  const tag=`<span class="tm">${p.team}</span>`+(showGame?`<span class="gm">${p.game}</span>`:"");
  let evCell;
  if(p.vegas_odds==null) evCell=`<div class="ev none"><span class="pill">no line</span></div>`;
  else evCell=`<div class="ev ${val?'pos':'neg'}"><span class="pill">${pct1(p.ev)}</span><small>${val?'value':'ev'}</small></div>`;
  const sub=p.vegas_odds==null
    ? `<span><b>no line</b></span><span class="s">·</span><span>fair <b>${fair(p.p_goal)}</b></span><span class="s">·</span><span>pt <b>${pct(p.p_point)}</b> ${fair(p.p_point)}</span>`
    : `<span>veg <b>${am(p.vegas_odds)}</b></span><span class="s">·</span><span>fair <b>${fair(p.p_goal)}</b></span><span class="s">·</span><span>pt <b>${pct(p.p_point)}</b> ${fair(p.p_point)}</span>`;
  return `<div class="p ${val?'val':''}" style="--d:${d}ms">
    <div class="rk">${i+1}</div>
    <div class="who"><div class="nm">${p.name}</div><div class="tag">${tag}</div></div>
    ${meterHTML(p)}
    <div class="pct">${Math.round(p.p_goal*100)}%<small>score</small></div>
    ${evCell}
    <div class="sub">${sub}</div>
  </div>`;
}

function render(){
  const host=document.getElementById("board");
  const animate=!mounted; mounted=true;
  host.className="board"+(view==="all"?" flat":"")+(animate?" animate":"");
  host.innerHTML="";
  if(view==="all"){
    let all=DATA.games.flatMap(g=>g.players.map(p=>Object.assign({game:g.game},p)));
    all.sort(sortFn); if(valOnly)all=all.filter(isVal);
    const sec=document.createElement("section");sec.className="game";
    const by=sortKey==="ev"?"by value":"by goal chance";
    sec.innerHTML=`<div class="lead-h"><h2>Leaderboard</h2><span class="by">${all.length} skaters · ${by}</span></div>`
      +(all.length?all.map((p,i)=>rowHTML(p,i,true)).join(""):'<div class="empty">No value bets on the board.</div>');
    host.appendChild(sec);return;
  }
  for(const g of DATA.games){
    let ps=g.players.slice().sort(sortFn); if(valOnly)ps=ps.filter(isVal);
    if(valOnly&&!ps.length)continue;
    const nv=g.players.filter(isVal).length;
    const sec=document.createElement("section");sec.className="game";
    sec.innerHTML=`<div class="match">
        <div class="vs">${g.away}<span class="x">@</span><span class="home">${g.home}</span></div>
        <div class="mtag">${nv?`<span class="v">${nv} value</span>`:"top 5"}</div>
      </div>`+ps.map((p,i)=>rowHTML(p,i)).join("");
    host.appendChild(sec);
  }
  if(!host.children.length)host.innerHTML='<div class="empty">No value bets on the board.</div>';
}

document.querySelectorAll("[data-view]").forEach(b=>b.onclick=()=>{
  view=b.dataset.view;mounted=false;
  document.querySelectorAll("[data-view]").forEach(x=>x.setAttribute("aria-pressed",x===b));render();});
document.querySelectorAll("[data-sort]").forEach(b=>b.onclick=()=>{
  sortKey=b.dataset.sort;
  document.querySelectorAll("[data-sort]").forEach(x=>x.setAttribute("aria-pressed",x===b));render();});
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
