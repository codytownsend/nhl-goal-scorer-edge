"""Generate a self-contained dashboard_<date>.html from predictions_<date>.json.

Design: a light "betting sheet" — a readable data table (Console archetype) on warm
paper. Big serif chance %, hairline rules, a single green value-accent for +EV.
Two views (by game / leaderboard), sort + value filter. Reflows to stacked cards
on mobile. Data embedded as a JS const (no server, no external data fetch).

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
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,500;0,6..72,600;1,6..72,500&family=Hanken+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* design · archetype: console · style: swiss/systematic + editorial · axes: form, texture
 * brief: audience=sharp bettor · decision=who scores + where's value vs Vegas · tone=precise, editorial
 * palette: paper oklch(98.5% .006 92) · ink oklch(26% .012 72) · value oklch(50% .15 150)
 * type: Newsreader (display/figures) + Hanken Grotesk (ui) */
:root{
  --paper:oklch(98.6% .006 92); --paper-2:oklch(96.8% .007 92); --paper-3:oklch(94.2% .009 92);
  --ink:oklch(26% .012 72); --ink-2:oklch(40% .012 76); --muted:oklch(52% .011 80);
  --faint:oklch(64% .009 84);
  --line:oklch(88% .008 88); --line-strong:oklch(80% .01 84);
  --val:oklch(48% .15 150); --val-ink:oklch(40% .13 150);
  --val-bg:oklch(94% .055 150); --val-line:oklch(72% .11 150);
  --bar:oklch(42% .03 220); --bar-track:oklch(92% .006 88);
  --serif:"Newsreader",Georgia,serif; --sans:"Hanken Grotesk",system-ui,sans-serif;
  --ease:cubic-bezier(.2,.7,.2,1); --r:10px;
}
*{box-sizing:border-box}
html,body{overflow-x:clip}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased;
  padding:clamp(18px,3.5vw,40px) clamp(14px,3.5vw,40px) 64px}
.num{font-variant-numeric:tabular-nums;font-feature-settings:"tnum"}
.wrap{max-width:1080px;margin:0 auto;opacity:0;animation:in .4s var(--ease) forwards}
@keyframes in{to{opacity:1}}
@media (prefers-reduced-motion:reduce){.wrap{animation:none;opacity:1}}

/* ---------- masthead ---------- */
header{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:flex-end;gap:14px 28px}
.mast h1{font-family:var(--serif);font-weight:600;font-size:clamp(30px,5.5vw,46px);
  line-height:1;letter-spacing:-.015em;margin:0}
.mast .dek{font-family:var(--serif);font-style:italic;font-size:15px;color:var(--muted);
  margin:7px 0 0}
.edition{text-align:right;font-size:12px;color:var(--muted);line-height:1.7}
.edition .k{color:var(--ink-2);font-weight:600}
.badge{display:inline-block;margin-top:3px;font-size:11px;font-weight:700;padding:3px 9px;
  border-radius:5px;border:1px solid var(--line-strong);color:var(--ink-2);
  font-variant-numeric:tabular-nums}
.badge.good{color:var(--val);border-color:var(--val-line);background:var(--val-bg)}
.badge.warnb{color:oklch(50% .12 60);border-color:oklch(78% .1 70)}
.badge.lowb{color:oklch(52% .14 28);border-color:oklch(78% .11 30)}
.rule{border:none;border-top:2px solid var(--ink);margin:14px 0 0}

/* ---------- control bar ---------- */
.bar{display:flex;flex-wrap:wrap;align-items:center;gap:12px 20px;margin:16px 0 24px}
.summary{font-size:13px;color:var(--muted)}
.summary b{color:var(--ink);font-weight:700}
.summary .v{color:var(--val);font-weight:700}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:18px;margin-left:auto}
.tabs{display:inline-flex;gap:3px}
.tabs button{font-family:var(--sans);font-size:13px;font-weight:600;color:var(--muted);
  background:none;border:none;border-bottom:2px solid transparent;padding:5px 2px;cursor:pointer;
  min-height:30px;transition:.14s var(--ease)}
.tabs button:hover{color:var(--ink)}
.tabs button[aria-pressed=true]{color:var(--ink);border-bottom-color:var(--ink)}
.tabs button:focus-visible{outline:2px solid var(--val);outline-offset:3px;border-radius:2px}
.tabs .lbl{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;
  color:var(--faint);align-self:center;margin-right:3px}
.vfilter{display:inline-flex;align-items:center;gap:8px;font-size:12.5px;font-weight:600;
  color:var(--muted);cursor:pointer}
.vfilter input{appearance:none;width:38px;height:22px;border-radius:999px;background:var(--paper-3);
  border:1px solid var(--line-strong);position:relative;cursor:pointer;transition:.14s var(--ease);flex:none}
.vfilter input::after{content:"";position:absolute;top:2px;left:2px;width:16px;height:16px;border-radius:50%;
  background:var(--paper);box-shadow:0 1px 2px oklch(0% 0 0 /.3);transition:.14s var(--ease)}
.vfilter input:checked{background:var(--val);border-color:var(--val)}
.vfilter input:checked::after{transform:translateX(16px)}
.vfilter input:focus-visible{outline:2px solid var(--val);outline-offset:2px}

/* ---------- sections ---------- */
.board{display:grid;gap:26px 24px;grid-template-columns:repeat(auto-fill,minmax(min(100%,490px),1fr))}
.board.flat{grid-template-columns:1fr}
.ghead{display:flex;align-items:baseline;justify-content:space-between;gap:12px;
  padding-bottom:7px;border-bottom:1.5px solid var(--line-strong);margin-bottom:2px}
.ghead h2{font-family:var(--serif);font-weight:600;font-size:19px;letter-spacing:-.01em;margin:0}
.ghead h2 .at{color:var(--faint);font-style:italic;padding:0 4px}
.ghead h2 .home{color:var(--ink-2)}
.ghead .n{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--faint)}
.ghead .n.v{color:var(--val)}

/* ---------- table ---------- */
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
thead th{font-family:var(--sans);font-size:10px;font-weight:700;text-transform:uppercase;
  letter-spacing:.07em;color:var(--faint);text-align:right;padding:9px 10px 7px;white-space:nowrap;
  border-bottom:1px solid var(--line)}
thead th.l{text-align:left}
tbody td{padding:11px 10px;border-bottom:1px solid var(--line);text-align:right;vertical-align:middle}
tbody tr:last-child td{border-bottom:none}
tbody tr.val td{background:var(--val-bg)}
tbody tr.val td.rk{box-shadow:inset 3px 0 0 var(--val)}
td.rk{text-align:center;font-family:var(--serif);font-weight:600;font-size:15px;color:var(--faint);width:30px}
td.who{text-align:left;line-height:1.25}
.nm{font-weight:600;font-size:15px;color:var(--ink)}
.sub{font-size:11px;color:var(--faint);margin-top:1px}
.sub .gm{color:var(--ink-2);font-weight:600}
/* chance: bar + big % */
td.chance{white-space:nowrap}
.cwrap{display:inline-flex;align-items:center;gap:10px;justify-content:flex-end}
.track{width:62px;height:7px;border-radius:999px;background:var(--bar-track);overflow:hidden;flex:none}
.track i{display:block;height:100%;border-radius:999px;background:var(--bar)}
.pg{font-family:var(--serif);font-weight:600;font-size:19px;color:var(--ink);min-width:40px;text-align:right}
.cfair{font-size:10.5px;color:var(--faint);margin-top:1px}
/* market */
.odds{font-weight:600;font-size:14px;color:var(--ink)}
.impl{font-size:10.5px;color:var(--faint);margin-top:1px}
/* edge + ev */
td.edge .e{font-weight:600;font-size:14px;color:var(--ink-2)}
td.edge .e.p{color:var(--val)}
.pill{display:inline-block;font-weight:700;font-size:13px;padding:3px 9px;border-radius:999px;
  color:var(--ink-2);background:var(--paper-3)}
.pill.p{color:var(--paper);background:var(--val)}
.pill.none{color:var(--faint);background:none;border:1px dashed var(--line-strong);font-size:11px;font-weight:600}
td.pt .v{font-weight:600;font-size:14px;color:var(--ink)}
td.pt .f{font-size:10.5px;color:var(--faint);margin-top:1px}
td.det{display:none}
.dash{color:var(--faint)}

.lead h2{font-family:var(--serif);font-weight:600;font-size:22px;margin:0 0 2px}
.empty{padding:28px 10px;text-align:center;color:var(--faint)}

/* ---------- legend ---------- */
.legend{max-width:1080px;margin:32px auto 0;padding-top:16px;border-top:1px solid var(--line);
  font-size:12px;color:var(--muted);line-height:1.8}
.legend b{color:var(--ink-2)}
.legend .sw{display:inline-block;width:20px;height:7px;border-radius:999px;background:var(--bar);
  vertical-align:middle;margin-right:5px}
.legend .sw.v{width:12px;height:12px;border-radius:3px;background:var(--val)}

/* ---------- mobile: table -> cards ---------- */
@media (max-width:700px){
  .board{gap:22px}
  thead{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
  table,tbody,tr,td{display:block}
  tbody tr{border:1px solid var(--line);border-radius:var(--r);padding:13px 15px;margin-top:10px;
    display:grid;grid-template-columns:1fr auto;column-gap:12px;row-gap:10px;
    grid-template-areas:"who ev" "chance chance" "det det"}
  tbody tr.val{border-color:var(--val-line);background:var(--val-bg)}
  tbody td{padding:0;border:none;text-align:left;background:none}
  td.rk,td.market,td.edge,td.pt{display:none}
  td.who{grid-area:who}
  td.ev{grid-area:ev;justify-self:end;align-self:start}
  td.chance{grid-area:chance}
  td.chance .cwrap{display:flex;justify-content:flex-start;gap:14px}
  td.chance .track{height:9px}
  td.chance .cfair{display:none}
  .track{flex:1;width:auto}
  td.det{display:block;grid-area:det}
  .detline{font-size:12.5px;color:var(--muted);display:flex;flex-wrap:wrap;gap:4px 16px}
  .detline b{color:var(--ink-2);font-weight:600}
}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="mast">
    <h1>Goal-Scorer Edge</h1>
    <p class="dek">Model vs. the market — anytime goal scorer</p>
  </div>
  <div class="edition" id="edition"></div>
</header>
<hr class="rule">

<div class="bar">
  <div class="summary" id="summary"></div>
  <div class="controls">
    <div class="tabs" role="group" aria-label="view">
      <button data-view="game" aria-pressed="true">By game</button>
      <button data-view="all" aria-pressed="false">Leaderboard</button>
    </div>
    <div class="tabs" role="group" aria-label="sort">
      <span class="lbl">Sort</span>
      <button data-sort="p_goal" aria-pressed="true">Likely</button>
      <button data-sort="ev" aria-pressed="false">Value</button>
    </div>
    <label class="vfilter"><input type="checkbox" id="valonly">Value only</label>
  </div>
</div>

<main class="board" id="board"></main>

<div class="legend">
  <span class="sw"></span><b>Chance</b> our model's probability the player scores (bar scaled to 50%). ·
  <span class="sw v"></span><b>Value</b> a +EV bet: our chance beats the Vegas price.
  <b>Edge</b> = our chance − Vegas implied. <b>EV</b> = return per $1 at the Vegas price.
  <b>Fair</b> = the odds our probability implies.
  <br>Top 5 per game · roster-based before lineups post · opponent goalie not start-adjusted · not betting advice.
</div>
</div>

<script>
const DATA = __DATA__;
let sortKey="p_goal", valOnly=false, view="game";
const MAXP=0.5;

const clamp=x=>Math.max(3,Math.min(100,x));
const pct=x=>x==null?"—":Math.round(x*100)+"%";
const ev1=x=>x==null?"—":((x>=0?"+":"")+(x*100).toFixed(1)+"%");
const edge1=x=>x==null?"—":((x>=0?"+":"")+(x*100).toFixed(1));
const am=o=>o==null?"—":((o>0?"+":"")+Math.round(o));
const fair=p=>p==null?"—":am(p>=0.5?-100*p/(1-p):100*(1-p)/p);
const isVal=p=>p.ev!=null&&p.ev>0;
const sortFn=(a,b)=>{const av=a[sortKey],bv=b[sortKey];
  if(av==null&&bv==null)return b.p_goal-a.p_goal;
  if(av==null)return 1;if(bv==null)return -1;return bv-av;};

function head(){
  const m=DATA,r=m.requests_remaining;
  let badge='<span class="badge">odds pending</span>';
  if(r!=null){const n=+r,c=n>100?"good":n>30?"warnb":"lowb";badge=`<span class="badge ${c}">${n} / 500 calls</span>`;}
  const t=m.odds_pulled_at?`odds <span class="k">${m.odds_pulled_at.slice(11,16)}</span>`
         :(m.generated?`built <span class="k">${m.generated.slice(11,16)}</span>`:"");
  document.getElementById("edition").innerHTML=
    `<div><span class="k">${m.date}</span></div><div>${t}${m.odds_source?" · the-odds-api":""}</div>${badge}`;
  const all=DATA.games.flatMap(g=>g.players), nv=all.filter(isVal).length;
  document.getElementById("summary").innerHTML=
    `<b>${DATA.games.length}</b> games &nbsp;·&nbsp; <b>${all.length}</b> skaters &nbsp;·&nbsp; `+
    (nv?`<span class="v">${nv} value bet${nv>1?"s":""}</span>`:`no value bets yet`);
}

function rowHTML(p,i,showGame){
  const val=isVal(p);
  const w=clamp(p.p_goal/MAXP*100);
  const sub=`<span class="tm">${p.team}</span>`+(showGame?` · <span class="gm">${p.game}</span>`:"");
  const market=p.vegas_odds==null
    ? `<span class="dash">—</span>`
    : `<div class="odds">${am(p.vegas_odds)}</div><div class="impl">${pct(p.vegas_prob)} impl</div>`;
  const edge=p.edge==null?`<span class="dash">—</span>`:`<span class="e ${p.edge>0?'p':''}">${edge1(p.edge)}</span>`;
  const evc=p.ev==null?`<span class="pill none">no line</span>`
    :`<span class="pill ${val?'p':''}">${ev1(p.ev)}</span>`;
  // mobile detail cell (a real <td>, hidden on desktop)
  const det=`<td class="det"><div class="detline">`+
    (p.vegas_odds==null?`<span>no line</span>`
      :`<span>line <b>${am(p.vegas_odds)}</b> (${pct(p.vegas_prob)})</span><span>edge <b>${edge1(p.edge)}</b></span>`)+
    `<span>fair <b>${fair(p.p_goal)}</b></span>`+
    `<span>point <b>${pct(p.p_point)}</b> (${fair(p.p_point)})</span></div></td>`;
  return `<tr class="${val?'val':''}">
    <td class="rk num">${i+1}</td>
    <td class="who"><div class="nm">${p.name}</div><div class="sub">${sub}</div></td>
    <td class="chance"><div class="cwrap"><span class="track"><i style="width:${w}%"></i></span>
      <span class="pg num">${Math.round(p.p_goal*100)}%</span></div><div class="cfair">fair ${fair(p.p_goal)}</div></td>
    <td class="market num">${market}</td>
    <td class="edge num">${edge}</td>
    <td class="ev num">${evc}</td>
    <td class="pt num"><div class="v">${pct(p.p_point)}</div><div class="f">${fair(p.p_point)}</div></td>
    ${det}
  </tr>`;
}

function tableHTML(rows){
  return `<table><thead><tr>
    <th class="l">#</th><th class="l">Skater</th><th>Chance</th>
    <th>Market</th><th>Edge</th><th>EV</th><th>Point</th>
  </tr></thead><tbody>${rows}</tbody></table>`;
}

function render(){
  const host=document.getElementById("board");
  host.className="board"+(view==="all"?" flat":"");host.innerHTML="";
  if(view==="all"){
    let all=DATA.games.flatMap(g=>g.players.map(p=>Object.assign({game:g.game},p)));
    all.sort(sortFn);if(valOnly)all=all.filter(isVal);
    const by=sortKey==="ev"?"by value":"by goal chance";
    const sec=document.createElement("section");sec.className="game lead";
    sec.innerHTML=`<div class="ghead"><h2>Leaderboard</h2><span class="n">${all.length} skaters · ${by}</span></div>`
      +(all.length?tableHTML(all.map((p,i)=>rowHTML(p,i,true)).join("")):'<div class="empty">No value bets on the board.</div>');
    host.appendChild(sec);return;
  }
  for(const g of DATA.games){
    let ps=g.players.slice().sort(sortFn);if(valOnly)ps=ps.filter(isVal);
    if(valOnly&&!ps.length)continue;
    const nv=g.players.filter(isVal).length;
    const sec=document.createElement("section");sec.className="game";
    sec.innerHTML=`<div class="ghead">
        <h2>${g.away}<span class="at">@</span><span class="home">${g.home}</span></h2>
        <span class="n ${nv?'v':''}">${nv?nv+" value":"top 5"}</span>
      </div>`+tableHTML(ps.map((p,i)=>rowHTML(p,i)).join(""));
    host.appendChild(sec);
  }
  if(!host.children.length)host.innerHTML='<div class="empty">No value bets on the board.</div>';
}

document.querySelectorAll("[data-view]").forEach(b=>b.onclick=()=>{view=b.dataset.view;
  document.querySelectorAll("[data-view]").forEach(x=>x.setAttribute("aria-pressed",x===b));render();});
document.querySelectorAll("[data-sort]").forEach(b=>b.onclick=()=>{sortKey=b.dataset.sort;
  document.querySelectorAll("[data-sort]").forEach(x=>x.setAttribute("aria-pressed",x===b));render();});
document.getElementById("valonly").onchange=e=>{valOnly=e.target.checked;render();};
head();render();
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
