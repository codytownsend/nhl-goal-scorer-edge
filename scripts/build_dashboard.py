"""Generate a self-contained dashboard_<date>.html from predictions_<date>.json.

Embeds the data as JS (no server, no CORS, shareable single file). Shows our
P(goal) + fair/preferred odds next to Vegas's line, with edge and EV, +EV rows
highlighted, and the API-calls-remaining badge. Rerun any time -- it reads the
cached JSON, never the API.

Usage: python -m scripts.build_dashboard 2026-10-08
"""
import json
import sys

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NHL Goal-Scorer Edge — __DATE__</title>
<style>
/* design · archetype: dashboard · style: technical + editorial · axes: structure, signal-hue
 * brief: audience=quant bettor · decision=find +EV anytime-goal bets vs Vegas · tone=trading-terminal
 * palette: ink oklch(16% .02 260) · accent-pos oklch(72% .17 150) · type: system grotesk + ui-monospace */
:root{
  --ink:oklch(15% .018 260); --panel:oklch(19% .02 260); --panel-2:oklch(22% .022 260);
  --line:oklch(31% .02 260); --line-soft:oklch(26% .018 260);
  --fg:oklch(94% .01 260); --fg-dim:oklch(72% .015 260); --fg-faint:oklch(58% .015 260);
  --pos:oklch(74% .17 150); --pos-bg:oklch(42% .11 150); --neg:oklch(66% .17 25); --warn:oklch(80% .13 85);
  --accent:oklch(80% .12 230);
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --sp:8px; --radius:10px;
}
*{box-sizing:border-box}
html,body{overflow-x:clip}
body{margin:0;background:var(--ink);color:var(--fg);font-family:var(--sans);
  font-size:14px;line-height:1.45;-webkit-font-smoothing:antialiased;
  padding:clamp(16px,3vw,32px);}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums;letter-spacing:-.01em}

header{max-width:1200px;margin:0 auto clamp(20px,3vw,32px);}
h1{font-size:clamp(20px,3.5vw,30px);font-weight:650;letter-spacing:-.02em;margin:0 0 6px}
h1 .dot{color:var(--pos)}
.meta{display:flex;flex-wrap:wrap;gap:6px 18px;color:var(--fg-faint);font-size:12.5px}
.meta b{color:var(--fg-dim);font-weight:500}
.badge{font-family:var(--mono);font-size:12px;padding:3px 9px;border-radius:999px;
  border:1px solid var(--line);color:var(--fg-dim);white-space:nowrap}
.badge.good{color:var(--pos);border-color:color-mix(in oklch,var(--pos),transparent 55%)}
.badge.warnb{color:var(--warn);border-color:color-mix(in oklch,var(--warn),transparent 55%)}
.badge.lowb{color:var(--neg);border-color:color-mix(in oklch,var(--neg),transparent 45%)}

.controls{max-width:1200px;margin:0 auto 18px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.controls .lbl{color:var(--fg-faint);font-size:12px;margin-right:2px}
button.seg{font-family:var(--sans);font-size:12.5px;color:var(--fg-dim);background:var(--panel);
  border:1px solid var(--line);padding:6px 11px;border-radius:7px;cursor:pointer;transition:.12s}
button.seg:hover{border-color:var(--fg-faint)}
button.seg[aria-pressed=true]{color:var(--ink);background:var(--fg);border-color:var(--fg)}
button.seg:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

main{max-width:1200px;margin:0 auto;display:grid;gap:clamp(14px,2vw,20px);
  grid-template-columns:repeat(auto-fill,minmax(min(100%,440px),1fr))}
.card{background:var(--panel);border:1px solid var(--line-soft);border-radius:var(--radius);overflow:hidden}
.card h2{margin:0;padding:11px 14px;font-size:13px;font-weight:600;letter-spacing:.02em;
  color:var(--fg);background:var(--panel-2);border-bottom:1px solid var(--line-soft);
  display:flex;justify-content:space-between;align-items:baseline}
.card h2 .away{color:var(--fg-dim)}
.tbl{width:100%;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
thead th{position:sticky;top:0;text-align:right;font-weight:500;color:var(--fg-faint);
  font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;padding:8px 10px;
  border-bottom:1px solid var(--line-soft);white-space:nowrap;background:var(--panel)}
thead th:first-child{text-align:left}
tbody td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line-soft);white-space:nowrap}
tbody td:first-child{text-align:left;font-weight:500}
tbody tr:last-child td{border-bottom:none}
.team{color:var(--fg-faint);font-size:11px;font-family:var(--mono);margin-left:6px}
.pg{color:var(--fg)} .pt{color:var(--fg-faint);font-size:11.5px}
.vg,.impl{color:var(--fg-dim)} .fair{color:var(--fg-dim)}
.edge.p,.ev.p{color:var(--pos)} .edge.n,.ev.n{color:var(--neg)} .muted{color:var(--fg-faint)}
tr.pos{background:linear-gradient(90deg,color-mix(in oklch,var(--pos-bg),transparent 78%),transparent 60%);
  box-shadow:inset 3px 0 0 var(--pos)}
tr.pos td:first-child{color:var(--fg)}
.legend{max-width:1200px;margin:22px auto 0;color:var(--fg-faint);font-size:11.5px;line-height:1.6;
  border-top:1px solid var(--line-soft);padding-top:14px}
.legend code{font-family:var(--mono);color:var(--fg-dim)}
.empty{color:var(--fg-faint);padding:20px;text-align:center;font-size:13px}
</style>
</head>
<body>
<header>
  <h1>NHL Goal-Scorer Edge<span class="dot">.</span></h1>
  <div class="meta" id="meta"></div>
</header>
<div class="controls">
  <span class="lbl">sort</span>
  <button class="seg" data-sort="ev">EV</button>
  <button class="seg" data-sort="edge">Edge</button>
  <button class="seg" data-sort="p_goal" aria-pressed="true">P(goal)</button>
  <span class="lbl" style="margin-left:10px">filter</span>
  <button class="seg" data-filter="pos">+EV only</button>
</div>
<main id="board"></main>
<div class="legend">
  <b style="color:var(--fg-dim)">P(goal)/P(pt)</b> our ensemble model (rate + play-sequence GRU). ·
  <b style="color:var(--fg-dim)">Fair</b> the American odds our P(goal) implies (break-even price). ·
  <b style="color:var(--fg-dim)">Vegas</b> median anytime-scorer line (includes vig). ·
  <b style="color:var(--fg-dim)">Edge</b> our P − Vegas implied. ·
  <b style="color:var(--fg-dim)">EV</b> <code>P(goal)·decimal − 1</code> per $1 at Vegas's price. +EV rows glow.
  <br>Roster-based before lineups post; opponent goalie not start-adjusted; GRU single-seed (±1–2pt). Not betting advice.
</div>
<script>
const DATA = __DATA__;
let sortKey="p_goal", posOnly=false;

function pct(x){return x==null?"—":(x*100).toFixed(1)+"%";}
function am(o){if(o==null)return "—";o=Math.round(o);return (o>0?"+":"")+o;}
function fair(p){if(p==null)return "—";let o=p>=0.5?-100*p/(1-p):100*(1-p)/p;return am(o);}
function signed(x,isPct){if(x==null)return "—";const v=(x*100);return (v>=0?"+":"")+v.toFixed(1)+(isPct?"%":"");}
function cls(x){return x==null?"muted":(x>0?"p":"n");}

function render(){
  const meta=DATA;
  const badge=(()=>{
    const r=meta.requests_remaining;
    if(r==null) return '<span class="badge">odds not pulled</span>';
    const n=+r, c=n>100?"good":n>30?"warnb":"lowb";
    return `<span class="badge ${c}">${n} / 500 API calls left</span>`;
  })();
  document.getElementById("meta").innerHTML=
    `<span><b>${meta.date}</b> slate · ${meta.games.length} games</span>`+
    `<span>generated <b>${(meta.generated||"").replace("T"," ")}</b></span>`+
    (meta.odds_pulled_at?`<span>odds <b>${meta.odds_pulled_at.replace("T"," ")}</b></span>`:"")+
    `<span>${meta.odds_source||"our model only — no odds yet"}</span>`+badge;

  const board=document.getElementById("board");board.innerHTML="";
  for(const g of meta.games){
    let ps=g.players.slice();
    ps.sort((a,b)=>{
      const av=a[sortKey], bv=b[sortKey];
      if(av==null&&bv==null)return b.p_goal-a.p_goal;
      if(av==null)return 1; if(bv==null)return -1; return bv-av;
    });
    if(posOnly) ps=ps.filter(p=>p.ev!=null&&p.ev>0);
    const card=document.createElement("section");card.className="card";
    const rows=ps.map(p=>{
      const isPos=p.ev!=null&&p.ev>0;
      return `<tr class="${isPos?'pos':''}">
        <td>${p.name}<span class="team">${p.team}</span></td>
        <td class="num pg">${pct(p.p_goal)}</td>
        <td class="num fair">${fair(p.p_goal)}</td>
        <td class="num vg">${am(p.vegas_odds)}</td>
        <td class="num impl">${pct(p.vegas_prob)}</td>
        <td class="num edge ${cls(p.edge)}">${signed(p.edge,true)}</td>
        <td class="num ev ${cls(p.ev)}">${signed(p.ev,true)}</td>
        <td class="num pt">${pct(p.p_point)}</td></tr>`;
    }).join("");
    card.innerHTML=`<h2><span><span class="away">${g.away}</span> @ ${g.home}</span></h2>
      <div class="tbl"><table><thead><tr>
        <th>Player</th><th>P(goal)</th><th>Fair</th><th>Vegas</th>
        <th>Impl</th><th>Edge</th><th>EV</th><th>P(pt)</th>
      </tr></thead><tbody>${rows||'<tr><td colspan="8" class="empty">no +EV plays</td></tr>'}</tbody></table></div>`;
    board.appendChild(card);
  }
}
document.querySelectorAll("[data-sort]").forEach(b=>b.onclick=()=>{
  sortKey=b.dataset.sort;
  document.querySelectorAll("[data-sort]").forEach(x=>x.setAttribute("aria-pressed",x===b));
  render();
});
const fb=document.querySelector("[data-filter]");
fb.onclick=()=>{posOnly=!posOnly;fb.setAttribute("aria-pressed",posOnly);render();};
render();
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
