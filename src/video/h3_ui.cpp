// Single-page UI of h3_server (served at /).
#include "h3_ui.h"

const char* h3_ui_html() {
	return R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MiniMax-H3 on Volta</title>
<style>
:root{--bg:#1d1f21;--bg2:#282a2e;--bg3:#373b41;--fg:#c5c8c6;--dim:#969896;--acc:#81a2be;--ok:#b5bd68;--warn:#f0c674;--err:#cc6666;--pur:#b294bb;--cy:#8abeb7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{display:flex;align-items:center;gap:16px;padding:10px 18px;background:var(--bg2);border-bottom:1px solid var(--bg3)}
header h1{font-size:16px;margin:0;color:var(--acc)}header .sp{flex:1}header .tag{color:var(--dim);font-size:12px}
main{display:grid;grid-template-columns:minmax(360px,440px) 1fr;gap:16px;padding:16px;align-items:start}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--bg2);border:1px solid var(--bg3);border-radius:6px;padding:14px;margin-bottom:14px}
.card h2{font-size:13px;margin:0 0 10px;color:var(--pur);text-transform:uppercase;letter-spacing:.06em}
label{display:block;color:var(--dim);font-size:12px;margin:10px 0 4px}
textarea,input[type=text],input[type=number],select{width:100%;background:var(--bg);color:var(--fg);border:1px solid var(--bg3);border-radius:4px;padding:7px 9px;font:inherit}
textarea{min-height:110px;resize:vertical}textarea:focus,input:focus,select:focus{outline:1px solid var(--acc)}
.row{display:flex;gap:10px}.row>*{flex:1}
.drop{position:relative;border:1px dashed var(--bg3);border-radius:6px;min-height:92px;display:flex;align-items:center;justify-content:center;color:var(--dim);cursor:pointer;overflow:hidden;font-size:12px}
.drop.on{border-color:var(--acc)}.drop img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
.drop .x{position:absolute;top:4px;right:4px;background:#000a;color:#fff;border:0;border-radius:3px;cursor:pointer;display:none;z-index:2}.drop.has .x{display:block}
button{background:var(--bg3);color:var(--fg);border:1px solid #4b5059;border-radius:4px;padding:7px 12px;font:inherit;cursor:pointer}button:hover{border-color:var(--acc)}
button.go{background:var(--acc);color:#10141a;font-weight:700;border-color:var(--acc);width:100%;padding:11px;margin-top:6px}button.go:disabled{opacity:.5}
.lora{display:grid;grid-template-columns:auto 1fr 90px 40px;gap:8px;align-items:center;padding:5px 0;border-top:1px solid var(--bg3)}
.lora:first-child{border-top:0}.lora .n{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.lora .m{color:var(--dim);font-size:11px}
input[type=range]{width:100%;accent-color:var(--acc)}
.pill{display:inline-block;border-radius:10px;padding:1px 8px;font-size:11px;border:1px solid var(--bg3);color:var(--dim)}
.pill.running{color:var(--warn);border-color:var(--warn)}.pill.done{color:var(--ok);border-color:var(--ok)}.pill.failed,.pill.cancelled{color:var(--err);border-color:var(--err)}.pill.queued{color:var(--cy);border-color:var(--cy)}
.job{border:1px solid var(--bg3);border-radius:6px;padding:12px;margin-bottom:12px;background:var(--bg)}
.job .top{display:flex;gap:8px;align-items:center;margin-bottom:6px}.job .top .sp{flex:1}.job .pr{color:var(--fg);margin:4px 0 8px;max-height:3.9em;overflow:hidden}
.bar{height:6px;background:var(--bg3);border-radius:3px;overflow:hidden;margin:6px 0}.bar>i{display:block;height:100%;background:var(--acc);transition:width .4s}
.meta{color:var(--dim);font-size:12px}video{width:100%;border-radius:4px;background:#000;margin:8px 0}
.acts{display:flex;gap:8px;flex-wrap:wrap}.err{color:var(--err);white-space:pre-wrap}.small{font-size:12px;color:var(--dim)}
</style></head><body>
<header><h1>MiniMax-H3 on Volta</h1><span class="tag" id="hw"></span><span class="sp"></span><span class="tag" id="qinfo"></span></header>
<main>
<section>
 <div class="card"><h2>Prompt</h2>
  <textarea id="prompt" placeholder="Describe the scene, the motion and the sound."></textarea>
  <h2 style="margin-top:14px">Keyframes <span class="small">(optional: image to video)</span></h2>
  <div class="row"><div><div class="drop" id="d0">First frame<button class="x" title="clear">&times;</button></div></div><div><div class="drop" id="d1">Last frame<button class="x" title="clear">&times;</button></div></div></div>
  <input type="file" id="file" accept="image/*" hidden>
  <div id="refcard" style="display:none"><h2 style="margin-top:14px">References <span class="small">(Ref2VA: reuse a subject, a scene or a voice; replaces keyframes)</span></h2>
   <div id="refs" class="row" style="flex-wrap:wrap"></div>
   <div class="row" style="margin-top:6px"><button id="addimg">+ image (up to 9)</button><button id="addvid">+ video (up to 3, 2-15 s, mp4/mov)</button><button id="addaud">+ audio (up to 3, 2-15 s)</button></div>
   <input type="file" id="rfile" accept="image/*" multiple hidden><input type="file" id="afile" accept="audio/*" hidden><input type="file" id="vfile" accept="video/mp4,video/quicktime" hidden>
   <div class="small">Refer to them in the prompt as &lt;Picture 1&gt;, &lt;Audio 1&gt;, ...</div></div>
 </div>
 <div class="card"><h2>Video</h2>
  <div class="row"><div><label>Aspect ratio</label><select id="aspect"><option value="16:9">16:9 landscape</option><option value="9:16">9:16 portrait</option><option value="1:1">1:1 square</option><option value="4:3">4:3</option><option value="3:4">3:4</option><option value="21:9">21:9 wide</option></select></div>
  <div><label>Resolution</label><select id="res"></select></div></div>
  <label>Duration: <b id="durv"></b></label><input type="range" id="dur" min="1" max="15" step="0.5" value="5">
  <div class="row"><div><label>Speed / quality</label><select id="turbo"></select></div><div><label>Seed</label><div class="row"><input type="number" id="seed" value="1"><button id="rnd" title="random seed" style="flex:0 0 40px">&#9856;</button></div></div></div>
  <div class="small" id="plan" style="margin-top:8px"></div>
 </div>
 <div class="card"><h2>LoRA</h2><div id="loras" class="small">no adapters found</div></div>
 <button class="go" id="go">Generate</button>
 <div class="small" id="msg" style="margin-top:8px"></div>
</section>
<section id="jobs"></section>
</main>
<script>
const $=s=>document.querySelector(s);let info=null,imgs=[null,null],pick=0,rimgs=[],raud=[],rvid=[];
const api=async(p,o)=>{const r=await fetch(p,o);if(!r.ok){let t=await r.text();throw new Error(t||r.status)}return r};
function canvas(aspect){let w,h;if(aspect>=1){w=768*aspect;h=768}else{w=768;h=768/aspect}const a=w*h,c=768*1344;if(a>c){const s=Math.sqrt(c/a);w*=s;h*=s}const al=v=>Math.max(32,Math.round(v/32)*32);return[al(w),al(h)]}
function fillRes(){const [r,c]=$('#aspect').value.split(':').map(Number);const asp=r/c;const sizes=[];let [w,h]=canvas(asp);sizes.push([w,h,'full']);const f=[0.75,0.5,0.34];for(const k of f){const sw=Math.max(32,Math.round(w*k/32)*32),sh=Math.max(32,Math.round(h*k/32)*32);sizes.push([sw,sh,Math.round(k*100)+'%'])}
 $('#res').innerHTML=sizes.map(s=>`<option value="${s[0]}x${s[1]}">${s[0]}x${s[1]} (${s[2]})</option>`).join('');plan()}
function frames(){const want=Math.round(parseFloat($('#dur').value)*24);let f=Math.max(22,want);while(f%17!=5)f++;return f}
function renderTurbo(){if(!info)return;const ref=rimgs.length>0||rvid.length>0;const cur=$('#turbo').value;const turbos=info.loras.filter(l=>l.turbo&&((l.task==='ref2v')===ref));$('#turbo').innerHTML='<option value="">Base model (50 steps, slow)</option>'+turbos.map(l=>`<option value="${l.file}">${l.label}</option>`).join('');const keep=turbos.find(l=>l.file===cur);const def=keep||turbos.find(l=>/4step_v1\.2_768p|4step/.test(l.file))||turbos[0];if(def)$('#turbo').value=def.file}
function renderRefs(){$('#refs').innerHTML=rimgs.map((d,i)=>`<div class="drop has" style="min-height:0;height:80px;width:110px;flex:0 0 110px"><img src="${d}"><button class="x" data-ri="${i}">&times;</button><span class="small">Picture ${i+1}</span></div>`).join('')+rvid.map((a,i)=>`<div class="drop has" style="min-height:0;height:80px;width:110px;flex:0 0 110px"><span class="small">Video ${i+1}<br>${a.name}</span><button class="x" data-vi="${i}">&times;</button></div>`).join('')+raud.map((a,i)=>`<div class="drop has" style="min-height:0;height:80px;width:110px;flex:0 0 110px"><span class="small">Audio ${i+1}<br>${a.name}</span><button class="x" data-ai="${i}">&times;</button></div>`).join('');document.querySelectorAll('[data-ri]').forEach(b=>b.onclick=()=>{rimgs.splice(+b.dataset.ri,1);renderRefs();renderTurbo();plan()});document.querySelectorAll('[data-vi]').forEach(b=>b.onclick=()=>{rvid.splice(+b.dataset.vi,1);renderRefs()});document.querySelectorAll('[data-ai]').forEach(b=>b.onclick=()=>{raud.splice(+b.dataset.ai,1);renderRefs()});renderTurbo();plan()}
function turboInfo(){const t=$('#turbo').value;if(!t)return{steps:50,shift:12,name:'base model, 50 steps'};const m=t.match(/_(\d)step/);return{steps:m?+m[1]:4,shift:/_768p/.test(t)?6:12,name:t}}
function plan(){const [w,h]=$('#res').value.split('x').map(Number);const f=frames(),ti=turboInfo();const lat=((f-5)/17|0)*5+2;const rows=64+lat*(w/32)*(h/32)+Math.round(f/24*40)*2+(imgs[0]?1:0)*(w/32)*(h/32)+(imgs[1]?1:0)*(w/32)*(h/32);
 $('#durv').textContent=(f/24).toFixed(2)+' s ('+f+' frames)';$('#plan').textContent=`${ti.steps} denoising steps, flow shift ${ti.shift}, ~${rows.toLocaleString()} tokens`+(imgs[0]||imgs[1]?', keyframe conditioning':'')}
['aspect','res','dur','turbo'].forEach(i=>$('#'+i).addEventListener('input',()=>{if(i==='aspect')fillRes();plan()}));
$('#rnd').onclick=()=>{$('#seed').value=Math.floor(Math.random()*2**31)};
document.querySelectorAll('.drop').forEach((d,i)=>{d.onclick=e=>{if(e.target.classList.contains('x'))return;pick=i;$('#file').click()};
 d.querySelector('.x').onclick=e=>{e.stopPropagation();setImg(i,null)};
 d.ondragover=e=>{e.preventDefault();d.classList.add('on')};d.ondragleave=()=>d.classList.remove('on');
 d.ondrop=e=>{e.preventDefault();d.classList.remove('on');if(e.dataTransfer.files[0])load(i,e.dataTransfer.files[0])}});
$('#file').onchange=e=>{if(e.target.files[0])load(pick,e.target.files[0]);e.target.value=''};
function load(i,f){const r=new FileReader();r.onload=()=>setImg(i,r.result);r.readAsDataURL(f)}
function setImg(i,data){imgs[i]=data;const d=$('#d'+i);d.querySelector('img')?.remove();d.classList.toggle('has',!!data);d.classList.toggle('on',false);
 if(data){const im=new Image();im.src=data;d.prepend(im);if(i===0){im.onload=()=>{const a=im.naturalWidth/im.naturalHeight;let best='16:9',bd=9;for(const [k,v] of Object.entries({'16:9':16/9,'9:16':9/16,'1:1':1,'4:3':4/3,'3:4':3/4,'21:9':21/9})){const dd=Math.abs(Math.log(a/v));if(dd<bd){bd=dd;best=k}}$('#aspect').value=best;fillRes()}}}plan()}
async function boot(){info=await (await api('/api/info')).json();if(info.ref2va){$('#refcard').style.display='block';$('#addimg').onclick=()=>$('#rfile').click();$('#addaud').onclick=()=>$('#afile').click();$('#addvid').onclick=()=>$('#vfile').click();$('#vfile').onchange=e=>{const f=e.target.files[0];if(f&&rvid.length<3){const r=new FileReader();r.onload=()=>{rvid.push({name:f.name.slice(0,14),data:r.result});renderRefs()};r.readAsDataURL(f)}e.target.value=''};$('#rfile').onchange=e=>{for(const f of e.target.files){if(rimgs.length>=9)break;const r=new FileReader();r.onload=()=>{rimgs.push(r.result);renderRefs()};r.readAsDataURL(f)}e.target.value=''};$('#afile').onchange=e=>{const f=e.target.files[0];if(f&&raud.length<3){const r=new FileReader();r.onload=()=>{raud.push({name:f.name.slice(0,14),data:r.result});renderRefs()};r.readAsDataURL(f)}e.target.value=''}}$('#hw').textContent=info.devices.length+' GPU(s): '+info.devices.join(', ')+' | '+info.model;
 renderTurbo();
 const ex=info.loras.filter(l=>!l.turbo);$('#loras').innerHTML=ex.length?ex.map((l,i)=>`<div class="lora"><input type="checkbox" id="lc${i}" data-f="${l.file}"><div class="n" title="${l.file}">${l.label}<div class="m">${l.format}, rank ${l.rank||'?'}, ${l.mb} MB${l.task?' , '+l.task:''}</div></div><input type="range" id="ls${i}" min="-2" max="2" step="0.05" value="1"><span id="lv${i}">1.00</span></div>`).join(''):'no adapters found';
 ex.forEach((l,i)=>{$('#ls'+i).oninput=e=>{$('#lv'+i).textContent=(+e.target.value).toFixed(2);$('#lc'+i).checked=true}});
 fillRes();plan();poll()}
$('#go').onclick=async()=>{const [w,h]=$('#res').value.split('x').map(Number);const ti=turboInfo();const loras=[];document.querySelectorAll('.lora').forEach((r,i)=>{const c=r.querySelector('input[type=checkbox]');if(c.checked)loras.push({file:c.dataset.f,scale:+$('#ls'+i).value})});
 const images=[];if(imgs[0])images.push({data:imgs[0],frame_index:0});if(imgs[1])images.push({data:imgs[1],frame_index:imgs[0]?-1:-1});
 if(!$('#prompt').value.trim()){$('#msg').textContent='Enter a prompt.';return}
 const body={prompt:$('#prompt').value,width:w,height:h,frames:frames(),steps:ti.steps,seed:+$('#seed').value,turbo:$('#turbo').value,loras,images:(rimgs.length||rvid.length)?[]:images,ref_images:rimgs.map(data=>({data})),ref_videos:rvid.map(a=>({data:a.data})),ref_audios:raud.map(a=>({data:a.data}))};
 $('#go').disabled=true;$('#msg').textContent='';try{await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});$('#seed').value=Math.floor(Math.random()*2**31)}catch(e){$('#msg').textContent=e.message}$('#go').disabled=false;poll(true)};
const stageName={images:'encoding images and audio',text:'encoding the prompt',load:'loading the model',denoise:'denoising',decode:'decoding video and audio',mux:'writing the file'};
function fmt(s){s=Math.round(s);return s>=60?Math.floor(s/60)+'m '+(s%60)+'s':s+'s'}
let last='';async function poll(force){try{const j=await (await api('/api/jobs')).json();const q=j.filter(x=>x.state==='queued'||x.state==='running').length;$('#qinfo').textContent=q?q+' job(s) active':'idle';
 const html=j.map(x=>{const p=x.params||{};const pct=x.state==='done'?100:x.state==='running'?Math.round(x.progress*100):0;
  let body=`<div class="top"><span class="pill ${x.state}">${x.state}</span><span class="meta">#${x.id} ${p.width}x${p.height}, ${p.frames} frames, seed ${p.seed}${p.turbo?'':' , base'}</span><span class="sp"></span>`;
  body+=x.state==='running'||x.state==='queued'?`<button data-c="${x.id}">cancel</button>`:`<button data-d="${x.id}">delete</button>`;body+='</div>';
  body+=`<div class="pr">${esc(p.prompt||'')}</div>`;
  if(x.state==='running'||x.state==='queued'){body+=`<div class="bar"><i style="width:${pct}%"></i></div><div class="meta">${x.state==='queued'?'waiting in the queue':(stageName[x.stage]||x.stage)+(x.stage==='denoise'?' (step '+x.step+' of '+x.steps+')':'')+', '+fmt(x.elapsed)+' elapsed'}</div>`}
  if(x.state==='done')body+=`<video src="/api/jobs/${x.id}/video" controls loop preload="metadata" poster="/api/jobs/${x.id}/thumb"></video><div class="meta">${x.t_total?'generated in '+fmt(x.t_total)+' (text '+fmt(x.t_text)+', denoise '+fmt(x.t_denoise)+', decode '+fmt(x.t_decode)+')':''}</div><div class="acts"><a href="/api/jobs/${x.id}/video" download="h3_${x.id}.mp4"><button>download</button></a><button data-r="${x.id}">reuse settings</button></div>`;
  if(x.error)body+=`<div class="err">${esc(x.error)}</div>`;return `<div class="job">${body}</div>`}).join('');
 if(html!==last||force){const vids=[...document.querySelectorAll('video')].some(v=>!v.paused);if(!vids){$('#jobs').innerHTML=html;last=html}}window._jobs=j;
 document.querySelectorAll('[data-c]').forEach(b=>b.onclick=()=>api('/api/jobs/'+b.dataset.c+'/cancel',{method:'POST'}).then(()=>poll(true)));
 document.querySelectorAll('[data-d]').forEach(b=>b.onclick=()=>api('/api/jobs/'+b.dataset.d,{method:'DELETE'}).then(()=>poll(true)));
 document.querySelectorAll('[data-r]').forEach(b=>b.onclick=()=>{const x=window._jobs.find(y=>y.id==b.dataset.r);if(!x)return;const p=x.params;$('#prompt').value=p.prompt;$('#seed').value=p.seed;$('#turbo').value=p.turbo||'';$('#dur').value=Math.round(p.frames/24*2)/2;plan()})}catch(e){}
 setTimeout(poll,1500)}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
boot();
</script></body></html>)HTML";
}
