// Single-page UI of the vall-e speech endpoint (served at /tts, and at / when the server serves speech alone), in the video page's style.
// It reads /api/tts/info, posts /v1/audio/speech with response_format "json" (an uploaded voice goes as voice_audio, base64), and keeps the
// results of this session in the page.

namespace engine::server {

const char* tts_page_html() {
	return R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>vall-e on Volta</title>
<style>
:root{color-scheme:dark;--bg:#1d1f21;--bg2:#282a2e;--bg3:#373b41;--fg:#c5c8c6;--dim:#969896;--acc:#81a2be;--ok:#b5bd68;--warn:#f0c674;--err:#cc6666;--pur:#b294bb;--cy:#8abeb7}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{display:flex;align-items:center;gap:16px;padding:10px 18px;background:var(--bg2);border-bottom:1px solid var(--bg3)}
header h1{font-size:16px;margin:0;color:var(--acc)}header .sp{flex:1}header .tag{color:var(--dim);font-size:12px}
main{display:grid;grid-template-columns:minmax(360px,440px) 1fr;gap:16px;padding:16px;align-items:start}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--bg2);border:1px solid var(--bg3);border-radius:6px;padding:14px;margin-bottom:14px}
.card h2{font-size:13px;margin:0 0 10px;color:var(--pur);text-transform:uppercase;letter-spacing:.06em}
label{display:block;color:var(--dim);font-size:12px;margin:10px 0 4px}
textarea,input[type=text],input[type=number],input[type=password],select{width:100%;background:var(--bg);color:var(--fg);border:1px solid var(--bg3);border-radius:4px;padding:7px 9px;font:inherit}
textarea{min-height:130px;resize:vertical}textarea:focus,input:focus,select:focus{outline:1px solid var(--acc)}
.row{display:flex;gap:10px}.row>*{flex:1}
.drop{position:relative;border:1px dashed var(--bg3);border-radius:6px;min-height:64px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px;color:var(--dim);cursor:pointer;font-size:12px;padding:8px}
.drop.on{border-color:var(--acc)}.drop audio{width:100%}
.drop .x{position:absolute;top:4px;right:4px;background:#000a;color:#fff;border:0;border-radius:3px;cursor:pointer;display:none;z-index:2}.drop.has .x{display:block}
button{background:var(--bg3);color:var(--fg);border:1px solid #4b5059;border-radius:4px;padding:7px 12px;font:inherit;cursor:pointer}button:hover{border-color:var(--acc)}
button.go{background:var(--acc);color:#10141a;font-weight:700;border-color:var(--acc);width:100%;padding:11px;margin-top:10px}button.go:disabled{opacity:.5}
input[type=range]{width:100%;accent-color:var(--acc)}input[type=checkbox]{accent-color:var(--acc)}
details summary{cursor:pointer;color:var(--pur);font-size:13px;text-transform:uppercase;letter-spacing:.06em}
.pill{display:inline-block;border-radius:10px;padding:1px 8px;font-size:11px;border:1px solid var(--bg3);color:var(--dim)}
.pill.running{color:var(--warn);border-color:var(--warn)}.pill.done{color:var(--ok);border-color:var(--ok)}.pill.failed{color:var(--err);border-color:var(--err)}
.job{border:1px solid var(--bg3);border-radius:6px;padding:12px;margin-bottom:12px;background:var(--bg)}
.job .top{display:flex;gap:8px;align-items:center;margin-bottom:6px}.job .top .sp{flex:1}.job .pr{color:var(--fg);margin:4px 0 8px;max-height:4.4em;overflow:hidden;white-space:pre-wrap}
.job audio{display:block;width:100%;margin:8px 0}
.meta{color:var(--dim);font-size:12px}.acts{display:flex;gap:8px;flex-wrap:wrap}.err{color:var(--err);white-space:pre-wrap}.small{font-size:12px;color:var(--dim)}
a{color:var(--acc)}
</style></head><body>
<header><h1>vall-e on Volta</h1><span class="tag" id="model"></span><span class="sp"></span><span class="tag" id="qinfo"></span></header>
<main>
<section>
 <div class="card"><h2>Text</h2>
  <textarea id="text" placeholder="What to say. Long texts are spoken sentence by sentence."></textarea>
  <div class="row"><div><label>Language</label><select id="lang"><option value="en">English</option><option value="de">German</option><option value="fr">French</option><option value="ko">Korean</option><option value="zh">Mandarin</option><option value="ja">Japanese (kana)</option></select></div>
  <div><label>Speed: <b id="speedv"></b></label><input type="range" id="speed" min="0.5" max="2" step="0.05" value="1"></div></div>
 </div>
 <div class="card"><h2>Voice</h2>
  <label>Voice</label><select id="voice"><option value="">(no voice prompt)</option></select>
  <label>or a recording of the voice <span class="small">(WAV, any rate; the first 10 s are used)</span></label>
  <div class="drop" id="drop"><span id="droptxt">Drop a WAV here, or click</span><button class="x" id="dropx" title="clear">&times;</button></div>
  <input type="file" id="file" accept="audio/wav,audio/x-wav,.wav" hidden>
 </div>
 <div class="card"><details id="adv"><summary>Sampling</summary>
  <div class="row"><div><label>Seed <span class="small">(empty: random)</span></label><div class="row"><input type="number" id="seed" min="0" placeholder="random"><button id="rnd" title="random seed" style="flex:0 0 40px">&#9856;</button></div></div>
  <div><label>Duration, s <span class="small">(empty: predicted)</span></label><input type="number" id="duration" min="0.5" max="300" step="0.1" placeholder="predicted"></div></div>
  <div class="row"><div><label>Demasking steps</label><input type="number" id="steps" min="1" max="100" value="25"></div><div><label>Flow steps</label><input type="number" id="flow_steps" min="1" max="64" value="8"></div></div>
  <div class="row"><div><label>Guidance <span class="small" id="cfgdef"></span></label><input type="number" id="cfg" min="0" max="20" step="0.5" placeholder=""></div>
  <div><label>Speaker guidance <span class="small" id="cpdef"></span></label><input type="number" id="cfg_prompt" min="0" max="10" step="0.5" placeholder=""></div></div>
  <div class="row"><div><label>Temperature</label><input type="number" id="temperature" min="0" max="2" step="0.05" value="1"></div><div><label>Sway</label><input type="number" id="sway" min="-1" max="1" step="0.1" value="-1"></div></div>
  <div class="row"><div><label>Pause between sentences, s</label><input type="number" id="pause" min="0" max="5" step="0.05" value="0.25"></div>
  <div><label>&nbsp;</label><label style="margin:0"><input type="checkbox" id="trim" checked> trim trailing silence</label></div></div>
  <div class="small" style="margin-top:8px">Empty guidance fields use the model's own (a distilled model runs unguided).</div>
 </details></div>
 <div class="card" id="keycard" style="display:none"><h2>API key</h2><input type="password" id="key" placeholder="the server's api_key"><div class="small">Kept in this browser.</div></div>
 <button class="go" id="go">Speak</button>
</section>
<section><div class="card"><h2>Results</h2><div id="jobs"><div class="small">Nothing yet.</div></div></div></section>
</main>
<script>
const $=id=>document.getElementById(id);
let info={},voiceB64=null,voiceName='',n=0,busy=0;
const esc=s=>s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function upd(){$('speedv').textContent=(+$('speed').value).toFixed(2)+'x';$('qinfo').textContent=busy?busy+' running':''}
$('speed').oninput=upd;
$('rnd').onclick=()=>{$('seed').value=Math.floor(Math.random()*2**31)};
$('key').value=localStorage.getItem('tts_key')||'';$('key').onchange=()=>localStorage.setItem('tts_key',$('key').value);
fetch('/api/tts/info').then(r=>r.json()).then(j=>{info=j;
 $('model').textContent=j.model+' · '+j.layers+' layers, d '+j.d_model;
 $('cfgdef').textContent='(model: '+j.cfg+')';$('cpdef').textContent='(model: '+j.cfg_prompt+')';
 $('cfg').placeholder=j.cfg;$('cfg_prompt').placeholder=j.cfg_prompt;
 for(const v of j.voices){const o=document.createElement('option');o.value=v;o.textContent=v;$('voice').appendChild(o)}
 if(j.key_required)$('keycard').style.display='';
}).catch(()=>{$('model').textContent='(no /api/tts/info)'});
// a dropped / chosen recording replaces the named voice
const drop=$('drop');
function setVoice(f){if(!f)return;const r=new FileReader();r.onload=()=>{voiceB64=r.result;voiceName=f.name;
 drop.classList.add('has');$('droptxt').innerHTML=esc(f.name)+'<audio controls src="'+r.result+'"></audio>';$('voice').value=''};r.readAsDataURL(f)}
function clearVoice(e){if(e)e.stopPropagation();voiceB64=null;voiceName='';drop.classList.remove('has');$('droptxt').textContent='Drop a WAV here, or click';$('file').value=''}
drop.onclick=e=>{if(e.target.tagName!=='AUDIO')$('file').click()};$('file').onchange=()=>setVoice($('file').files[0]);$('dropx').onclick=clearVoice;
drop.ondragover=e=>{e.preventDefault();drop.classList.add('on')};drop.ondragleave=()=>drop.classList.remove('on');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('on');setVoice(e.dataTransfer.files[0])};
$('voice').onchange=()=>{if($('voice').value)clearVoice()};
function num(id){const v=$(id).value;return v===''?undefined:+v}
function request(){const b={input:$('text').value.trim(),language:$('lang').value,speed:+$('speed').value,response_format:'json',
 steps:num('steps'),flow_steps:num('flow_steps'),cfg:num('cfg'),cfg_prompt:num('cfg_prompt'),temperature:num('temperature'),sway:num('sway'),
 pause:num('pause'),trim:$('trim').checked,duration:num('duration'),seed:num('seed')};
 if(voiceB64)b.voice_audio=voiceB64;else if($('voice').value)b.voice=$('voice').value;
 for(const k in b)if(b[k]===undefined)delete b[k];return b}
function settings(b){return Object.entries(b).filter(([k])=>!['input','voice_audio','response_format'].includes(k)).map(([k,v])=>k+' '+v).join(' · ')}
function apply(b){$('text').value=b.input;$('lang').value=b.language||'en';$('speed').value=b.speed||1;
 for(const k of ['steps','flow_steps','cfg','cfg_prompt','temperature','sway','pause','duration','seed'])$(k).value=b[k]===undefined?'':b[k];
 $('trim').checked=b.trim!==false;if(b.voice){$('voice').value=b.voice;clearVoice()}upd()}
$('go').onclick=async()=>{const b=request();if(!b.input){$('text').focus();return}
 const id=++n,el=document.createElement('div');el.className='job';
 if($('jobs').firstChild&&$('jobs').firstChild.classList&&!$('jobs').firstChild.classList.contains('job'))$('jobs').innerHTML='';
 el.innerHTML='<div class="top"><span class="pill running">running</span><span class="meta">#'+id+(voiceName&&b.voice_audio?' · voice '+esc(voiceName):b.voice?' · voice '+esc(b.voice):'')+'</span><span class="sp"></span><span class="meta t"></span></div><div class="pr">'+esc(b.input)+'</div><div class="body"></div>';
 $('jobs').prepend(el);busy++;upd();const t0=performance.now();const tick=setInterval(()=>{el.querySelector('.t').textContent=((performance.now()-t0)/1000).toFixed(1)+' s'},100);
 const h={'Content-Type':'application/json'};if($('key').value)h['Authorization']='Bearer '+$('key').value;
 try{const r=await fetch('/v1/audio/speech',{method:'POST',headers:h,body:JSON.stringify(b)});const j=await r.json();
  if(!r.ok)throw new Error(j.error?j.error.message:r.status);
  const url=URL.createObjectURL(new Blob([Uint8Array.from(atob(j.audio),c=>c.charCodeAt(0))],{type:'audio/wav'}));
  el.querySelector('.pill').className='pill done';el.querySelector('.pill').textContent='done';
  b.seed=j.seed;el.querySelector('.body').innerHTML='<audio controls src="'+url+'"></audio><div class="meta">'+j.duration.toFixed(2)+' s of speech in '+j.elapsed.toFixed(2)+' s ('+(j.duration/j.elapsed).toFixed(1)+'x real time) · '+esc(settings(b))+'</div>'+
   '<div class="acts" style="margin-top:8px"><a download="vall-e-'+id+'.wav" href="'+url+'"><button>Download</button></a><button class="re">Reuse settings</button><button class="rm">Remove</button></div>';
  el.querySelector('audio').play().catch(()=>{});
 }catch(e){el.querySelector('.pill').className='pill failed';el.querySelector('.pill').textContent='failed';
  el.querySelector('.body').innerHTML='<div class="err">'+esc(String(e.message||e))+'</div><div class="acts" style="margin-top:8px"><button class="re">Reuse settings</button><button class="rm">Remove</button></div>'}
 finally{clearInterval(tick);busy--;upd()}
 el.querySelector('.re').onclick=()=>apply(b);el.querySelector('.rm').onclick=()=>el.remove();
};
$('text').onkeydown=e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey))$('go').click()};
upd();
</script></body></html>)HTML";
}

} // namespace engine::server
