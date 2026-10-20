import sys, os, json, time, math, torch, torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, render_template_string, request, Response
from transformers import AutoTokenizer
from model.model_minimind_dllm import MiniMindDLLMConfig, MiniMindForMaskedDiffusion

app = Flask(__name__)
model, tokenizer, device = None, None, 'cpu'

HTML = """
<!DOCTYPE html><html><head><meta charset="UTF-8"><title>DLLM</title>
<style>
:root{
  --bg:#24262b;--surface:#34373d;--panel:#2b2e34;--border:#4a4f57;
  --fg:#f3f4f6;--dim:#b0b4bb;--accent:#7ea6e8;--reveal:#63d297;
  --mask:#f07c7c;--mask-dim:rgba(240,124,124,0.12);
}
*{box-sizing:border-box;margin:0;padding:0}
html{font-size:15px}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column;overflow:hidden;justify-content:center}

.wrap{max-width:860px;width:100%;margin:0 auto;display:flex;flex-direction:column;height:calc(100vh - 48px);max-height:720px;box-shadow:0 8px 24px rgba(0,0,0,0.18)}
.top-bar{padding:16px 20px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border);background:var(--surface);border-radius:10px 10px 0 0}
.top-bar .title{font-size:14px;font-weight:600;color:var(--fg);letter-spacing:0}
.top-bar .status{font-size:12px;color:var(--dim);letter-spacing:0}
.top-bar .status .dot{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:6px;background:var(--dim);vertical-align:middle}
.top-bar .status .dot.live{background:var(--reveal)}

.main{flex:1;display:flex;flex-direction:column;overflow:hidden;min-height:0}

.meta-bar{padding:10px 20px;display:flex;gap:16px;border-bottom:1px solid var(--border);font-size:12px;color:var(--dim);letter-spacing:0;background:var(--surface)}
.meta-bar .tag{padding:0;border:none;border-radius:0;background:transparent}

.output{flex:1;overflow-y:auto;padding:20px;font-size:14px;line-height:1.8;white-space:pre-wrap;word-break:break-all;background:var(--surface);min-height:0;scrollbar-width:none}
.output::-webkit-scrollbar{display:none}
.output .token{transition:color .15s ease}
.output .token.mask{color:var(--mask);opacity:.9;font-weight:600}
.output .token.hidden{color:var(--dim);opacity:.15}
.output .token.filled{color:var(--fg)}
.output .token.new{color:var(--reveal);font-weight:600}
.output.token-mode{white-space:normal;display:flex;flex-wrap:wrap;gap:3px;align-content:flex-start}
.output.token-mode .token{padding:2px 6px;border:1px solid var(--border);border-radius:4px;font-size:12px;line-height:1.4;background:var(--panel)}
.output.token-mode .token.mask{background:rgba(240,124,124,0.18);border-color:rgba(240,124,124,0.48);box-shadow:inset 0 0 0 1px rgba(240,124,124,0.12)}
.output.token-mode .token.hidden{border-color:transparent;background:transparent}
.output.token-mode .token.new{border-color:var(--reveal);background:rgba(99,210,151,0.10)}

.progress-wrap{padding:8px 20px;border-top:1px solid var(--border);background:var(--surface)}
.progress-track{height:2px;background:rgba(255,255,255,0.08);border-radius:1px;overflow:hidden}
.progress-fill{height:100%;width:0;background:var(--accent);transition:width .2s ease;border-radius:1px}

.bottom{padding:16px 20px;border-top:1px solid var(--border);background:var(--surface);border-radius:0 0 10px 10px}
.input-row{display:flex;gap:10px;margin-bottom:10px}
.input-row input[type=text]{flex:1;padding:10px 14px;background:var(--panel);border:1px solid var(--border);color:var(--fg);font-family:inherit;font-size:13px;outline:none;border-radius:6px;transition:border-color .2s}
.input-row input[type=text]:focus{border-color:var(--accent)}
.input-row input[type=text]::placeholder{color:var(--dim)}
.input-row button{padding:10px 16px;background:var(--panel);border:1px solid var(--border);color:var(--fg);font-family:inherit;font-size:12px;font-weight:500;letter-spacing:0;cursor:pointer;border-radius:6px;transition:all .15s}
.input-row button:hover{border-color:var(--accent);color:var(--accent)}
.input-row button:disabled{opacity:.3;cursor:default}
.params-row{display:flex;gap:16px;font-size:11px;color:var(--dim)}
.params-row label{display:flex;align-items:center;gap:6px}
.params-row input{width:64px;padding:5px 8px;background:var(--panel);border:1px solid var(--border);color:var(--fg);font-family:inherit;font-size:12px;border-radius:6px;outline:none;text-align:center}
.params-row input:focus{border-color:var(--accent)}
</style></head>
<body>
<div class="wrap">
<div class="top-bar">
  <div class="title">MiniMind dLLM</div>
  <div class="status" id="statusBar"><span class="dot"></span>IDLE</div>
</div>
<div class="main">
  <div class="meta-bar" id="metaBar">
    <span class="tag" id="stepTag">step --</span>
    <span class="tag" id="maskTag">masks --</span>
    <span class="tag" id="blockTag">block --</span>
    <span class="tag" id="viewToggle" style="cursor:pointer;user-select:none;margin-left:auto">text</span>
  </div>
  <div class="output" id="tokens"></div>
</div>
<div class="progress-wrap"><div class="progress-track"><div class="progress-fill" id="progressFill"></div></div></div>
<div class="bottom">
  <div class="input-row">
    <input type="text" id="prompt" placeholder="prompt >" value="Python实现快速排序算法">
    <button id="genBtn" onclick="generate()">生成</button>
  </div>
  <div class="params-row">
    <label>block <input type="number" id="blockSize" value="32"></label>
    <label>steps <input type="number" id="steps" value="32"></label>
    <label>max <input type="number" id="maxTokens" value="128"></label>
  </div>
</div>
</div>
<script>
let prev=[],running=false,viewMode='text',lastData=null;
document.getElementById('viewToggle').addEventListener('click',()=>{
  viewMode=viewMode==='text'?'token':'text';
  document.getElementById('viewToggle').textContent=viewMode==='text'?'文本':'分词';
  const o=document.getElementById('tokens');
  if(viewMode==='token')o.classList.add('token-mode');else o.classList.remove('token-mode');
  if(lastData)render(lastData,0);
});
async function generate(){
  if(running)return;
  running=true;
  const btn=document.getElementById('genBtn');
  btn.disabled=true;btn.textContent='...';
  const dot=document.querySelector('.dot');
  dot.classList.add('live');
  document.getElementById('statusBar').innerHTML='<span class="dot live"></span>生成中';
  const p=document.getElementById('prompt').value,b=+document.getElementById('blockSize').value;
  const s=+document.getElementById('steps').value,m=+document.getElementById('maxTokens').value;
  document.getElementById('tokens').innerHTML='';
  document.getElementById('progressFill').style.width='0%';
  prev=[];
  const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({prompt:p,block_size:b,steps:s,max_tokens:m})});
  const reader=r.body.getReader(),dec=new TextDecoder();
  let buf='';
  while(1){
    const{done,value}=await reader.read();
    if(done)break;
    buf+=dec.decode(value,{stream:true});
    const parts=buf.split('\\n\\n');
    buf=parts.pop();
    parts.forEach(p=>{const l=p.trim();if(l.startsWith('data:'))try{render(JSON.parse(l.slice(5)),s)}catch(e){}});
  }
  running=false;btn.disabled=false;btn.textContent='生成';
  document.getElementById('statusBar').innerHTML='<span class="dot"></span>完成';
}
function render(d,totalS){
  lastData=d;
  const masks=d.tokens.filter(t=>t.is_mask&&!t.hidden).length;
  document.getElementById('stepTag').textContent='step '+d.step+'/'+d.total_steps;
  document.getElementById('maskTag').textContent='masks '+masks;
  document.getElementById('blockTag').textContent='block '+d.block;
  const pct=Math.min(100,Math.round(d.step/d.total_steps*100));
  document.getElementById('progressFill').style.width=pct+'%';
  if(d.done)document.getElementById('progressFill').style.width='100%';
  const c=document.getElementById('tokens');
  c.innerHTML='';
  if(viewMode==='token'){
    d.tokens.forEach((t,i)=>{
      const s=document.createElement('span');
      s.className='token '+(t.hidden?'hidden':t.is_mask?'mask':'filled'+(!d.done&&prev[i]!==t.text?' new':''));
      s.textContent=t.hidden?'·':t.is_mask?'[M]':t.text||' ';
      c.appendChild(s);
    });
  }else{
    let h='';
    d.tokens.forEach((t,i)=>{
      const cls=t.hidden?'hidden':t.is_mask?'mask':'filled'+(!d.done&&prev[i]!==t.text?' new':'');
      const txt=t.hidden?'':t.is_mask?'[M]':(t.text||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      h+='<span class="token '+cls+'">'+txt+'</span>';
    });
    c.innerHTML=h;
  }
  prev=d.tokens.map(t=>t.text);
}
document.getElementById('prompt').addEventListener('keydown',e=>{if(e.key==='Enter')generate()});
</script></body></html>
"""

def load_model():
    global model, tokenizer, device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('../model')
    model = MiniMindForMaskedDiffusion(MiniMindDLLMConfig(hidden_size=768, num_hidden_layers=8))
    ckp = '../out/dllm_768.pth'
    if os.path.exists(ckp): model.load_state_dict(torch.load(ckp, map_location=device), strict=False)
    model = model.half().eval().to(device)
    print(f"[Model loaded] device={device}")

def make_tokens(x, prompt_len, mask_id, eos_id):
    tokens, found_eos = [], False
    for t in x[0, prompt_len:]:
        tid = t.item()
        if tid == eos_id: found_eos = True
        if found_eos: tokens.append({'is_mask': True, 'hidden': True, 'text': ''})
        else: tokens.append({'is_mask': tid == mask_id, 'text': '' if tid == mask_id else tokenizer.decode([tid])})
    return tokens

@app.route('/')
def index(): return render_template_string(HTML)

@app.route('/generate', methods=['POST'])
def generate():
    d = request.json
    prompt, block_size, steps, max_tokens = d.get('prompt', '你好'), d.get('block_size', 128), d.get('steps', 128), d.get('max_tokens', 256)
    def stream():
        input_ids = tokenizer(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True), return_tensors="pt").input_ids.to(device)
        mask_id, eos_id, prompt_len = model.config.mask_token_id, tokenizer.eos_token_id, input_ids.shape[1]
        x = torch.cat([input_ids, torch.full((1, max_tokens), mask_id, dtype=torch.long, device=device)], dim=1)
        num_blocks = math.ceil(max_tokens / block_size)
        global_step, total_steps = 0, num_blocks * steps
        yield f"data:{json.dumps({'block': 0, 'step': 0, 'total_steps': total_steps, 'tokens': make_tokens(x, prompt_len, mask_id, eos_id)})}\n\n"
        for b in range(num_blocks):
            block_start = prompt_len + b * block_size
            block_end = min(prompt_len + (b + 1) * block_size, prompt_len + max_tokens)
            cur_block = block_end - block_start
            for step in range(steps):
                mask_pos = (x[:, block_start:block_end] == mask_id)
                if not mask_pos.any(): break
                with torch.no_grad(): logits = model(input_ids=x).logits[:, block_start:block_end, :]
                probs = F.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(1, -1)
                conf = torch.where(mask_pos, torch.gather(probs, -1, sampled.unsqueeze(-1)).squeeze(-1), -1.0)
                _, top_idx = conf.topk(min(cur_block if step == steps - 1 else max(1, cur_block // steps), mask_pos.sum().item()), dim=-1)
                x[0, block_start + top_idx[0]] = sampled[0, top_idx[0]]
                global_step += 1
                yield f"data:{json.dumps({'block': b+1, 'step': global_step, 'total_steps': total_steps, 'tokens': make_tokens(x, prompt_len, mask_id, eos_id)})}\n\n"
                time.sleep(0.05)
            if eos_id and (x[:, prompt_len:] == eos_id).any(): break
        yield f"data:{json.dumps({'block': num_blocks, 'step': total_steps, 'total_steps': total_steps, 'tokens': make_tokens(x, prompt_len, mask_id, eos_id), 'done': True})}\n\n"
    return Response(stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    load_model()
    print("🎲 DLLM Demo: http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
