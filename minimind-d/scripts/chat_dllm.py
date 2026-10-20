import sys, os, argparse, math, shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.model_minimind_dllm import MiniMindDLLMConfig, MiniMindForMaskedDiffusion, add_gumbel_noise

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.text import Text

model, tokenizer, device = None, None, 'cpu'


def load_model(args):
    global model, tokenizer, device
    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    config = MiniMindDLLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    model = MiniMindForMaskedDiffusion(config)
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = f'{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
    if os.path.exists(ckp):
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)
        print(f"[加载权重] {ckp}")
    model = model.half().eval().to(device)


prev_ids = None

def render_tokens(x, prompt_len, mask_id, eos_id):
    global prev_ids
    ids = x[prompt_len:].tolist()
    t = Text()
    for i, tid in enumerate(ids):
        if tid == mask_id:
            t.append("[M]", style="dim red")
        elif tid == eos_id:
            t.append("[E]", style="dim green")
        elif prev_ids is not None and i < len(prev_ids) and prev_ids[i] == mask_id:
            t.append(tokenizer.decode([tid]), style="bold bright_green")
        else:
            t.append(tokenizer.decode([tid]), style="white")
    prev_ids = ids[:]
    return t


def update_tui(live, layout, progress, task_id, x, prompt_len, mask_id, eos_id, step, total_steps):
    masks_left = int((x[prompt_len:] == mask_id).sum().item())
    total_gen = len(x[prompt_len:])
    revealed = total_gen - masks_left
    pct = f"{int(100 * step / max(total_steps, 1))}%"
    progress.update(task_id, completed=step, total=total_steps, masks=masks_left, pct=pct)
    layout["text"].update(Panel(
        render_tokens(x, prompt_len, mask_id, eos_id),
        title=f"[bold cyan]Denoising[/bold cyan]  [dim]{revealed}/{total_gen} revealed[/dim]",
        subtitle=f"[dim]Step {step}/{total_steps}[/dim]",
        border_style="bright_blue", padding=(1, 1),
    ))
    layout["progress"].update(Panel(progress, border_style="dim"))
    live.refresh()


def stream_generate_tui(input_ids, prompt_len, args):
    global prev_ids
    prev_ids = None
    mask_id = model.config.mask_token_id
    eos_id = tokenizer.eos_token_id
    block_size, steps, max_tokens = args.block_size, args.steps, args.max_new_tokens

    num_blocks = math.ceil(max_tokens / block_size)
    total_steps = num_blocks * steps
    T = prompt_len + max_tokens
    x = torch.full((1, T), eos_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    x[:, prompt_len:] = mask_id
    unmasked_index = (x != mask_id)

    try:
        tw = shutil.get_terminal_size().columns
    except Exception:
        tw = 120
    console = Console(width=tw)

    progress = Progress(
        SpinnerColumn(), TextColumn("[bold blue]Diffusion"), BarColumn(), MofNCompleteColumn(),
        TextColumn("•"), TextColumn("[cyan]Masks: {task.fields[masks]}"),
        TextColumn("•"), TextColumn("[magenta]{task.fields[pct]:>4s}"), TimeRemainingColumn(), expand=True,
    )
    layout = Layout()
    layout.split_column(Layout(name="text", ratio=1), Layout(name="progress", size=3))
    task_id = progress.add_task("gen", total=total_steps, masks=max_tokens, pct="0%")

    with Live(layout, console=console, auto_refresh=False, screen=True) as live:
        global_step = 0
        for b in range(num_blocks):
            block_end = min(prompt_len + (b + 1) * block_size, T)

            for step in range(steps):
                mask_index = (x == mask_id)
                mask_count = mask_index[:, :block_end].sum(-1).min().item()
                if mask_count == 0: break
                n_unmask = max(1, round(mask_count / (steps - step)))

                with torch.no_grad():
                    logits = model(input_ids=x).logits
                if args.top_k > 0:
                    logits[logits < torch.topk(logits, args.top_k, dim=-1)[0][..., -1:]] = -float('inf')
                x0 = torch.argmax(add_gumbel_noise(logits, args.temperature), dim=-1)
                p = F.softmax(logits.float(), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                x0_p[:, block_end:] = -float('inf')
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-float('inf'), device=device))
                _, idx = torch.topk(confidence[0], k=min(n_unmask, int(mask_count)))
                x[0, idx] = x0[0, idx]

                global_step += 1
                update_tui(live, layout, progress, task_id, x[0], prompt_len, mask_id, eos_id, global_step, total_steps)

            if eos_id and (x[:, prompt_len:] == eos_id).any(): break

    gen_ids = x[0, prompt_len:].tolist()
    if eos_id in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(eos_id)]
    final_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"\n🤖️: {final_text}\n")
    return final_text


def chat_loop(args):
    is_pt = 'pt' in args.weight
    prompts = ['你有什么特长？', '为什么天空是蓝色的', '请用Python写一个计算斐波那契数列的函数', '解释一下"光合作用"的基本过程']

    print(f"\n{'='*50}")
    print(f"  MiniMind DLLM TUI Chat ({'pretrain' if is_pt else 'sft'} mode)")
    print(f"  steps={args.steps} block={args.block_size} max_tokens={args.max_new_tokens}")
    print(f"  temperature={args.temperature} top_k={args.top_k}")
    print(f"{'='*50}\n")

    mode = input('[0] 自动测试\n[1] 手动输入\n> ').strip()
    prompt_iter = prompts if mode == '0' else iter(lambda: input('\n💬 > '), '')

    for prompt in prompt_iter:
        if mode == '0':
            print(f'\n💬: {prompt}')

        if is_pt:
            input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        else:
            tpl = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(tpl, return_tensors="pt", truncation=True).input_ids.to(device)

        stream_generate_tui(input_ids, input_ids.shape[1], args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DLLM TUI Chat")
    parser.add_argument('--load_from', default='../model', type=str, help="tokenizer路径")
    parser.add_argument('--save_dir', default='../out', type=str, help="权重目录")
    parser.add_argument('--weight', default='dllm', type=str, help="权重前缀名")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_new_tokens', default=128, type=int, help="最大生成长度")
    parser.add_argument('--steps', default=32, type=int, help="去噪总步数")
    parser.add_argument('--block_size', default=32, type=int, help="每个block的token数")
    parser.add_argument('--temperature', default=0.5, type=float, help="采样温度")
    parser.add_argument('--top_k', default=50, type=int, help="top-k采样")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="推理设备")
    args = parser.parse_args()

    try:
        load_model(args)
        chat_loop(args)
    except KeyboardInterrupt:
        print("\n退出")
