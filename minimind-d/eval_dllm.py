import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer
from model.model_minimind_dllm import MiniMindDLLMConfig, MiniMindForMaskedDiffusion
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    model = MiniMindForMaskedDiffusion(MiniMindDLLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    ))
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
    model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind DLLM推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='dllm', type=str, help="权重名称前缀（dllm, dllm_pt）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_new_tokens', default=256, type=int, help="最大生成长度")
    parser.add_argument('--steps', default=32, type=int, help="去噪总步数")
    parser.add_argument('--block_size', default=32, type=int, help="每个block的token数")
    parser.add_argument('--temperature', default=0.4, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_k', default=50, type=int, help="top-k采样（保留概率最高的k个token）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()

    prompts = [
        '请介绍一下你自己',
        '推荐一部好看的科幻电影',
        '如何快速学习一门新的编程语言'
    ]

    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))

    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        generated = model.generate(
            inputs=inputs["input_ids"],
            max_new_tokens=args.max_new_tokens,
            steps=args.steps,
            block_size=args.block_size,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_ids = generated[0, inputs['input_ids'].shape[1]:].tolist()
        if tokenizer.eos_token_id in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(tokenizer.eos_token_id)]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        print(f'🧠: {response}\n')

if __name__ == "__main__":
    main()
