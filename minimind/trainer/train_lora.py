import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0 or step == iters:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}{moe_suffix}.pth'
            # LoRA는 LoRA 가중치만 저장
            save_lora(model, lora_save_path)
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()

        del input_ids, labels, res, loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    parser.add_argument("--save_dir", type=str, default="../out", help="모델 저장 디렉토리")
    parser.add_argument("--lora_name", type=str, default="lora_medical", help="LoRA 가중치 이름(예: lora_identity/lora_medical)")
    parser.add_argument("--epochs", type=int, default=10, help="학습 에포크 수")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="초기 학습률")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="학습 장치")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="혼합 정밀도 유형")
    parser.add_argument("--num_workers", type=int, default=8, help="데이터 로딩 작업자 수")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="기울기 누적 단계 수")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="기울기 클리핑 임계값")
    parser.add_argument("--log_interval", type=int, default=10, help="로그 출력 간격")
    parser.add_argument("--save_interval", type=int, default=1000, help="모델 저장 간격")
    parser.add_argument('--hidden_size', default=768, type=int, help="은닉층 차원")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="은닉층 수")
    parser.add_argument('--max_seq_len', default=340, type=int, help="학습 시 최대 절단 길이(중국어 1토큰≈1.5~1.7자)")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="MoE 구조 사용 여부(0=아니요, 1=예)")
    parser.add_argument("--data_path", type=str, default="../dataset/lora_medical.jsonl", help="LoRA 학습 데이터 경로")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="학습을 시작할 가중치(기본값: full_sft)")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="체크포인트 자동 감지 및 학습 재개 여부(0=아니요, 1=예)")
    parser.add_argument("--use_wandb", action="store_true", help="wandb 사용 여부")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb 프로젝트 이름")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="torch.compile 가속 사용 여부(0=아니요, 1=예)")
    args = parser.parse_args()

    # ========== 1. 환경 및 난수 시드 초기화 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 디렉토리와 모델 매개변수 설정 및 체크포인트 확인 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 혼합 정밀도 설정 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. wandb 설정 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 모델 정의, LoRA 적용 및 LoRA 이외 매개변수 고정 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    apply_lora(model)
    
    # 매개변수 집계
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 전체 매개변수 수: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 매개변수 수: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 매개변수 비율: {lora_params_count / total_params * 100:.2f}%")
    
    # LoRA 이외 매개변수를 고정하고 LoRA 매개변수 수집
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    
    # ========== 6. 데이터 및 옵티마이저 정의 ==========
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    # ========== 7. 체크포인트에서 상태 복원 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 8. 컴파일 및 분산 래핑 ==========
    if args.use_compile == 1:
        args.use_compile = 0
        Logger('[LoRA] 몽키 패치한 forward는 torch.compile과 호환되지 않아 use_compile을 자동으로 비활성화함')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 9. 학습 시작 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 처음 {start_step}개 단계를 건너뛰고 단계 {start_step + 1}시작')
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)
    
    # ========== 10. 분산 프로세스 정리 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()