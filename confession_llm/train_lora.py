"""Step 4: LoRA / QLoRA fine-tune of Qwen3-4B-Instruct-2507 on the confession dataset.

    # 24GB+ GPU (A10G/L4/3090/4090/A100): bf16 LoRA
    python -m confession_llm.train_lora
    # 16GB GPU (free Colab T4): 4-bit QLoRA
    python -m confession_llm.train_lora --qlora --batch-size 1 --grad-accum 16

~400 stories x 3 epochs takes roughly 15-30 min on an L4/A10G.
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data", default="data/sft")
    ap.add_argument("--out", default="outputs/qwen3-4b-nri-confessions-lora")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--qlora", action="store_true", help="load the base model in 4-bit")
    ap.add_argument("--push-to-hub", default=None, help="e.g. yourname/nri-confessions-lora (pushed private)")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    files = {"train": f"{args.data}/train.jsonl"}
    try:
        open(f"{args.data}/val.jsonl").close()
        files["validation"] = f"{args.data}/val.jsonl"
    except FileNotFoundError:
        pass
    ds = load_dataset("json", data_files=files)
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in ("prompt", "completion")])
    has_val = "validation" in ds and len(ds["validation"]) > 0

    quant = None
    if args.qlora:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, quantization_config=quant, device_map="auto",
        attn_implementation="sdpa",
    )

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        max_length=args.max_length,
        completion_only_loss=True,  # learn the stories, not the system/user prompt
        bf16=bf16,
        fp16=not bf16 and torch.cuda.is_available(),
        logging_steps=5,
        eval_strategy="epoch" if has_val else "no",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"] if has_val else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved LoRA adapter to {args.out}")

    if args.push_to_hub:
        trainer.model.push_to_hub(args.push_to_hub, private=True)
        tokenizer.push_to_hub(args.push_to_hub, private=True)
        print(f"Pushed (private) to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
