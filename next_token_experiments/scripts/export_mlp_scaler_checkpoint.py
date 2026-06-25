import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ska_distill.warm_init import LogSingularValueResidualMLP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-14m")
    parser.add_argument("--layer_idx", type=int, default=0)
    parser.add_argument("--ranks", default="16,32,64")
    parser.add_argument("--batch_texts", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_train_pairs", type=int, default=16384)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--lambda_value", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--residual_scale", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", default="runs/mlp_scaler_pythia14m.pt")
    return parser.parse_args()


def load_texts(split, n_texts):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    texts = [ex["text"] for ex in ds if ex["text"].strip()]
    return texts[:n_texts]


def get_hidden_states_and_wk(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    train_texts = load_texts("train", args.batch_texts)
    val_texts = load_texts("validation", max(1, args.batch_texts // 2))

    def encode(texts):
        return tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        ).to(device)

    with torch.no_grad():
        train_out = model(**encode(train_texts), output_hidden_states=True)
        val_out = model(**encode(val_texts), output_hidden_states=True)

    x_train = train_out.hidden_states[args.layer_idx + 1].detach().float()
    x_val = val_out.hidden_states[args.layer_idx + 1].detach().float()

    layer = model.gpt_neox.layers[args.layer_idx]
    w_qkv = layer.attention.query_key_value.weight.detach().float()
    d_model = model.config.hidden_size
    w_k = w_qkv[d_model:2 * d_model, :]

    return x_train, x_val, w_k


def transition_pairs(x, w_comp):
    z = x @ w_comp.T
    z_t = z[:, :-1, :].reshape(-1, z.shape[-1])
    z_next = z[:, 1:, :].reshape(-1, z.shape[-1])
    return z_t, z_next


def sample_pairs(z_t, z_next, max_pairs, seed):
    n_pairs = z_t.shape[0]
    sample_size = min(max_pairs, n_pairs)
    generator = torch.Generator(device=z_t.device)
    generator.manual_seed(seed)
    idx = torch.randperm(n_pairs, generator=generator, device=z_t.device)[:sample_size]
    return z_t[idx], z_next[idx]


def differentiable_koopman_loss(x_train, x_val, w_comp, args, seed):
    train_z, train_next = transition_pairs(x_train, w_comp)
    val_z, val_next = transition_pairs(x_val, w_comp)
    train_z, train_next = sample_pairs(
        train_z,
        train_next,
        max_pairs=args.max_train_pairs,
        seed=seed,
    )

    n = train_z.shape[0]
    rank = train_z.shape[1]
    eye = torch.eye(rank, dtype=train_z.dtype, device=train_z.device)

    g = train_z.T @ train_z / n
    m = train_next.T @ train_z / n
    g_reg = g + args.ridge * eye

    a_t = torch.linalg.solve(g_reg.T, m.T)
    a = a_t.T

    pred_next = val_z @ a.T
    mse = torch.mean((pred_next - val_next) ** 2)
    rel_mse = mse / (torch.mean(val_next ** 2) + 1e-12)

    eigvals = torch.linalg.eigvalsh(g_reg)
    condition = eigvals[-1] / torch.clamp(eigvals[0], min=1e-12)
    loss = rel_mse + args.lambda_value * torch.log(condition)

    return loss, rel_mse, condition


def train_rank_scaler(x_train, x_val, w_k, rank, args, device):
    torch.manual_seed(args.seed + rank)

    _, singular_values, vt = torch.linalg.svd(w_k.to(device), full_matrices=False)
    singular_values = singular_values[:rank]
    vt = vt[:rank, :]

    scaler = LogSingularValueResidualMLP(
        hidden_dim=args.hidden_dim,
        residual_scale=args.residual_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(scaler.parameters(), lr=args.lr, weight_decay=1e-4)

    for step in range(args.steps):
        g = scaler(singular_values)
        w_comp = torch.diag(g) @ vt
        loss, rel_mse, condition = differentiable_koopman_loss(
            x_train,
            x_val,
            w_comp,
            args=args,
            seed=args.seed,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scaler.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0 or step == args.steps - 1:
            with torch.no_grad():
                base_g = torch.sqrt(singular_values + 1e-8)
                rel_g_change = torch.norm(g - base_g) / (torch.norm(base_g) + 1e-12)
            print(
                f"rank={rank} step={step}/{args.steps} "
                f"loss={loss.item():.6f} rel_mse={rel_mse.item():.6f} "
                f"cond={condition.item():.2f} rel_g_change={rel_g_change.item():.6f}",
                flush=True,
            )

    return {k: v.detach().cpu() for k, v in scaler.state_dict().items()}


def main():
    args = parse_args()
    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training MLP scaler checkpoint on {device}", flush=True)
    print(f"model={args.model}, layer_idx={args.layer_idx}, ranks={ranks}", flush=True)
    x_train, x_val, w_k = get_hidden_states_and_wk(args, device)

    checkpoint = {
        "metadata": {
            "model": args.model,
            "layer_idx": args.layer_idx,
            "ranks": ranks,
            "batch_texts": args.batch_texts,
            "max_length": args.max_length,
            "max_train_pairs": args.max_train_pairs,
            "ridge": args.ridge,
            "lambda_value": args.lambda_value,
            "steps": args.steps,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "residual_scale": args.residual_scale,
            "seed": args.seed,
        }
    }

    for rank in ranks:
        checkpoint[rank] = train_rank_scaler(x_train, x_val, w_k, rank, args, device)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Saved MLP scaler checkpoint to {output}")


if __name__ == "__main__":
    main()
