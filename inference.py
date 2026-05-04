import json, argparse, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.modeling_t3 import T3Model
from utils import (
    denoise_k_step_hard,
    denoise_k_step_soft_embed_v2,
    get_denoise_decode_config,
    get_denoise_reveal_config,
    get_policy_label,
    load_ckpt,
)


def extract_code_block(text):
    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


def extract_function_body(text):
    if "def " not in text:
        return text.strip()

    body = text.split("def ", 1)[1]
    if ":\n" in body:
        body = body.split(":\n", 1)[1]

    if '"""\n' in body:
        body = body.split('"""\n', 1)[-1]

    elif "'''\n" in body:
        body = body.split("'''\n", 1)[-1]

    return body

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default="cfg6-p1-l30-r32-a64-lm-t6-mlp-7.5lr-2-denoise-mix-6-90-2/state_0")
parser.add_argument("--gen_length", type=int, default=768)
parser.add_argument("--steps", type=int, default=768)
parser.add_argument("--block_size", type=int, default=6)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--prompt_mode", choices=["chat", "raw"], default="chat")
parser.add_argument("--extract_code", action="store_true")
args = parser.parse_args()

DEVICE = args.device
GEN_LEN = args.gen_length
MASK_TOKEN_ID = 126336

model_config = f"{args.ckpt_path}/config.json"
with open(model_config) as f:
    model_config = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(model_config["pretrained_model_name_or_path"], trust_remote_code=True)
model = T3Model(model_config, device=DEVICE)
model.eval()
load_ckpt(args.ckpt_path, model, None, None, map_location="cpu")
denoise_cfg = model_config.get("denoise", {})
reveal_cfg = get_denoise_reveal_config(denoise_cfg)
decode_cfg = get_denoise_decode_config(denoise_cfg, default_k=reveal_cfg["k"])
reveal_mode = get_policy_label(reveal_cfg.get("policy"), "ar_force")
reveal_k = int(reveal_cfg["k"])

messages = []
convroles = ["user", "assistant"]
roles = {"human", "user", "gpt", "assistant"}

user_input = input("Enter your question: ")
messages.append({"role": "user", "content": user_input})

if args.prompt_mode == "raw":
    prompt_text = user_input
else:
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

input_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
input_ids = input_ids.to(DEVICE).unsqueeze(0)
seq_len = input_ids.shape[1]
max_len = seq_len + GEN_LEN

start_time = time.time()
x = torch.full((input_ids.shape[0], max_len), MASK_TOKEN_ID, dtype=torch.long, device=DEVICE)
position_ids = torch.arange(0, x.shape[1], device=DEVICE).unsqueeze(0)
attention_mask = torch.ones_like(x, dtype=torch.bool, device=DEVICE)
attention_bias = torch.zeros((x.shape[-1], x.shape[-1]), dtype=torch.bool, device=DEVICE)
attention_bias[:seq_len, :seq_len] = True
for block_idx in range(GEN_LEN // args.block_size):
    attention_bias[seq_len + block_idx * args.block_size:seq_len + (block_idx + 1) * args.block_size, :seq_len + (block_idx + 1) * args.block_size] = True

x[:, :seq_len] = input_ids.clone()
past_key_values = None
x0 = x[:, :seq_len + args.block_size]
p0 = position_ids[:, :seq_len + args.block_size]
attn_mask = attention_mask[:, :x0.shape[-1]]
attn_bias = attention_bias[:x0.shape[-1], :x0.shape[-1]]
attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)
for block_idx in range(GEN_LEN // args.block_size):
    with torch.no_grad():
        think_outputs = model(
            input_ids=x0,
            attention_mask=attn_mask,
            attention_bias=attn_bias,
            position_ids=p0,
            use_cache=True,
            past_key_values=past_key_values,
            output_hidden_states=True
        )
    past_key_values = think_outputs.past_key_values
    new_past_key_values = []
    for i in range(len(past_key_values)):
        new_past_key_values.append(())
        for j in range(len(past_key_values[i])):
            new_past_key_values[i] += (past_key_values[i][j][:, :, :-args.block_size],)
    past_key_values = new_past_key_values

    think_rps = think_outputs.hidden_states  # [B, S+C, H]
    B = think_rps.size(0)
    H = think_rps.size(-1)

    s = seq_len + args.block_size * block_idx
    e = seq_len + (args.block_size * (block_idx + 1))
    if x0.shape[-1] > 2 * args.block_size:
        talk_input_ids = x0[:, s:e]
        talk_rps = think_rps[:, s:e, :]
    else:
        talk_input_ids = x0[:, -args.block_size:]
        talk_rps = think_rps[:, -args.block_size:, :]

    talk_attn_mask = torch.ones_like(talk_input_ids, dtype=torch.long)
    talk_attn_bias = torch.zeros((1, 1, args.block_size, args.block_size), device=DEVICE, dtype=torch.float32)
    loss_mask = torch.ones_like(talk_attn_mask, dtype=torch.float32)

    talk_input_embeds = F.embedding(talk_input_ids, model.talk_embed_weight)  # initial emb (step 0)
    for idx in range(model.length):
        if not loss_mask.bool().any():
            break
        talk_outputs = model(
            input_ids=None,
            inputs_embeds=talk_input_embeds,
            inputs_repres=talk_rps,
            attention_mask=talk_attn_mask,
            attention_bias=talk_attn_bias,
            use_cache=False,
            output_hidden_states=True
        )
        logits = talk_outputs.logits.float()
        talk_rps = talk_outputs.hidden_states

        if model_config["soft_inputs"]["enabled"]:
            talk_input_ids, talk_input_embeds, loss_mask = denoise_k_step_soft_embed_v2(
                    input_ids=talk_input_ids,
                    target=None,
                    loss_mask=loss_mask,
                    logits=logits,
                    emb_weight=model.talk_embed_weight,
                    k_reveal=reveal_k,
                    soft_topk=model_config["soft_inputs"]["top_k"],
                    soft_temp=model_config["soft_inputs"]["temperature"],
                    mode=reveal_cfg.get("policy", "ar_force"),
                    decode_cfg=decode_cfg,
                )
        else:
            # Hard iterative reveal fallback: reveal only k positions each denoise step.
            talk_input_ids, loss_mask, _, _ = denoise_k_step_hard(
                input_ids=talk_input_ids,
                target=None,
                loss_mask=loss_mask,
                logits=logits,
                reveal_cfg=reveal_cfg,
                decode_cfg=decode_cfg,
            )
            talk_input_embeds = F.embedding(talk_input_ids, model.talk_embed_weight)

    x[:, s:e] = talk_input_ids
    x0 = x[:, s:e + args.block_size]
    p0 = position_ids[:, s:e + args.block_size]
    attn_mask = attention_mask[:, :e + args.block_size]
    attn_bias = attention_bias[:e + args.block_size, :e + args.block_size]
    attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)
print(f"Decode time: {time.time()-start_time:.3f}")
print("AI Reply:")
decoded = tokenizer.decode(x[:, seq_len:seq_len + GEN_LEN].detach().view(-1).tolist(), skip_special_tokens=True)
print(decoded)
