import json, argparse, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.modeling_t3 import T3Model
from utils import load_ckpt, denoise_k_step_soft_embed_v2

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default="cfg5-8-32-2/state_0")
parser.add_argument("--gen_length", type=int, default=128)
parser.add_argument("--steps", type=int, default=128)
parser.add_argument("--block_size", type=int, default=8)
parser.add_argument("--think_device1", type=str, default="cuda:0")
parser.add_argument("--think_device2", type=str, default="cuda:0")
parser.add_argument("--talk_device", type=str, default="cuda:0")
args = parser.parse_args()

THINK_DEVICE1 = args.think_device1
THINK_DEVICE2 = args.think_device2
TALK_DEVICE = args.talk_device
GEN_LEN = args.gen_length
MASK_TOKEN_ID = 126336

model_config = f"{args.ckpt_path}/config.json"
with open(model_config) as f:
    model_config = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(model_config["pretrained_model_name_or_path"], trust_remote_code=True)
model = T3Model(model_config, think_dev1=THINK_DEVICE1, think_dev2=THINK_DEVICE2, talk_dev=TALK_DEVICE)
model.eval()
load_ckpt(args.ckpt_path, model, None, None, map_location="cpu")

messages = [{
    "role": "system",
    "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    }]
convroles = ["user", "assistant"]
roles = {"human", "user", "gpt", "assistant"}

user_input = input("Enter your question: ")
messages.append({"role": "user", "content": user_input})

user_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
input_ids = tokenizer(user_input, return_tensors="pt", add_special_tokens=False).input_ids[0]
input_ids = input_ids.to(THINK_DEVICE1).unsqueeze(0)
seq_len = input_ids.shape[1]
max_len = seq_len + GEN_LEN

start_time = time.time()
x = torch.full((input_ids.shape[0], max_len), MASK_TOKEN_ID, dtype=torch.long).to(THINK_DEVICE1)
attention_mask = torch.ones_like(x, dtype=torch.bool, device=THINK_DEVICE1)
attention_bias = torch.zeros((x.shape[-1], x.shape[-1]), dtype=torch.bool, device=THINK_DEVICE1)
attention_bias[:seq_len, :seq_len] = True
for block_idx in range(GEN_LEN // args.block_size):
    attention_bias[seq_len + block_idx * args.block_size:seq_len + (block_idx + 1) * args.block_size, :seq_len + (block_idx + 1) * args.block_size] = True

x[:, :seq_len] = input_ids.clone()
past_key_values = None
x0 = x[:, :seq_len + args.block_size]
attn_mask = attention_mask[:, :x0.shape[-1]]
attn_bias = attention_bias[:x0.shape[-1], :x0.shape[-1]]
attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)
for block_idx in range(GEN_LEN // args.block_size):
    with torch.no_grad():
        think_outputs = model(
            input_ids=x0,
            attention_mask=attn_mask,
            attention_bias=attn_bias,
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

    think_rps = think_outputs.hidden_states  # [B, S+C, H] on THINK_DEVICE
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
    
    if THINK_DEVICE2 != TALK_DEVICE:
        talk_input_ids = talk_input_ids.to(TALK_DEVICE)
        talk_rps = talk_rps.to(TALK_DEVICE)
    
    talk_attn_mask = torch.ones_like(talk_input_ids, dtype=torch.long, device=TALK_DEVICE)
    talk_attn_bias = torch.zeros((1, 1, args.block_size, args.block_size), device=TALK_DEVICE, dtype=torch.float32)
    loss_mask = torch.ones_like(talk_attn_mask, dtype=torch.float32, device=TALK_DEVICE)

    talk_input_embeds = F.embedding(talk_input_ids, model.talk_embed_weight)  # initial emb (step 0)
    for idx in range(model.length):
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
        
        # denoise step updates input_ids + loss_mask (both on TALK_DEVICE)
        if model_config["soft_inputs"]["enabled"]:
            talk_input_ids, talk_input_embeds, loss_mask = denoise_k_step_soft_embed_v2(
                    input_ids=talk_input_ids,
                    target=None,
                    loss_mask=loss_mask,
                    logits=logits,
                    emb_weight=model.talk_embed_weight,
                    soft_topk=model_config["soft_inputs"]["top_k"],
                    soft_temp=model_config["soft_inputs"]["temperature"],
                    mode="ar_force",
                    sample_tokens=False,
                    # temperature=0.1,
                    # top_p=0.9
                )
        else:
            raise NotImplementedError("Currently haven't implement hard embed")
        
    if TALK_DEVICE != THINK_DEVICE1:
        talk_input_ids = talk_input_ids.to(THINK_DEVICE1)
    
    x[:, s:e] = talk_input_ids
    x0 = x[:, s:e + args.block_size]
    attn_mask = attention_mask[:, :e + args.block_size]
    attn_bias = attention_bias[:e + args.block_size, :e + args.block_size]
    attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)
print(f"Decode time: {time.time()-start_time:.3f}")
print("AI Reply:")
print(tokenizer.decode(x[:, seq_len:].detach().view(-1).tolist()))
