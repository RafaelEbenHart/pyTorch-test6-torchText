import os
import torch
import torch.nn.functional as f

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# try load a saved full model first, fallback to checkpoint with state_dict
MODEL_FULL = "model_full.pth"
CHECKPOINT = "model_checkpoint.pth"

# reconstruct vocab & model depending on saved files
if os.path.exists(MODEL_FULL):
    print(f"Loading full model from {MODEL_FULL} ...")
    model = torch.load(MODEL_FULL, map_location=device)
    # try to load vocab saved alongside (same folder)
    vocab_path = "vocab.pth"
    if os.path.exists(vocab_path):
        vocab = torch.load(vocab_path, map_location=device)
        string_to_int = vocab['string_to_int']
        int_to_string = vocab['int_to_string']
    else:
        # if full model file included vocab inside, attempt to read attributes
        try:
            string_to_int = model.string_to_int
            int_to_string = model.int_to_string
        except Exception:
            raise RuntimeError("vocab mappings not found. Please save 'vocab.pth' or include mappings in checkpoint.")
else:
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Neither {MODEL_FULL} nor {CHECKPOINT} found.")
    print(f"Loading checkpoint from {CHECKPOINT} ...")
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    string_to_int = checkpoint['string_to_int']
    int_to_string = checkpoint['int_to_string']
    # prefer reconstructing model from model_args + state_dict
    if 'model' in checkpoint:
        model = checkpoint['model']
    else:
        # import GPTArch only when needed to reconstruct from state_dict
        from transformer import GPTArch
        model_args = checkpoint.get('model_args', {}) or {}
        try:
            # try directly
            model = GPTArch(**model_args)
        except TypeError:
            # filter model_args to only parameters accepted by GPTArch.__init__
            import inspect
            sig = inspect.signature(GPTArch.__init__)
            valid_params = set(list(sig.parameters.keys())[1:])  # skip 'self'
            filtered_args = {k: v for k, v in model_args.items() if k in valid_params}
            if filtered_args:
                model = GPTArch(**filtered_args)
            elif 'vocab_size' in checkpoint:
                # fallback to older key
                model = GPTArch(checkpoint['vocab_size'])
            else:
                # last resort: try no-arg constructor
                model = GPTArch()
        # load state dict
        model.load_state_dict(checkpoint['model_state_dict'])

# move model to device and set eval
model.to(device)
model.eval()

# local encode/decode using saved vocab mappings
encode = lambda s: [string_to_int.get(c, 0) for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

def generate_response(model, prompt, max_tokens=200, temperature=0.8, stop_token="\nUser1:"):
    model.eval()
    context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)  # (1, L)
    generated_text = ""
    with torch.inference_mode():
        for _ in range(max_tokens):
            out = model(context)
            logits = out[0] if isinstance(out, tuple) else out
            logits = logits[:, -1, :] / max(1e-8, temperature)
            probs = f.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1,1)
            token_id = int(next_token.item())
            # append decoded char(s)
            generated_text += int_to_string[token_id]
            # stop if model starts producing next User1 prompt
            if stop_token in generated_text:
                generated_text = generated_text.split(stop_token)[0]
                break
            # extend context for next step
            context = torch.cat([context, next_token.to(device)], dim=1)
    return generated_text.strip()

def main():
    print("\nChat with AI as User2 (type 'quit' to exit)")
    print("-------------------------------------------")
    while True:
        user_input = input("\nUser1: ")
        if user_input.strip().lower() in ("quit", "exit"):
            break
        # format prompt consistent with your training data
        ui_label = "EL-EN"
        prompt_label = "User2"   # label yang model kenal
        prompt = f"User1: {user_input}\n{prompt_label}:"
        response = generate_response(model, prompt, max_tokens=200, temperature=0.7, stop_token="\nUser1:")
        print(f"{ui_label}:", response)

if __name__ == "__main__":
    main()