import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F

# ====================================
# 1. Load model and tokenizer
# ====================================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# ====================================
# 2. Read input Excel file
# ====================================
input_path = "/.../anchordata/generalization_syn_prefixes_all.xlsx"
df = pd.read_excel(input_path)

# Storage for entropy values and top-5 predictions
entropy_values = []
top5_predictions = []

# ====================================
# 3. Iterate through each sentence
# ====================================
for item in df["Item"]:
    # Tokenize input
    inputs = tokenizer(item, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Forward pass through GPT-2
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Use the logits of the last token
    last_token_logits = logits[0, -1, :]
    probs = F.softmax(last_token_logits, dim=0)
    log_probs = torch.log(probs + 1e-10)

    # Compute entropy
    entropy = -(probs * log_probs).sum().item()
    entropy_values.append(entropy)

    # Retrieve Top-5 predicted tokens and their probabilities
    topk = torch.topk(probs, k=5)
    topk_tokens = topk.indices.tolist()
    topk_probs = topk.values.tolist()
    decoded_tokens = [tokenizer.decode([tok]).strip() for tok in topk_tokens]
    formatted_top5 = [f"{tok} ({prob:.4f})" for tok, prob in zip(decoded_tokens, topk_probs)]
    top5_predictions.append(formatted_top5)

# ====================================
# 4. Add results to DataFrame
# ====================================
df["Entropy"] = entropy_values
df["Top5_Predictions"] = top5_predictions

# ====================================
# 5. Select top 50 items with highest entropy
# ====================================
top50 = df.sort_values("Entropy", ascending=False).head(50)

# ====================================
# 6. Save results to Excel
# ====================================
output_path = ".../anchordata/generalization_syn_prefixes_top.xlsx"
top50.to_excel(output_path, index=False)

print(f"✅ Processing complete. Results saved to: {output_path}")