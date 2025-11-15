# Load model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("the model was running on", device)
tokenizer = AutoTokenizer.from_pretrained(
    "pGenomeOcean/GenomeOcean-4B",
    trust_remote_code=True,
    padding_side="left",
)
model = AutoModelForCausalLM.from_pretrained(
    "pGenomeOcean/GenomeOcean-4B",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
).to("cuda") 

import os
import pandas as pd
# Load your data

cds_sequences = pd.read_csv('updated_combined_cds_sequences.csv', header=0)
cds_sequences['SeqID'] = cds_sequences['SeqID&protein_id'].str.split('|').str[1].str.split('_cds_').str[0]
import psutil
# Get system memory info
mem = psutil.virtual_memory()
# Print available RAM in GB
print(f"🔹 Total RAM: {mem.total / (1024**3):.2f} GB")
print(f"🔹 Used RAM: {mem.used / (1024**3):.2f} GB")
print(f"🔹 Available RAM: {mem.available / (1024**3):.2f} GB")

output_dir = 'single_DNA_embeddings'
os.makedirs(output_dir, exist_ok=True)

print("number of unique SeqID", cds_sequences.groupby('SeqID').ngroups)

# choose from the list
seqid_list = sorted(cds_sequences['SeqID'].unique()) # 17769 SeqID

# choose a range from the list
selected_seqids = seqid_list[0:3000]  # Python 是 0-based

for seqid, group in cds_sequences.groupby('SeqID'):
    if seqid not in selected_seqids:
        continue
    output_file = os.path.join(output_dir, f"{seqid}.pt")
    if os.path.exists(output_file):
        print(f"⚠️ {seqid}.pt already exists. Skipping.")
        continue
    batch_size = 4 
    single_embeddings = []
    # **Progress bar for this part**
    for i in range(0, group.shape[0], batch_size):
        batch = group.iloc[i : i + batch_size]
        # batch_seq_id = batch['SeqID'].tolist()
        batch_seq = batch['DNA_sequences'].tolist()

        # Tokenize sequences
        output = tokenizer.batch_encode_plus(
            batch_seq,
            max_length=10240,
            return_tensors='pt',
            padding='longest',
            truncation=True
        )
        input_ids = output['input_ids'].cuda()
        attention_mask = output['attention_mask'].cuda()

        # Model forward pass without gradients
        with torch.no_grad():
            # model_output = model(input_ids=input_ids, attention_mask=attention_mask)[0]
            model_output = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.detach().cpu()
        # Move to CPU ASAP
        model_output = model_output.cpu()
        attention_mask = attention_mask.unsqueeze(-1).detach().cpu()
        # Compute embeddings
        embedding = torch.sum(model_output * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        single_embeddings.append(embedding)      
        # Free GPU memory
        # del input_ids, attention_mask, model_output, output
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    part_filename = os.path.join(output_dir, seqid + '.pt')
    torch.save(torch.cat(single_embeddings, dim=0), part_filename)
