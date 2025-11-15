import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from sklearn.metrics import accuracy_score,precision_score,recall_score, confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef

# ==============================
# Dataset Definition (with Label)
# ==============================
class DNAPairDataset(Dataset):
    """
    Dataset for pairs (chromosome and plasmid embeddings) with labels,
    loaded from a CSV file with three columns: Chromosome, Plasmid, Label.
    Embeddings are loaded from .pt files.
    """
    def __init__(self, csv_path, embeddings_folder):
        self.data = pd.read_csv(csv_path)
        self.embeddings_folder = embeddings_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq_id_chr = row["Chromosome"]
        seq_id_plas = row["Plasmid"]
        label = row["Label"]  # Assumed to be 0 or 1

        emb_path_chr = os.path.join(self.embeddings_folder, f"{seq_id_chr}.pt")
        emb_path_plas = os.path.join(self.embeddings_folder, f"{seq_id_plas}.pt")

        embedding_chr = torch.load(emb_path_chr)   # [L_chr, 3072]
        embedding_plas = torch.load(emb_path_plas) # [L_plas, 3072]

        return embedding_chr, embedding_plas, label

def create_padding_mask(seq_lengths, max_length):
    """
    Creates a padding mask of shape [B, max_length] with True where
    positions are padding, and False for real tokens.
    """
    mask = torch.arange(max_length).expand(len(seq_lengths), max_length) >= seq_lengths.unsqueeze(1)
    return mask

# ==============================
# Collate Function (with labels)
# ==============================
def clip_collate_fn(batch):
    embeddings_chr, embeddings_plas, labels = [], [], []
    len_chr_list, len_plas_list = [], []

    # Determine maximum lengths within the batch
    max_len_chr = max(item[0].size(0) for item in batch)
    max_len_plas = max(item[1].size(0) for item in batch)

    for emb_chr, emb_plas, label in batch:
        len_chr_list.append(emb_chr.size(0))
        len_plas_list.append(emb_plas.size(0))
        labels.append(label)

        # Pad chromosome embeddings
        pad_chr = torch.zeros(max_len_chr - emb_chr.size(0), emb_chr.size(1))
        emb_chr_padded = torch.cat([emb_chr, pad_chr], dim=0)
        embeddings_chr.append(emb_chr_padded)

        # Pad plasmid embeddings
        pad_plas = torch.zeros(max_len_plas - emb_plas.size(0), emb_plas.size(1))
        emb_plas_padded = torch.cat([emb_plas, pad_plas], dim=0)
        embeddings_plas.append(emb_plas_padded)

    embeddings_chr = torch.stack(embeddings_chr)   # [B, max_len_chr, 3072]
    embeddings_plas = torch.stack(embeddings_plas) # [B, max_len_plas, 3072]

    # Create masks (True for padding)
    mask_chr = create_padding_mask(torch.tensor(len_chr_list), max_len_chr)
    mask_plas = create_padding_mask(torch.tensor(len_plas_list), max_len_plas)

    labels = torch.tensor(labels, dtype=torch.float32)  # [B]

    return (embeddings_chr, embeddings_plas), labels, (mask_chr, mask_plas)

# ============================
# CLIP Model Definition (Supervised Version)
# ============================
class CLIPModel(nn.Module):
    def __init__(self, input_dim=3072, num_heads=4):
        super(CLIPModel, self).__init__()

        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

        self.projection_head_chr = nn.Linear(input_dim, 128)
        self.projection_head_plas = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()

        self.self_attn_chr = nn.MultiheadAttention(embed_dim=128,
                                                   num_heads=num_heads,
                                                   batch_first=True)
        self.self_attn_plas = nn.MultiheadAttention(embed_dim=128,
                                                    num_heads=num_heads,
                                                    batch_first=True)

    def forward(self, emb_chr, emb_plas, mask_chr, mask_plas):
        emb_chr = self.relu(self.projection_head_chr(emb_chr))
        emb_plas = self.relu(self.projection_head_plas(emb_plas))

        # Self-attention
        attn_output_chr, _ = self.self_attn_chr(emb_chr, emb_chr, emb_chr, key_padding_mask=mask_chr)
        attn_output_plas, _ = self.self_attn_plas(emb_plas, emb_plas, emb_plas, key_padding_mask=mask_plas)

        valid_mask_chr = ~mask_chr  # [B, L_chr] => True where data is valid
        valid_mask_plas = ~mask_plas

        # Masked mean pooling
        chr_sum = (attn_output_chr * valid_mask_chr.unsqueeze(-1)).sum(dim=1)
        plas_sum = (attn_output_plas * valid_mask_plas.unsqueeze(-1)).sum(dim=1)

        chr_count = valid_mask_chr.sum(dim=1, keepdim=True).clamp(min=1)
        plas_count = valid_mask_plas.sum(dim=1, keepdim=True).clamp(min=1)

        chr_avg = chr_sum / chr_count
        plas_avg = plas_sum / plas_count

        # Normalize and compute scaled cosine similarity
        norm_chr = F.normalize(chr_avg, dim=-1)
        norm_plas = F.normalize(plas_avg, dim=-1)

        similarity = F.cosine_similarity(norm_chr, norm_plas, dim=1)
        similarity = similarity * self.logit_scale.exp()

        return similarity  # [B]

def supervised_loss(similarity, labels):
    return F.binary_cross_entropy_with_logits(similarity, labels)

# ============================
# Distributed Setup Function
# ============================
def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# ============================
# Metrics Computation Helper
# ============================
def compute_metrics(y_true, y_pred, y_prob):
    """
    Computes:
    - Accuracy
    - Balanced Accuracy (average of sensitivity and specificity)
    - Precision
    - Recall (Sensitivity)
    - Specificity
    - AUC
    - F1
    - MCC
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall_val = recall_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_val
    f1 = f1_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0
    balanced_acc = (sensitivity + specificity) / 2.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5

    # MCC
    mcc = matthews_corrcoef(y_true, y_pred)

    # Return a tuple of metrics
    return accuracy, balanced_acc, precision, recall_val, specificity, sensitivity, auc, f1, mcc

# ----------------
# Helper: gather a tensor from all ranks
# ----------------
def gather_tensor(local_tensor, device):
    """
    #### important notice: make sure the dataset size is divisible by the GPU node numbers. 
    #### other wise, it might cause error  //////// possibly,  DistributedSampler function wil resolve the problem by default
    Gathers a 1D or 2D tensor from all ranks, concatenating along dim=0.
    Returns the concatenated result on *every* rank.
    (If the dataset is huge, consider gathering only on rank 0 or do it in chunks.)
    """
    local_tensor = local_tensor.to(device) 
    # generate the size of the empty tensor based on the local_tensor of current GPU node (rank:0)
    gather_list = [torch.zeros_like(local_tensor) for _ in range(dist.get_world_size())]# the tensor list
    # we generate a list of empty tensor, and the tensor shape is same as that of local_tensor
    dist.all_gather(gather_list, local_tensor) # collect the tensor from all nodes in a list 
    # after this round, gather_list[0] contains the local_tensor from rank 0, same as gather_list[1] ... gather_list[world_size-1]
    return torch.cat(gather_list, dim=0)

if __name__ == '__main__':
    # Initialize distributed processing.
    local_rank = setup_distributed()
    device = torch.device("cuda", local_rank)
    
    # Create a folder for checkpoints/logs
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", "checkpoint_latest.pt")

    # Dataset paths (CSV includes "Chromosome", "Plasmid", and "Label")
    embeddings_folder = '../single_DNA_embeddings'
    train_csv = 'pos_neg_random_train_5.csv'
    val_csv = 'pos_neg_random_val_5.csv'
    test_csv = 'pos_neg_random_test_5.csv'
    train_dataset = DNAPairDataset(train_csv, embeddings_folder)
    val_dataset = DNAPairDataset(val_csv, embeddings_folder)
    test_dataset = DNAPairDataset(test_csv, embeddings_folder)
    
    if dist.get_rank() == 0:
        print("training dataset file is", train_csv)
        print("val dataset file is", train_csv)
        print("test dataset file is", train_csv)
        print("Embedding files' directory:", embeddings_folder)
        
    if dist.get_rank() == 0: # only run the bewlow codes at the rank: 0 gpu node
        print("Size of training, validation, and test sets:", len(train_dataset), len(val_dataset), len(test_dataset))
    
    # Create DistributedSamplers
    # By default, PyTorch’s DistributedSampler ensures each rank gets (approximately) the same number of samples, 
    # even if the dataset size is not perfectly divisible by your total number of processes (i.e., GPUs).
    # Default behavior (i.e., drop_last=False in the DistributedSampler constructor):
    # drop_last=True: 
    # Drops the leftover samples so that the total is divisible by world_size.
    # Each rank again gets the same number of samples, but the final “incomplete” samples are ignored for that epoch.
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset, shuffle=False)
    test_sampler  = DistributedSampler(test_dataset, shuffle=False)
    
    batch_size = 8  # per process
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=clip_collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=clip_collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=clip_collate_fn)
    
    # Create and wrap the model with DDP.
    model = CLIPModel().to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 50
    start_epoch = 0

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device,weights_only= False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        metrics_log = checkpoint.get('metrics_log', [])
        if dist.get_rank() == 0:
            print(f"Resuming training from epoch {start_epoch}")
    else:
        best_val_loss = float('inf')
        metrics_log = []
    
    for epoch in range(start_epoch, num_epochs):
        # ---------------------------
        # Training Loop
        # ---------------------------
        model.train()
        train_sampler.set_epoch(epoch)

        total_loss = 0
        num_samples = 0

        # Accumulate local predictions/labels for entire epoch on each rank
        train_preds_list = []
        train_labels_list = []
        train_probs_list = []
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training" , disable=(dist.get_rank() != 0)) # only enable the rank: 0; avoid duplicate progress bar
        for (emb_chr, emb_plas), labels, (mask_chr, mask_plas) in train_bar:
            # current_rank = dist.get_rank()
            # print(f"[Rank={current_rank}] emb_chr shape: {emb_chr.shape}") 
            # above function will return like [Rank=2] emb_chr shape: torch.Size([4,  5021, 3072])
            #                                 [Rank=0] emb_chr shape: torch.Size([4,  5211, 3072])
            # so on, becuase the four node receive the same command line, and therefore, print their own emb_chr.shape
            emb_chr = emb_chr.to(device)
            emb_plas = emb_plas.to(device)
            mask_chr = mask_chr.to(device)
            mask_plas = mask_plas.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            similarity = model(emb_chr, emb_plas, mask_chr, mask_plas)  # [B]
            loss = supervised_loss(similarity, labels)
            loss.backward()
            optimizer.step()

            batch_size_current = emb_chr.size(0)
            total_loss += loss.item() * batch_size_current
            num_samples += batch_size_current
            train_bar.set_postfix(loss=total_loss / num_samples) # the loss at rank: 0 

            preds = (similarity > 0).float()
            probs = torch.sigmoid(similarity)

            train_preds_list.append(preds.detach().cpu())
            train_labels_list.append(labels.detach().cpu())
            train_probs_list.append(probs.detach().cpu())

        # Locally computed training loss average
        train_loss_avg_local = total_loss / (num_samples if num_samples > 0 else 1)

        # Gather final training loss average across ranks (simple scalar reduce)
        train_loss_tensor = torch.tensor([train_loss_avg_local], device=device)
        # this part, actually the 
        # print(f"Rank={dist.get_rank()}, device={train_loss_tensor.device}, value={train_loss_tensor}")
        # import time
        # time.sleep(5)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        # print(f"Rank={dist.get_rank()}, device={train_loss_tensor.device}, value={train_loss_tensor}")
        # time.sleep(5)
        # node_rank = int(os.environ.get("NODE_RANK", 0))
        train_loss_avg = train_loss_tensor.item() / dist.get_world_size()
        # print(f"[Node={node_rank}, Rank={dist.get_rank()}] world_size={dist.get_world_size()}")
        # time.sleep(5)
        # print(f"[Node={node_rank}, Rank={dist.get_rank()}] train_loss_tensor.item() = {train_loss_tensor.item()}")
        # time.sleep(5)
        # print("train_loss_tensor", train_loss_avg)
        # time.sleep(5)
        # Gather local preds/labels/probs into a single tensor
        train_preds_all_local  = torch.cat(train_preds_list, dim=0)
        train_labels_all_local = torch.cat(train_labels_list, dim=0)
        train_probs_all_local  = torch.cat(train_probs_list, dim=0)

        # Use all_gather so each rank has complete predictions
        train_preds_all  = gather_tensor(train_preds_all_local,  device)
        train_labels_all = gather_tensor(train_labels_all_local, device)
        train_probs_all  = gather_tensor(train_probs_all_local,  device)
        
        # Compute metrics only on rank 0 using the aggregated predictions
        if dist.get_rank() == 0:
            train_preds_np  = train_preds_all.cpu().numpy()
            train_labels_np = train_labels_all.cpu().numpy()
            train_probs_np  = train_probs_all.cpu().numpy()

            train_metrics = compute_metrics(train_labels_np, train_preds_np, train_probs_np)
            # train_metrics is 9-values: 
            # [accuracy, balanced_acc, precision, recall, specificity, sensitivity, auc, f1, mcc]
        else:
            train_metrics = [float('nan')] * 9  # placeholders

        # -----------------------
        # Validation Evaluation
        # -----------------------
        model.eval()
        val_sampler.set_epoch(epoch)

        val_loss_total = 0
        total_val_samples = 0

        val_preds_list = []
        val_labels_list = []
        val_probs_list = []

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", disable=(dist.get_rank() != 0))
        with torch.no_grad():
            for (emb_chr, emb_plas), labels, (mask_chr, mask_plas) in val_bar:
                emb_chr = emb_chr.to(device)
                emb_plas = emb_plas.to(device)
                mask_chr = mask_chr.to(device)
                mask_plas = mask_plas.to(device)
                labels = labels.to(device)
                
                similarity = model(emb_chr, emb_plas, mask_chr, mask_plas)
                loss = supervised_loss(similarity, labels)

                batch_size_current = emb_chr.size(0)
                val_loss_total += loss.item() * batch_size_current
                total_val_samples += batch_size_current

                preds = (similarity > 0).float()
                probs = torch.sigmoid(similarity)

                val_preds_list.append(preds.detach().cpu())
                val_labels_list.append(labels.detach().cpu())
                val_probs_list.append(probs.detach().cpu())

        # Locally computed validation loss
        val_loss_avg_local = val_loss_total / (total_val_samples if total_val_samples > 0 else 1)

        # Gather val loss across ranks
        val_loss_tensor = torch.tensor([val_loss_avg_local], device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss_avg = val_loss_tensor.item() / dist.get_world_size()

        # Gather local val predictions
        val_preds_all_local  = torch.cat(val_preds_list, dim=0)
        val_labels_all_local = torch.cat(val_labels_list, dim=0)
        val_probs_all_local  = torch.cat(val_probs_list, dim=0)

        val_preds_all  = gather_tensor(val_preds_all_local,  device)
        val_labels_all = gather_tensor(val_labels_all_local, device)
        val_probs_all  = gather_tensor(val_probs_all_local,  device)

        if dist.get_rank() == 0:
            val_preds_np  = val_preds_all.cpu().numpy()
            val_labels_np = val_labels_all.cpu().numpy()
            val_probs_np  = val_probs_all.cpu().numpy()

            val_metrics = compute_metrics(val_labels_np, val_preds_np, val_probs_np)
        else:
            val_metrics = [float('nan')] * 9

        # -----------------------
        # Test Evaluation (only if improved)
        # -----------------------
        # We'll only do test eval if this rank-0 sees improvement
        # but we must broadcast the "improved" flag to ensure consistent behavior.
        improved = False
        if dist.get_rank() == 0 and val_loss_avg < best_val_loss:
            improved = True
        improved_flag = torch.tensor([1 if improved else 0], device=device) # generate a flag tensor(1), if improved
        dist.broadcast(improved_flag, src=0) #  broadcast the flat to each node, instead of only rank 0
        improved = bool(improved_flag.item()) # genearate the boolean label 

        if improved: # here every node has tbe improved label; False or True
            if dist.get_rank() == 0: # all rank has the same val_loss_avg after the dist.all_reduce, just ask one node to save file; no need all of them
                best_val_loss = val_loss_avg
                model_path = os.path.join("checkpoints", f"best_model_epoch_{epoch+1}.pt")
                # Only rank 0 saves the model state_dict
                torch.save(model.state_dict(), model_path)
                print(f"Validation loss improved to {val_loss_avg:.4f}. Saved model to {model_path}")

            # -----------------------------
            # All ranks do the test forward
            # -----------------------------
            test_loss_total = 0
            total_test_samples = 0
            test_preds_list = []
            test_labels_list = []
            test_probs_list = []

            with torch.no_grad():
                for (emb_chr, emb_plas), labels, (mask_chr, mask_plas) in tqdm(test_loader, desc="Testing", disable=(dist.get_rank() != 0)):
                    emb_chr = emb_chr.to(device)
                    emb_plas = emb_plas.to(device)
                    mask_chr = mask_chr.to(device)
                    mask_plas = mask_plas.to(device)
                    labels = labels.to(device)

                    similarity = model(emb_chr, emb_plas, mask_chr, mask_plas)
                    loss = supervised_loss(similarity, labels)

                    batch_size_current = emb_chr.size(0)
                    test_loss_total += loss.item() * batch_size_current
                    total_test_samples += batch_size_current

                    preds = (similarity > 0).float()
                    probs = torch.sigmoid(similarity)

                    test_preds_list.append(preds.detach().cpu())
                    test_labels_list.append(labels.detach().cpu())
                    test_probs_list.append(probs.detach().cpu())

            # Locally computed test loss
            test_loss_avg_local = test_loss_total / (total_test_samples if total_test_samples > 0 else 1)

            # Sum across ranks
            test_loss_tensor = torch.tensor([test_loss_avg_local], device=device)
            dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
            test_loss_avg = test_loss_tensor.item() / dist.get_world_size()

            # Gather local test predictions
            test_preds_all_local  = torch.cat(test_preds_list, dim=0)
            test_labels_all_local = torch.cat(test_labels_list, dim=0)
            test_probs_all_local  = torch.cat(test_probs_list, dim=0)

            test_preds_all  = gather_tensor(test_preds_all_local,  device)
            test_labels_all = gather_tensor(test_labels_all_local, device)
            test_probs_all  = gather_tensor(test_probs_all_local,  device)

            if dist.get_rank() == 0:
                test_preds_np  = test_preds_all.cpu().numpy()
                test_labels_np = test_labels_all.cpu().numpy()
                test_probs_np  = test_probs_all.cpu().numpy()

                test_metrics = compute_metrics(test_labels_np, test_preds_np, test_probs_np)
            else:
                test_metrics = [float('nan')] * 9
        else:
            # If not improved, we do not evaluate on test => placeholders
            test_loss_avg = float('nan')
            test_metrics = [float('nan')] * 9

        # -----------------------
        # Logging & Saving
        # -----------------------
        if dist.get_rank() == 0:
            # Unpack train/val/test metrics
            # Each is 9 values: 
            # [acc, bal_acc, prec, recall, spec, sens, auc, f1, mcc]
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            print(f"Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")

            print("Training Metrics:")
            print(f"  Accuracy:          {train_metrics[0]:.4f}")
            print(f"  Balanced Accuracy: {train_metrics[1]:.4f}")
            print(f"  Precision:         {train_metrics[2]:.4f}")
            print(f"  Recall:            {train_metrics[3]:.4f}")
            print(f"  Specificity:       {train_metrics[4]:.4f}")
            print(f"  Sensitivity:       {train_metrics[5]:.4f}")
            print(f"  AUC:               {train_metrics[6]:.4f}")
            print(f"  F1:                {train_metrics[7]:.4f}")
            print(f"  MCC:               {train_metrics[8]:.4f}\n")

            print("Validation Metrics:")
            print(f"  Accuracy:          {val_metrics[0]:.4f}")
            print(f"  Balanced Accuracy: {val_metrics[1]:.4f}")
            print(f"  Precision:         {val_metrics[2]:.4f}")
            print(f"  Recall:            {val_metrics[3]:.4f}")
            print(f"  Specificity:       {val_metrics[4]:.4f}")
            print(f"  Sensitivity:       {val_metrics[5]:.4f}")
            print(f"  AUC:               {val_metrics[6]:.4f}")
            print(f"  F1:                {val_metrics[7]:.4f}")
            print(f"  MCC:               {val_metrics[8]:.4f}")

            if not math.isnan(test_loss_avg):
                print("\nTest Metrics (NEW BEST):")
                print(f"  Loss:              {test_loss_avg:.4f}")
                print(f"  Accuracy:          {test_metrics[0]:.4f}")
                print(f"  Balanced Accuracy: {test_metrics[1]:.4f}")
                print(f"  Precision:         {test_metrics[2]:.4f}")
                print(f"  Recall:            {test_metrics[3]:.4f}")
                print(f"  Specificity:       {test_metrics[4]:.4f}")
                print(f"  Sensitivity:       {test_metrics[5]:.4f}")
                print(f"  AUC:               {test_metrics[6]:.4f}")
                print(f"  F1:                {test_metrics[7]:.4f}")
                print(f"  MCC:               {test_metrics[8]:.4f}\n")
            else:
                print("\nTest Metrics: Not evaluated in this epoch.\n")

            # Save metrics into a dictionary
            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": train_loss_avg,
                "train_accuracy": train_metrics[0],
                "train_balanced_accuracy": train_metrics[1],
                "train_precision": train_metrics[2],
                "train_recall": train_metrics[3],
                "train_specificity": train_metrics[4],
                "train_sensitivity": train_metrics[5],
                "train_auc": train_metrics[6],
                "train_f1": train_metrics[7],
                "train_mcc": train_metrics[8],  # <-- add MCC
                "val_loss": val_loss_avg,
                "val_accuracy": val_metrics[0],
                "val_balanced_accuracy": val_metrics[1],
                "val_precision": val_metrics[2],
                "val_recall": val_metrics[3],
                "val_specificity": val_metrics[4],
                "val_sensitivity": val_metrics[5],
                "val_auc": val_metrics[6],
                "val_f1": val_metrics[7],
                "val_mcc": val_metrics[8],      # <-- add MCC
                "test_loss": test_loss_avg,
                "test_accuracy": test_metrics[0],
                "test_balanced_accuracy": test_metrics[1],
                "test_precision": test_metrics[2],
                "test_recall": test_metrics[3],
                "test_specificity": test_metrics[4],
                "test_sensitivity": test_metrics[5],
                "test_auc": test_metrics[6],
                "test_f1": test_metrics[7],
                "test_mcc": test_metrics[8],    # <-- add MCC
            }

            metrics_log.append(epoch_log)

            # Save all metrics to CSV in the checkpoints folder
            df_metrics = pd.DataFrame(metrics_log)
            df_metrics.to_csv(os.path.join("checkpoints", "metrics_log_strict_random_1_projection_head_128_h4.csv"), index=False)

            # Save checkpoint for resuming
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'metrics_log': metrics_log
            }
            torch.save(checkpoint, checkpoint_path)
