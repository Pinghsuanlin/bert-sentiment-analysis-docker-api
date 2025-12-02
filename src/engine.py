# ============================ Define Training and Evaluation Functions ============================= #
# functions to train and evaluate the BERT model, handling forward and backward passes, loss computation, and metric calculations.
# Which abstracts the training and evaluation steps to keep the main script clean

from sched import scheduler
import torch.nn as nn
import torch
from tqdm import tqdm



# No module-level loss function; pass as parameter to functions

bce_loss = nn.BCEWithLogitsLoss()

def train_fn(data_loader, model, optimizer, device, accumulation_steps, scheduler, loss_fn):
    # set model to training mode (enables dropout)
    model.train()
    # clear gradients
    optimizer.zero_grad()

    for bi, d in enumerate(data_loader):
        ids = d['ids'].to(device, dtype=torch.long) # move tensors to GPU/CPU and convert to long (integer) dtype
        token_type_ids = d['token_type_ids'].to(device, dtype=torch.long)
        mask = d['mask'].to(device, dtype=torch.long)
        targets = d['targets'].to(device, dtype=torch.float)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        # compute loss
        loss = loss_fn(outputs, targets)
        # backward pass
        loss.backward()

        # run forward pass
        if (bi + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # Handle last batch if not a multiple of accumulation_steps
    if len(data_loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


def eval_fn(data_loader, model, device, loss_fn=bce_loss):
    model.eval()
    fin_outputs = []
    fin_targets = []
    with torch.no_grad():
        for d in tqdm(data_loader, total=len(data_loader)):
            ids = d['ids'].to(device, dtype=torch.long)
            token_type_ids = d['token_type_ids'].to(device, dtype=torch.long)
            mask = d['mask'].to(device, dtype=torch.long)
            targets = d['targets'].to(device, dtype=torch.float)

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                token_type_ids=token_type_ids
            )

            # Use in-place sigmoid and move to CPU in batches
            batch_outputs = torch.sigmoid(outputs).cpu().detach().numpy()
            batch_targets = targets.cpu().detach().numpy()
            fin_outputs.extend(batch_outputs.tolist())
            fin_targets.extend(batch_targets.tolist())
    
    return fin_outputs, fin_targets
