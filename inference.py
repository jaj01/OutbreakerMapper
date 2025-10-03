# inference.py (minimal template)
import torch
import numpy as np
import pandas as pd
import os
from typing import Dict, Any

# Import your model class (ensure model.py is in PYTHONPATH or same folder)
# from src.model import TGCN_GRU

def load_model(model_path: str, device: str = None):
    """
    Load the trained PyTorch model. Adapt the model constructor below to match your saved model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # === ADAPT: instantiate model with same hyperparams used in training ===
    # in_feats = ... (number of node features)
    # model = TGCN_GRU(in_feats=in_feats, gcn_hidden=64, gru_hidden=64, dropout=0.2)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.to(device)
    # model.eval()
    # return model

    raise NotImplementedError("Fill in model creation + load lines to match your project.")

def run_counterfactual_prediction(model, node_features_seq, edge_index_seq, edge_weight_seq, mobility_scale=1.0, device=None):
    """
    Run inference with scaled mobility.
    - model: loaded TGCN_GRU (in eval mode)
    - node_features_seq: list/array of length T of node-feature arrays [N, F] (or tensor shape [T,N,F])
    - edge_index_seq: list of edge_index tensors for each t (PyG style, shape [2,E])
    - edge_weight_seq: list of edge_weight arrays for each t (length E)
    - mobility_scale: float in [0,1] where 0=full mobility lockdown, 1=no change.
    Returns:
    - preds: np.array shape [N] predicted case counts
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Ensure tensors
    import torch
    # Convert seqs to tensors if needed:
    if isinstance(node_features_seq, list):
        x_seq = torch.stack([torch.tensor(x, dtype=torch.float32) for x in node_features_seq], dim=0)  # [T,N,F]
    else:
        x_seq = torch.tensor(node_features_seq, dtype=torch.float32)

    # Apply mobility_scale: we assume edge_weight_seq contains mobility weights at indices corresponding to mobility edges
    # If you have separate spatial/mobility edge attributes, multiply only mobility part by mobility_scale.
    scaled_edge_weights = []
    for w in edge_weight_seq:
        w_t = np.array(w, dtype=float)
        # === ADAPT: If your edge_attr contains multiple channels (e.g., [inv_dist, mob_norm]),
        # you should scale only the mobility channel. Example below assumes single weight per edge representing combined weight.
        w_t = w_t * mobility_scale
        scaled_edge_weights.append(torch.tensor(w_t, dtype=torch.float32))

    # Move data to device
    x_seq = x_seq.to(device)
    edge_index_seq_t = [ei.to(device) if hasattr(ei, 'to') else ei for ei in edge_index_seq]
    edge_weight_seq_t = [ew.to(device) if hasattr(ew, 'to') else ew for ew in scaled_edge_weights]

    # Model forward: calls model(x_seq, edge_index_seq_t, edge_weight_seq_t) depending on your model signature
    model.to(device)
    model.eval()
    with torch.no_grad():
        preds = model(x_seq, edge_index_seq_t, edge_weight_seq_t)  # adapt signature if different
        preds = preds.detach().cpu().numpy()

    return preds
