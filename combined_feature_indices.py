import torch
import numpy as np
import os
 
# Path where your 10 fold indices are located
folder_path = "./"  
base_filename = "rfe_indices_"          #change name of the Feature selection algorithm in use
num_folds = 10

all_indices = []

for i in range(1, num_folds + 1):
    file_path = os.path.join(folder_path, f"{base_filename}{i}.pt")
    indices = torch.load(file_path, weights_only=False)
    all_indices.append(np.array(indices))


flattened = np.concatenate(all_indices)
max_dim = flattened.max() + 1
frequency = np.zeros(max_dim, dtype=int)

for idx in flattened:
    frequency[idx] += 1


K = len(all_indices[0])
top_k_indices = np.argsort(frequency)[-K:][::-1].copy()
torch.save(torch.tensor(top_k_indices), os.path.join(folder_path, "rfe_indices.pt"))


print("Saved consensus indices")
