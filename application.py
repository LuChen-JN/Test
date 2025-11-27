from time import time
from torch.utils.data import DataLoader
import torch
import argparse
import warnings, os
import pandas as pd
import logging
logging.disable(logging.WARNING)
import numpy as np
from utils import graph_collate_func
from dataloader import NTIDataset
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import model_glu
import sys

# Model application
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="FNA-SpeEvo for NTI prediction")
parser.add_argument('--data', type=str, metavar='TASK',help='dataset',default="visual")
args = parser.parse_args()

def visual(id,smiles,attention_weights):
        """
        Compounds are visualized with RDKit and attention weights highlighted in red on atoms.

        parameters:
        - attention_weights:(torch. Tensor or np.ndarray) is an attention matrix of the shape [num_atoms, 128].
        - smiles: (str) The SMILES string of the compound
        """
        mol = Chem.MolFromSmiles(smiles)
        atom_counts = 0
        for atom in mol.GetAtoms():
            atomic_symbol = atom.GetSymbol()
            if atomic_symbol != 'H':  # Exclude hydrogen atoms
              atom_counts+=1

        attention_weights=attention_weights[:atom_counts,:]
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()

        if mol is None:
            raise ValueError("Invalid SMILES string。")

        # Calculate the average attention weight for each atom
        atom_weights = attention_weights.mean(axis=1)  # Calculate the mean
        atom_weights = atom_weights / np.max(atom_weights)

        top_n_atoms = np.argsort(atom_weights)[-5:].tolist()

        if len(atom_weights) != atom_counts:
            raise ValueError(f"The length of the attention weight {len(atom_weights)} != "
                             f"The number of molecule atoms {atom_counts} ")

        # Use RDKit to draw molecular images
        img = Draw.MolToImage(mol, size=(400, 400), highlightAtoms=top_n_atoms,
                            highlightAtomColors="red")
        
        # Display the image
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.axis('off')
        #plt.title('Molecular Attention Visualization', fontsize=12)
        plt.savefig(f"{id}.tif",bbox_inches='tight', dpi=600,transparent=True)
        plt.close()

def save_metrics(metric, filename):
        with open(filename, 'a') as f:
            f.write(' '.join(map(str, metric)) + '\n')



#  Please set the file path according to your PC
torch.cuda.empty_cache()
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
dataFolder = f'./datasets/{args.data}'
test_path=os.path.join(dataFolder, 'predict.csv')
df_test=pd.read_csv(test_path)
test_dataset=NTIDataset(df_test.index.values,df_test,True)
dataset_test=DataLoader(test_dataset,shuffle=False,collate_fn= graph_collate_func,batch_size=1,num_workers=0,drop_last=False)

protein_dim = 128
atom_dim = 128
hid_dim = 128
n_layers = 3
n_heads = 8
pf_dim = 256
dropout = 0.1
batch = 64
lr = 6e-4
weight_decay = 1e-4
decay_interval = 30
iteration = 100
kernel_size = 7

encoder = model_glu.Encoder(protein_dim, hid_dim, 3, kernel_size, dropout, device)
decoder = model_glu.Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, model_glu.DecoderLayer,model_glu.SelfAttention,model_glu.PositionwiseFeedforward, dropout, device)
model = model_glu.Predictor2(encoder, decoder, device)
#model.load_state_dict(torch.load("model.pt",weights_only=True))
model.load_state_dict(torch.load("model.pt"))
model.to(device)
model.eval()


# Whether visualization is required
model_glu.picture=False
# model_glu.picture=True

# Create a file of the prediction results
file_name="predict.txt"
with open(file_name, 'w') as f:
    f.write(' '.join(map(str, ["SMILES","Sequence","Y"])) + '\n')


with torch.no_grad():
    for idx, (temp1, temp2, temp3) in enumerate(dataset_test):
        temp1 = temp1.to(device)
        temp2 = temp2.to(device)

        smile=df_test.iloc[idx]["SMILES"]
        dna=df_test.iloc[idx]["Sequence"]
        predicted_labels= model(temp1, temp2, train=False)
        
        thisMaxtrix=model_glu.maxtrix

        metrics = [smile, dna, predicted_labels[0]]
        save_metrics(metrics, file_name)
        if model_glu.picture:
            visual(idx,smile,thisMaxtrix)







