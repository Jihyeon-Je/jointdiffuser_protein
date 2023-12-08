from rdkit import Chem                                                                                                                                                                   
from rdkit import RDLogger          
from rdkit.Chem import AllChem
from tqdm.auto import tqdm
       
import numpy as np                                                                                                                              
import os
from torch.utils.data import Dataset
import protein
import residue_constants
import einops
from datasets import LigProtDatabase, mol_extraction, load_feats_from_pdb

import torch
import torch.nn.functional as F

RDLogger.DisableLog('rdApp.*')                                                                                                                                                           

import pickle

directory = '/Users/jihyeonje/Downloads/PDBBind_processed/'
ligpaths = []
protpaths = []

# iterate over files in
# that directory
for dir in os.listdir(directory):
    if dir !='.DS_Store':
        foldr = os.path.join(directory, dir)
    for i in os.listdir(foldr):
        if i.endswith('.sdf'):
            ligpaths.append(os.path.join(foldr, i))
        elif i.endswith('.pdb'):
            protpaths.append(os.path.join(foldr, i))

def make_train_valid_dfs(protpaths, ligpaths):
    max_id = len(protpaths)
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_prot = list(map(protpaths.__getitem__, train_ids))
    train_ligs = list(map(ligpaths.__getitem__, train_ids))
    valid_prot = list(map(protpaths.__getitem__, valid_ids))
    valid_ligs = list(map(ligpaths.__getitem__, valid_ids))
    return train_prot, train_ligs, valid_prot, valid_ligs

train_prot, train_lig, valid_prot, valid_lig = make_train_valid_dfs(protpaths, ligpaths)

save_dir = f'/Users/jihyeonje/unidiffuser/test/feats/train'
idx = 0
for i in range(len(train_prot)):
    protpath = train_prot[i]
    ligpath = train_lig[i]
    
    suppl = Chem.SDMolSupplier(ligpath, sanitize=False)
    try:
        x, atom_types, one_hot = mol_extraction(suppl[0])
        lig = torch.cat([torch.tensor(x), torch.tensor(one_hot)], dim=1)
        
        p_feats = load_feats_from_pdb(protpath)
        bb_coords = p_feats['bb_coords']
        prot = einops.rearrange(bb_coords, 'h w c -> c h w')
        
        c_pad = torch.zeros(3, 1296 - prot.shape[1], 4)
        prot = torch.cat([prot, c_pad], dim=1)
        
        pad_lig = torch.zeros(100, lig.shape[1])
        pad_lig[:lig.shape[0], :] = lig
        prot = prot.detach().cpu().numpy()
        lig = pad_lig.detach().cpu().numpy()
        np.save(os.path.join(save_dir, f'{idx}.npy'), (prot,lig))
        idx +=1
    except:
        pass


save_dir = f'/Users/jihyeonje/unidiffuser/test/feats/test'

idx = 0
for i in range(len(valid_prot)):
    protpath = valid_prot[i]
    ligpath = valid_lig[i]
    
    suppl = Chem.SDMolSupplier(ligpath, sanitize=False)
    try:
        x, atom_types, one_hot = mol_extraction(suppl[0])
        lig = torch.cat([torch.tensor(x), torch.tensor(one_hot)], dim=1)
        
        p_feats = load_feats_from_pdb(protpath)
        bb_coords = p_feats['bb_coords']
        prot = einops.rearrange(bb_coords, 'h w c -> c h w')
        
        c_pad = torch.zeros(3, 1296 - prot.shape[1], 4)
        prot = torch.cat([prot, c_pad], dim=1)
        
        pad_lig = torch.zeros(100, lig.shape[1])
        pad_lig[:lig.shape[0], :] = lig
        prot = prot.detach().cpu().numpy()
        lig = pad_lig.detach().cpu().numpy()
        np.save(os.path.join(save_dir, f'{idx}.npy'), (prot,lig))
        idx +=1
    except:
        pass
