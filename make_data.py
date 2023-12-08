from rdkit import Chem                                                                                                                                                                   
from rdkit import RDLogger          
from rdkit.Chem import AllChem
from tqdm.auto import tqdm
import pickle
import numpy as np                                                                                                                              
import os
from torch.utils.data import Dataset
import protein
import residue_constants
from datasets import LigProtDatabase

import torch
import torch.nn.functional as F

RDLogger.DisableLog('rdApp.*')                                                                                                                                                           

def main():
        
    directory = '/Users/jihyeonje/Downloads/PDBBind_processed'
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
            elif i.endswith('protein.pdb'):
                protpaths.append(os.path.join(foldr, i))

    clean_ligpaths = []
    clean_protpaths = []
    for i in range(len(ligpaths)):
        suppl = Chem.SDMolSupplier(ligpaths[i])
        if suppl is None: continue
        try:
            updated_mol= Chem.AddHs(suppl[0])
            AllChem.EmbedMolecule(updated_mol)
            clean_ligpaths.append(ligpaths[i])
            clean_protpaths.append(protpaths[i])
        except:
            print('fail')
            pass
    
    filehandler = open("/Users/jihyeonje/unidiffuser/cleanligs.pickle","wb")
    pickle.dump(clean_ligpaths,filehandler)
    filehandler.close()
        
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

    train_prot, train_lig, valid_prot, valid_lig = make_train_valid_dfs(clean_protpaths, clean_ligpaths)

    datas = LigProtDatabase(protpaths = train_prot, ligpaths = train_lig)
    save_dir = f'/Users/jihyeonje/unidiffuser/test/feats/train'

    for idx, data in tqdm(enumerate(datas)):
        if data is not None:
            try:
                prot, lig = data
                
                c_pad = torch.zeros(3, 1296 - prot.shape[1], 4)
                prot = torch.cat([prot, c_pad], dim=1)
            
                pad_lig = torch.zeros(100, lig.shape[1])
                pad_lig[:lig.shape[0], :] = lig
                prot = prot.detach().cpu().numpy()
                lig = pad_lig.detach().cpu().numpy()
                np.save(os.path.join(save_dir, f'{idx}.npy'), (prot,lig))
            except:
                pass


    datas = LigProtDatabase(protpaths = valid_prot, ligpaths = valid_lig)
    save_dir = f'/Users/jihyeonje/unidiffuser/test/feats/test'

    for idx, data in tqdm(enumerate(datas)):
        if data is not None:
            try:
                prot, lig = data
                
                c_pad = torch.zeros(3, 1296 - prot.shape[1], 4)
                prot = torch.cat([prot, c_pad], dim=1)
            
                pad_lig = torch.zeros(100, lig.shape[1])
                pad_lig[:lig.shape[0], :] = lig
                prot = prot.detach().cpu().numpy()
                lig = pad_lig.detach().cpu().numpy()
                np.save(os.path.join(save_dir, f'{idx}.npy'), (prot,lig))
            except:
                pass

if __name__ == "__main__":
    main()
