from datasets import atom_encoder, atom_decoder
import os, sys
import glob
import pickle
import numpy as np
import Bio.PDB


bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
                'Cl': 127, 'Br': 141, 'I': 161},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
                'I': 214},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
                'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
                'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
                'I': 194},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
                'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
                'I': 187},
          'B': {'H':  119, 'Cl': 175},
          'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
                 'F': 160, 'Cl': 202, 'Br': 215, 'I': 243 },
          'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
                 'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
                 'Br': 214},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
                'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
                'I': 234},
          'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
                 'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
          'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
                'S': 210, 'F': 156, 'N': 177, 'Br': 222},
          'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
                'S': 234, 'F': 187, 'I': 266},
          'As': {'H': 152}
          }

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
          'P': {'O': 150, 'S': 186},
          'S': {'P': 186}}


bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}


def print_table(bonds_dict):
    letters = ['H', 'C', 'O', 'N', 'P', 'S', 'F', 'Si', 'Cl', 'Br', 'I']

    new_letters = []
    for key in (letters + list(bonds_dict.keys())):
        if key in bonds_dict.keys():
            if key not in new_letters:
                new_letters.append(key)

    letters = new_letters

    for j, y in enumerate(letters):
        if j == 0:
            for x in letters:
                print(f'{x} & ', end='')
            print()
        for i, x in enumerate(letters):
            if i == 0:
                print(f'{y} & ', end='')
            if x in bonds_dict[y]:
                print(f'{bonds_dict[y][x]} & ', end='')
            else:
                print('- & ', end='')
        print()


# print_table(bonds3)


def check_consistency_bond_dictionaries():
    for bonds_dict in [bonds1, bonds2, bonds3]:
        for atom1 in bonds1:
            for atom2 in bonds_dict[atom1]:
                bond = bonds_dict[atom1][atom2]
                try:
                    bond_check = bonds_dict[atom2][atom1]
                except KeyError:
                    raise ValueError('Not in dict ' + str((atom1, atom2)))

                assert bond == bond_check, (
                    f'{bond} != {bond_check} for {atom1}, {atom2}')


stdv = {'H': 5, 'C': 1, 'N': 1, 'O': 2, 'F': 3}
margin1, margin2, margin3 = 10, 5, 3

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3,
                 'Si': 4, 'P': [3, 5],
                 'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2],
                 'Bi': [3, 5]}


def get_bond_order(atom1, atom2, distance, check_exists=True):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if (atom1 not in bonds1) or (atom2 not in bonds1) :
            return 0
        
        if atom2 not in bonds1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of
    # the QM9 true samples.
    if distance < bonds1[atom1][atom2] + margin1:

        # Check if atoms in bonds2 dictionary.
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2
            if distance < thr_bond2:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3
                    if distance < thr_bond3:
                        return 3        # Triple
                return 2            # Double
        return 1                # Single
    return 0                    # No bond


import numpy as np
import torch

import plotly.graph_objects as go

qm9_with_h = {
    'name': 'qm9',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'n_nodes': {22: 3393, 17: 13025, 23: 4848, 21: 9970, 19: 13832, 20: 9482, 16: 10644, 13: 3060,
                15: 7796, 25: 1506, 18: 13364, 12: 1689, 11: 807, 24: 539, 14: 5136, 26: 48, 7: 16, 10: 362,
                8: 49, 9: 124, 27: 266, 4: 4, 29: 25, 6: 9, 5: 5, 3: 1},
    'max_n_nodes': 29,
    'atom_types': {1: 635559, 2: 101476, 0: 923537, 3: 140202, 4: 2323},
    'distances': [903054, 307308, 111994, 57474, 40384, 29170, 47152, 414344, 2202212, 573726,
                  1490786, 2970978, 756818, 969276, 489242, 1265402, 4587994, 3187130, 2454868, 2647422,
                  2098884,
                  2001974, 1625206, 1754172, 1620830, 1710042, 2133746, 1852492, 1415318, 1421064, 1223156,
                  1322256,
                  1380656, 1239244, 1084358, 981076, 896904, 762008, 659298, 604676, 523580, 437464, 413974,
                  352372,
                  291886, 271948, 231328, 188484, 160026, 136322, 117850, 103546, 87192, 76562, 61840,
                  49666, 43100,
                  33876, 26686, 22402, 18358, 15518, 13600, 12128, 9480, 7458, 5088, 4726, 3696, 3362, 3396,
                  2484,
                  1988, 1490, 984, 734, 600, 456, 482, 378, 362, 168, 124, 94, 88, 52, 44, 40, 18, 16, 8, 6,
                  2,
                  0, 0, 0, 0,
                  0,
                  0, 0],
    'colors_dic': ['#FFFFFF99', 'C7', 'C0', 'C3', 'C1'],
    'radius_dic': [0.46, 0.77, 0.77, 0.77, 0.77],
    'with_h': True}

def draw_mol(pos, atoms, mask=None):
    color_map = {'C': 'lemonchiffon',
             'O': 'lightblue',
             'N': 'lavender',
             'S': 'lightcyan',
             'H': 'lightgreen',
             'F': 'orange'} 
    
    #positions = pos.view(-1, 3)
    #positions_centered = positions - positions.mean(dim=0, keepdim=True)

    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    color_code=[]
    #for i in range(len(atoms)):
    #    color_code.append(color_map[qm9_with_h['atom_decoder'][atoms[i]]])

    x_edges_1 = []
    y_edges_1 = []
    z_edges_1 = []

    x_edges_2 = []
    y_edges_2 = []
    z_edges_2 = []
    
    x_edges_3 = []
    y_edges_3 = []
    z_edges_3 = []
    
    x_mask = []
    y_mask = []
    z_mask = []

    x_mask_node = []
    y_mask_node = []
    z_mask_node = []
    
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atoms[i]], \
                           atom_decoder[atoms[j]]
            draw_edge_int = get_bond_order(atom1, atom2, dist)
            if mask is not None:
                for k in range(len(mask)):
                    x_mask_node.append(x[mask[k]])
                    y_mask_node.append(y[mask[k]])
                    z_mask_node.append(z[mask[k]])
                if (i in mask or j in mask) & (draw_edge_int!=0):
                    x_coors = [p1[0],p2[0],None]
                    x_mask += x_coors
                    y_coors = [p1[1],p2[1],None]
                    y_mask += y_coors
                    z_coors = [p1[2],p2[2],None]
                    z_mask += z_coors
            if draw_edge_int==1:
                x_coors = [p1[0],p2[0],None]
                x_edges_1 += x_coors
                y_coors = [p1[1],p2[1],None]
                y_edges_1 += y_coors
                z_coors = [p1[2],p2[2],None]
                z_edges_1 += z_coors
            elif draw_edge_int==2:
                x_coors = [p1[0],p2[0],None]
                x_edges_2 += x_coors
                y_coors = [p1[1],p2[1],None]
                y_edges_2 += y_coors
                z_coors = [p1[2],p2[2],None]
                z_edges_2 += z_coors
            elif draw_edge_int==3:
                x_coors = [p1[0],p2[0],None]
                x_edges_3 += x_coors
                y_coors = [p1[1],p2[1],None]
                y_edges_3 += y_coors
                z_coors = [p1[2],p2[2],None]
                z_edges_3 += z_coors
    #create a trace for the nodes
    trace_nodes = go.Scatter3d(x=x,
                            y=y,
                            z=z,
                            mode='markers',
                            marker=dict(symbol='circle',
                                        size=10,
                                        #color=color_code,
                                        line=dict(color='black', width=0.5)),
                            text=atoms,
                            hoverinfo='text')
    
    trace_edges1 = go.Scatter3d(x=x_edges_1,
                        y=y_edges_1,
                        z=z_edges_1,
                        mode='lines',
                        line=dict(color='grey', width=10),
                        hoverinfo='none'
)
    trace_edges2 = go.Scatter3d(x=x_edges_2,
                    y=y_edges_2,
                    z=z_edges_2,
                    mode='lines',
                    line=dict(color='darkgrey', width=11),
                    hoverinfo='none'
)
    trace_edges3 = go.Scatter3d(x=x_edges_3,
                y=y_edges_3,
                z=z_edges_3,
                mode='lines',
                line=dict(color='black', width=12),
                hoverinfo='none'
)
    
    #we need to set the axis for the plot 
    axis = dict(showbackground=False,
                showline=True,
                zeroline=False,
                showgrid=True,
                showticklabels=False,
                title='')
    #also need to create the layout for our plot
    layout = go.Layout(
                    width=650,
                    height=625,
                    showlegend=False,
                    scene=dict(xaxis=dict(axis),
                            yaxis=dict(axis),
                            zaxis=dict(axis),
                            ),
                    margin=dict(t=100),
                    hovermode='closest')
    
    #Include the traces we want to plot and create a figure
    data = [trace_nodes, trace_edges1, trace_edges2, trace_edges3]
    fig = go.Figure(data=data, layout=layout)

    fig.show()




def extract_edges(pos, atoms):
   
    positions = pos.view(-1, 3)
    positions_centered = positions - positions.mean(dim=0, keepdim=True)

    x = positions_centered[:, 0]
    y = positions_centered[:, 1]
    z = positions_centered[:, 2]

    x_edges_1 = []
    y_edges_1 = []
    z_edges_1 = []

    x_edges_2 = []
    y_edges_2 = []
    z_edges_2 = []
    
    x_edges_3 = []
    y_edges_3 = []
    z_edges_3 = []
    
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = qm9_with_h['atom_decoder'][atoms[i]], \
                            qm9_with_h['atom_decoder'][atoms[j]]
            draw_edge_int = bond_analyze.get_bond_order(atom1, atom2, dist)

            if draw_edge_int==1:
                x_coors = [p1[0],p2[0],None]
                x_edges_1 += x_coors
                y_coors = [p1[1],p2[1],None]
                y_edges_1 += y_coors
                z_coors = [p1[2],p2[2],None]
                z_edges_1 += z_coors
            elif draw_edge_int==2:
                x_coors = [p1[0],p2[0],None]
                x_edges_2 += x_coors
                y_coors = [p1[1],p2[1],None]
                y_edges_2 += y_coors
                z_coors = [p1[2],p2[2],None]
                z_edges_2 += z_coors
            elif draw_edge_int==3:
                x_coors = [p1[0],p2[0],None]
                x_edges_3 += x_coors
                y_coors = [p1[1],p2[1],None]
                y_edges_3 += y_coors
                z_coors = [p1[2],p2[2],None]
                z_edges_3 += z_coors
    edge1 = {'x': x_edges_1, 'y': y_edges_1, 'z':z_edges_1}
    edge2 = {'x': x_edges_2, 'y': y_edges_2, 'z':z_edges_2}
    edge3 = {'x': x_edges_3, 'y': y_edges_3, 'z':z_edges_3}
    return x, y, z, edge1, edge2, edge3

from rdkit import Chem



bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]

def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)

def build_xae_molecule(positions, atom_types, atom_decoder):
    """ Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool)                  (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
    """
    atom_decoder = atom_decoder
    n = positions.shape[0]
    X = atom_types
    A = torch.zeros((n, n), dtype=torch.bool)
    E = torch.zeros((n, n), dtype=torch.int)
    positions = torch.tensor(positions)
    pos = positions.unsqueeze(0)
    dists = torch.cdist(pos, pos, p=2).squeeze(0)
    for i in range(n):
        for j in range(i):
            pair = sorted([atom_types[i], atom_types[j]])
            order = get_bond_order(atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j])
            
            if order > 0:
                # Warning: the graph should be DIRECTED
                A[i, j] = 1
                E[i, j] = order
    return X, A, E


def build_molecule(positions, atom_types, atom_decoder):
    atom_decoder = atom_decoder
    X, A, E = build_xae_molecule(positions, atom_types, atom_decoder)
    mol = Chem.RWMol()
    for atom in X:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    all_bonds = torch.nonzero(A)
    for bond in all_bonds:
        mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()])
    return mol

class BasicMolecularMetrics(object):
    def __init__(self, atom_decoder, dataset_smiles_list=None):
        self.atom_decoder = atom_decoder


    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []

        for graph in generated:
            mol = build_molecule(*graph, self.atom_decoder)
            smiles = mol2smiles(mol)
            if smiles is not None:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                valid.append(smiles)

        return valid, len(valid) / len(generated)

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    # def compute_novelty(self, unique):
    #     num_novel = 0
    #     novel = []
    #     for smiles in unique:
    #         if smiles not in self.dataset_smiles_list:
    #             novel.append(smiles)
    #             num_novel += 1
    #     return novel, num_novel / len(unique)

    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid, validity = self.compute_validity(generated)
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            
        else:
            validity = 0
            uniqueness = 0.0
        return [validity, uniqueness]

def check_stability(positions, atom_type, debug=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            pair = sorted([atom_type[i], atom_type[j]])
            order = get_bond_order(atom1, atom2, dist)
            
            nr_bonds[i] += order
            nr_bonds[j] += order
    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = allowed_bonds[atom_decoder[atom_type_i]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        if not is_stable and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    ratio = nr_stable_bonds/len(x)
    return molecule_stable, nr_stable_bonds, ratio

def parse_bond(coords):
    """
    Bond length matrix has order N-CA, CA-C, C-O, C-N
    Bond angle matrix has order N-CA-C, CA-C-O, CA-C-N, O-C-N
    """
    bb_atoms = ['N', 'CA', 'C', 'O']
    coords = np.transpose(np.stack(coords), (1,2,0))
    #coords = np.stack(coords).reshape((-1, 4, 3))
    nres = coords.shape[0]
    
    bond_lengths = np.zeros((nres, 4, 2, 3))
    bond_angles = np.zeros((nres, 4, 3, 3))
    n_idx, ca_idx, c_idx, o_idx = 0, 1, 2, 3
    n_ca, ca_c, c_o, c_n = 0, 1, 2, 3 # bond type indices
    n_ca_c, ca_c_o, ca_c_n, o_c_n = 0, 1, 2, 3 # bond angle indices
    
    # gathering bond lengths
    bond_lengths[:, n_ca, 0] = coords[:, n_idx]
    bond_lengths[:, n_ca, 1] = coords[:, ca_idx]
    bond_lengths[:, ca_c, 0] = coords[:, ca_idx]
    bond_lengths[:, ca_c, 1] = coords[:, c_idx]
    bond_lengths[:, c_o, 0] = coords[:, c_idx]
    bond_lengths[:, c_o, 1] = coords[:, o_idx]
    bond_lengths[0, c_n, 0] = coords[-1, c_idx]
    bond_lengths[1:, c_n, 0] = coords[:-1, c_idx]
    bond_lengths[:, c_n, 1] = coords[:, n_idx]
    
    # gather bond angles
    bond_angles[:, n_ca_c, 0] = coords[:, n_idx]
    bond_angles[:, n_ca_c, 1] = coords[:, ca_idx]
    bond_angles[:, n_ca_c, 2] = coords[:, c_idx]
    bond_angles[:, ca_c_o, 0] = coords[:, ca_idx]
    bond_angles[:, ca_c_o, 1] = coords[:, c_idx]
    bond_angles[:, ca_c_o, 2] = coords[:, o_idx]
    bond_angles[:, ca_c_n, 0] = coords[:, ca_idx]
    bond_angles[:, ca_c_n, 1] = coords[:, c_idx]
    bond_angles[:-1, ca_c_n, 2] = coords[1:, n_idx]
    bond_angles[-1, ca_c_n, 2] = coords[0, n_idx]
    bond_angles[:, o_c_n, 0] = coords[:, o_idx]
    bond_angles[:, o_c_n, 1] = coords[:, c_idx]
    bond_angles[:-1, o_c_n, 2] = coords[1:, n_idx]
    bond_angles[-1, o_c_n, 2] = coords[0, n_idx]
    
    return bond_lengths, bond_angles

def calculate_bond_lengths(bond_lengths):
    c = bond_lengths[:, :, 0] - bond_lengths[:, :, 1]
    c = np.linalg.norm(c, axis=-1)
    return c

def calculate_angles(angles):
    v = angles[:, :, 0] - angles[:, :, 1]
    v /= np.linalg.norm(v, axis=-1)[...,None]
    w = angles[:, :, 2] - angles[:, :, 1]
    w /= np.linalg.norm(w, axis=-1)[...,None]
    x = np.sum(v*w, axis=-1)
    x = np.arccos(np.clip(x, -1.0, 1.0))
    return x * 180 / np.pi