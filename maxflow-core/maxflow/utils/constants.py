# Standard Organic Atom Types (SOTA Standard: Expanded to all 118 elements for Flagship compatibility)
ALLOWED_ATOM_TYPES = [6, 7, 8, 16, 15, 9, 17, 35, 53] # Main organic symbols for classification
ATOM_SYM_TO_IDX = {6: 0, 7: 1, 8: 2, 16: 3, 15: 4, 9: 5, 17: 6, 35: 7, 53: 8}
IDX_TO_ATOM_SYM = {v: k for k, v in ATOM_SYM_TO_IDX.items()}
NUM_ATOM_TYPES = len(ALLOWED_ATOM_TYPES)

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)), # 118 elements
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_num_h_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
}

# Pharmacophore Features (RDKit)
PHARMACOPHORE_FAMILIES = [
    'Donor', 
    'Acceptor', 
    'Aromatic', 
    'Hydrophobe', 
    'PosIonizable', 
    'NegIonizable'
]

# Standard Amino Acids
amino_acids = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
