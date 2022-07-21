# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:41:31 2022

@author: Study
"""

import dgl
from collections import defaultdict
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from rdkit import Chem
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
random_state=34

# embed molecule with GNN
def collate(gs):
    return dgl.batch(gs)   

def smi2infomax(smiles):
    '''
    using pre-trained GNN to calculate 2D fingerprint

    Parameters
    ----------
    smiles : pd.series
        example: smiles = clean['CanonicalSMILES']
                df = smi2infomax(smiles)
    Returns
    -------
    pd.DataFrame
        n X 300 dataframe
        you need to be careful about the failed calculation,
        in case there is a mis match between output df and smiles list.

    '''
    model = load_pretrained('gin_supervised_infomax')
    ## generate molecular graphs
    graphs = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            g = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=PretrainAtomFeaturizer(),
                               edge_featurizer=PretrainBondFeaturizer(),
                               canonical_atom_order=True)
            graphs.append(g)
    
        except:
            print(str(smi)+ ' wrong')
            exit
        
    data_loader = DataLoader(graphs, batch_size=256, collate_fn=collate, shuffle=False)
    readout = AvgPooling()
    mol_emb = []
    for batch_id, bg in enumerate(data_loader):
        nfeats = [bg.ndata.pop('atomic_number').to('cpu'),
                  bg.ndata.pop('chirality_type').to('cpu')]
        efeats = [bg.edata.pop('bond_type').to('cpu'),
                  bg.edata.pop('bond_direction_type').to('cpu')]
        with torch.no_grad():
            node_repr = model(bg, nfeats, efeats)
        mol_emb.append(readout(bg, node_repr))
    mol_emb = torch.cat(mol_emb, dim=0).detach().cpu().numpy()
    ## Freezed fingerprints: could be retrained in future model
    return pd.DataFrame(data=mol_emb)




from autogluon.tabular import  TabularPredictor
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict

def train( y, X, train_index, test_index, name, path, load=False,eval_metric=None):
    '''
    Use autogluon to train and return perf matrix and save the model at path
    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.
    train_index : TYPE
        DESCRIPTION.
    test_index : TYPE
        DESCRIPTION.
    name : TYPE
        name of y column in the dataframe
    path : TYPE
        path to save model
    load : TYPE, optional
        whether load a model. The default is False.
    eval_metric : TYPE, optional
        choose metric to optimize on. The default is None.

    Returns
    -------
    perf : TYPE
        performace matrix.

    '''
    df = pd.concat([y,X],axis=1)
    df_train = df.loc[train_index]
    df_test = df.loc[test_index]
    y_test = df_test[name]
    X_test = df_test.drop(columns=name)
    if load:
        predictor = TabularPredictor.load(name)
    else:
        # delete ag_args_fit option if you don't use GPU
        predictor = TabularPredictor(label=name,path=path,eval_metric=eval_metric).fit(df_train,ag_args_fit={'num_gpus': 1})
    results = predictor.fit_summary(show_plot=False)
    y_pred = predictor.predict_proba(X_test)
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    return perf

def split(length,ratio):
    index = np.arange(length)
    np.random.shuffle(index)
    train_index = index[0:int(length*ratio)]
    test_index = index[int(length*ratio):]
    return list(train_index), list(test_index)





