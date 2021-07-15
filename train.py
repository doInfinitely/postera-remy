import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
import pandas as pd
from rdkit.Chem import AllChem
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# TODO: Hyper-parameter search
SPLIT_FRAC = 0.7
DEEP_MUSCLE = 1
ADJ_OR_DIST = 'DIST'
KINASE_NAME = 'ALL'
MEASUREMENT_TYPE = 'ALL'
LEARNING_RATE = 1e-12
MOMENTUM = 0.9

#atoms = ['C','H','O']
class Predicate(nn.Module):
    def __init__(self,a,b,atomSetList):
        super().__init__()
        self.lambd = lambda x: x[1] in x[0] and x[2] in x[0]
        self.atoms = atomSetList
        atoms = self.atoms
        self.bond_vector = [({x,y}, a, b) for x in atoms for y in atoms]
    def forward(self, x):
        mask_vector = [int(self.lambd(y)) for y in self.bond_vector]
        atoms = self.atoms
        mask = torch.from_numpy(np.array(mask_vector).reshape((len(atoms),len(atoms))))
        return x*mask
class Sampler(nn.Module):
    def __init__(self, atoms):
        super().__init__()
        self.weights = torch.normal(torch.tensor(np.array([0.0 for i in range(len(atoms)**2)])))
    def forward(self,x):
        atoms = x[0]
        dist = x[1]
        adj = x[2]
        predicates = [Predicate(atoms[i],atoms[j],atoms) for i in range(len(atoms)) for j in range(len(atoms))]
        softmax = nn.Softmax(dim=0)(self.weights)
        if self.training:
            if ADJ_OR_DIST == 'DIST':
                output = torch.sum(torch.stack([y.forward(dist)*softmax[j] for j,y in enumerate(predicates)]))
            elif ADJ_OR_DIST == 'ADJ':
                output = torch.sum(torch.stack([y.forward(adj)*softmax[j] for j,y in enumerate(predicates)]))
            return output
        sample = torch.utils.data.WeightedRandomSampler(nn.Softmax(dim=0)(self.weights), 1) 
        if ADJ_OR_DIST == 'DIST':
            output = torch.sum(torch.stack([y.forward(dist) for j,y in enumerate(predicates)]))
        elif ADJ_OR_DIST == 'ADJ':
            output = torch.sum(torch.stack([y.forward(adj) for j,y in enumerate(predicates)]))
        return output[sample]

#sampler = Sampler(atoms)
#x = torch.tensor(np.array([[[0,1,0],[1,0,1],[0,1,0]]]))
#print(sampler.forward(x))

class Prefix(nn.Module):
    def __init__(self, atoms, channels):
        super().__init__()
        self.atoms = atoms
        self.components = [Sampler(atoms) for i in range(channels)]
    def prep_mask(self, atoms):
        for x in self.components:
            self.atoms = atoms
    def forward(self,x):
        return torch.tensor(np.array([y.forward(x) for y in self.components], dtype=np.double))

class MyDataset(Dataset):
    def __init__(self, csv='./kinase_JAK.csv'):
        self.atomSet = set()
        self.maxAtoms = 0
        self.X = []
        self.Y = []
        self.Z = []
        self.W = []
        self.df = pd.read_csv(csv)
        #self.df.head()
        for i,x in enumerate(self.df['SMILES']):
            if KINASE_NAME == 'ALL' or self.df['Kinase_name'][i] == KINASE_NAME:
                if MEASUREMENT_TYPE == 'ALL' or self.df['measurement_type'] == MEASUREMENT_TYPE:
                    mol = Chem.MolFromSmiles(x)
                    mol = Chem.rdmolops.AddHs(mol)
                    AllChem.EmbedMolecule(mol)                    
                    dist = AllChem.Get3DDistanceMatrix(mol)
                    adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
                    num = mol.GetNumAtoms()
                    if num > self.maxAtoms:
                        self.maxAtoms = num
                    atoms = [mol.GetAtomWithIdx(j).GetSymbol() for j in range(mol.GetNumAtoms())]
                    self.atomSet.update(atoms)
                    model = Prefix(atoms, num)
                    self.X.append(dist)
                    self.W.append(adj)
                    self.Y.append(self.df['measurement_value'][i])
                    self.Z.append(atoms)
            if i > 1:
                break
    def pad(self, x):
        shape = np.shape(x)
        padded_array = np.zeros((self.maxAtoms, self.maxAtoms))
        padded_array[:shape[0],:shape[1]] = x
        return padded_array
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        X = self.X[idx]
        Z = [x for x in self.Z[idx]]
        Z += ['' for i in range(self.maxAtoms-len(Z))]
        X = torch.from_numpy(self.pad(X))
        W = self.W[idx]
        W = torch.from_numpy(self.pad(W))
        return (Z,X,W), self.Y[idx]

dataset = MyDataset()
atoms = dataset[0][0][0]
maxAtoms = dataset.maxAtoms
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*SPLIT_FRAC),len(dataset)-int(len(dataset)*SPLIT_FRAC)])

train_dataloader = DataLoader(train_dataset,batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

prefix = Prefix(atoms,maxAtoms)
model = torch.nn.Sequential(
    prefix,
)

'''
model = torch.nn.Sequential(
    Prefix(atoms,maxAtoms),
    nn.Linear(maxAtoms, maxAtoms).double(),
    nn.ReLU(),
    nn.Linear(maxAtoms,1).double()
)
'''
for i in range(DEEP_MUSCLE):
    model.add_module('linear_{}'.format(i),nn.Linear(maxAtoms, maxAtoms).double())
    model.add_module('ReLU_{}'.format(i),nn.ReLU())
model.add_module('output_layer!', nn.Linear(maxAtoms,1).double())
#print(model.forward(dist).type())

optim = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
'''
for i in range(5):
    loss = nn.MSELoss()
    temp = model.forward(dist)
    print(temp)
    output = loss(temp,torch.tensor(1.0).double())
    output.backward()
    optim.step()
'''
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f'Avg loss {test_loss:>8f} \n')

train_loop(train_dataloader, model, nn.MSELoss(), optim)
test_loop(test_dataloader, model, nn.MSELoss())
