# Initialization settings

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import numpy as np
import time
import argparse
from scipy.io import savemat
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.cuda.amp import GradScaler, autocast

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

'''
The version of python is  3.10.14 
The version of torch is  2.5.1
The version of numpy is  1.26.4
The version of argparse is  1.1
'''

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "Global HyperParameter")
parser.add_argument('-f', type=str, default = "Read additional parameters")

parser.add_argument('--LogSavedir', default = 'Case I' )
parser.add_argument('--savedir', default = 'Exper. 001 ( MgCF via MgNet )',
                    help = "Experimental result saving address sub-file name")
parser.add_argument('--Logdir', default = os.path.join('../Saved Results'))

parser.add_argument('--SampleLength', default = 2**11 )
parser.add_argument('--DataRoot', default = os.path.join('/Datasets/Bearings with Noisy labels'))
parser.add_argument('--Datasets', default = [ 'Multiple', 'Damage' ] )
parser.add_argument('--NL', default = 'Noisy labels' )
parser.add_argument('--NoiseType', default = { 
    'Symmetric':[ 90, 80, 70, 60, 50 ], 'Asymmetric':[ 45, 40, 35, 30, 25 ] } )

parser.add_argument('--Dims', default = 2**6 )
parser.add_argument('--Granularity', default = 100 )

parser.add_argument('--Epochs', default = [ 5, 100 ] )
parser.add_argument('--InitialLearningRate', default = 1e-3 * 3.0)
parser.add_argument('--BatchSize', default = 2**9 )

parser.add_argument('--device', default = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
parser.add_argument('--num_workers', default = 0)

Args = parser.parse_args()

Args.logdir = os.path.join(Args.Logdir, Args.LogSavedir, Args.savedir)
os.makedirs(Args.logdir, exist_ok=True)

def GetTensorDataset( training, dataname, NoiseType, intensity ):

    '''
    training: 'tes' or 'tra'
    '''

    npy_x = training + '_samples.npy'
    npy_y = training + '_targets.npy'
    
    path_x = os.path.join(os.path.join(Args.DataRoot, 'Clean'), dataname)
    
    if training == 'tra':
        path_y = os.path.join(os.path.join( os.path.join(
            os.path.join(Args.DataRoot, Args.NL), NoiseType), dataname ), str(intensity))
    else:
        path_y = os.path.join(os.path.join(Args.DataRoot, 'Clean'), dataname)

    samples, targets = [], []
    for domain in os.listdir(path_x):
        
        data_x = os.path.join( path_x, domain )
        sample = np.load(os.path.join( data_x, npy_x ))
        
        data_y = os.path.join(path_y, domain)
        target = np.load(os.path.join(data_y, npy_y))
        
        samples.append( torch.as_tensor(sample, dtype = torch.float32) )
        targets.append( torch.as_tensor(target, dtype = torch.int64) )

    dataset = TensorDataset( torch.cat(samples), torch.cat(targets) )
    
    if training == 'tes':
        loader = DataLoader(dataset, batch_size = Args.BatchSize,
                            num_workers = Args.num_workers, shuffle = False)
        return loader
    else:
        total_size = len(dataset)
        val_size = int(total_size * 0.2)
        tra_size = total_size - val_size
        tra_dataset, val_dataset = random_split(dataset, [tra_size, val_size])
        tra_loader = DataLoader(tra_dataset, batch_size = Args.BatchSize,
                                num_workers = Args.num_workers, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size = Args.BatchSize,
                                num_workers = Args.num_workers, shuffle=True)
        
        return tra_loader, val_loader, target.max() + 1

# Model architecture
'''
Diagnostic Framework

@article{deng2023mgnet,
  title={MgNet: A fault diagnosis approach for multi-bearing system 
  based on auxiliary bearing and multi-granularity information fusion},
  author={Deng, Jin and Liu, Han and Fang, Hairui and Shao, Siyu 
  and Wang, Dong and Hou, Yimin and Chen, Dongsheng and Tang, Mingcong},
  journal={Mechanical Systems and Signal Processing},
  volume={193},
  pages={110253},
  year={2023},
  publisher={Elsevier}
}
'''
class Linear(nn.Module):
    def __init__(self, in_features, out_features,
                 norm = None, act = None,
                 zeros_init = False, with_bias = True ):
        super().__init__()
        
        self.Linear = nn.Linear(in_features, out_features)
        
        if zeros_init:
            nn.init.zeros_(self.Linear.weight)
        else:
            nn.init.kaiming_uniform_(self.Linear.weight)
            
        if with_bias:
            nn.init.zeros_(self.Linear.bias)
        
        self.norm = norm
        self.act = act
        
    def forward(self, x):
        
        x = self.Linear(x)
        
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        
        return x

class Conv1D(nn.Module):
    def __init__(self, in_channels, out_channel,
                 kernel = 5, stride = 1, norm = None, act = None):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channel, kernel, stride, padding = int((kernel-1)/2))
            
        nn.init.kaiming_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        
        self.norm = norm
        self.act = act
        
    def forward(self, x):
        
        x = self.conv(x)
        
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        
        return x

class Add(nn.Module):
    def __init__(self, number = 2):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(number, dtype=torch.float32), requires_grad=True)
        
    def forward(self, x):
        w = F.relu(self.weights)
        return x[0]*w[0] + x[1]*w[1]

class ACON(nn.Module):
    def __init__(self):
        super().__init__()
        self.α = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        
    def forward(self, x):
        α = F.relu(self.α)
        x = α[0] * x * torch.sigmoid(x * α[1])
        return x

class Mg(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Sequential(
                Conv1D( d_in, d_out//4, kernel = 2**(k+2)+1, stride = 2,
                       norm = nn.BatchNorm1d(d_out//4), act = ACON() ),  
                Conv1D( d_out//4, d_out//4, kernel = 2**(k+2)+1, stride = 2,
                       norm = nn.BatchNorm1d(d_out//4), act = ACON() )
                ) for k in range(4)
        ])
        
        self.add = Add()
        
        self.idx = nn.Sequential( Conv1D(
            d_in, d_out, kernel = 1, stride = 1, 
            norm = nn.BatchNorm1d(d_out), act = ACON() ),
                                 nn.MaxPool1d(4, 4) )
        
        self.point = Conv1D( d_out, d_out, kernel = 1, stride = 1,
                            norm = nn.BatchNorm1d(d_out), act = ACON() )
    
    def forward(self, x):
        
        i = self.idx(x)
        
        xes = []
        for conv in self.convs:
            xes.append(conv(x))
        x = torch.cat(xes, dim = 1)
        
        x = self.point(x)
        
        x = self.add([x, i])
        
        return x

class MgNet_backbone(nn.Module):
    def __init__( self, dims, Stages = [4, 16, 64, 128] ):
        super().__init__()
        
        '''
        Stages: In the original MgNet experiment, the Stages was set to [4, 16, 64, 128]
        and the default single channel data input, which can be modified according to actual needs.
        '''
        
        self.stages = nn.Sequential(
            Mg(1, Stages[0]), Mg(Stages[0], Stages[1]),
            Mg(Stages[1], Stages[2]), Mg(Stages[2], Stages[3])
            )
        
        self.weights = nn.Parameter(torch.as_tensor([1, 0], dtype=torch.float32), requires_grad=True)
        self.projector = Linear( Stages[3], dims )
        
    def forward(self, x):

        weights = F.relu( self.weights )
        z = self.stages(x)
        
        return self.projector( z.mean(2) * weights[0] + z.max(2)[0] * weights[1] )

class Get_Model(nn.Module):
    def __init__(self, categories, dims = Args.Dims ):
        super().__init__()
        
        self.backbone =  MgNet_backbone( dims )
        self.predictor = Linear( dims, categories, zeros_init = True )
        
    def forward(self, x):
        
        z = self.backbone(x)
        prediction = self.predictor( z )
        
        return prediction

# Model training

def smooth_one_hot(true_labels, categories, smoothing = 0.3):
    confidence = smoothing/(categories-1)
    true_labels = F.one_hot(true_labels, categories)
    with torch.no_grad():
        smooth_label = true_labels * (1.0 - smoothing - confidence) + confidence
    return smooth_label

def GetAccuracy( model, loader, categories ):
    
    model.eval().to(Args.device)
    criterion = nn.CrossEntropyLoss()
    criterion.to(Args.device).eval()
    
    prefetcher = iter(loader)
    
    count, correct, losses = 0, 0, []
    
    for _ in range(len(loader)):
        sample, target = next(prefetcher)
        count += target.shape[0]
        target = target.to(Args.device)

        y = F.one_hot(target, categories).float()
        x = sample.to(Args.device)
        
        with torch.no_grad():
            prediction = model(x)
            
        loss = criterion( prediction, y )
        losses.append( loss.item()  )
        correct += torch.eq(prediction.argmax(dim=1), target).sum().float().item()
        
    return correct/count * 100, np.mean(losses)

def GradientUpdate( model, loader, optimizer, scheduler, categories ):
    '''
    Supervised learning process (Warm-up)
    '''
    model.train()
    model.to(Args.device)
    criterion = nn.CrossEntropyLoss()
    criterion.eval()
    criterion.to(Args.device)

    count, correct, losses = 0, 0, []
    scaler = GradScaler()
    prefetcher = iter(loader)

    for _ in range(len(loader)):
        
        sample, target = next(prefetcher)
        count += target.shape[0]
        
        y = smooth_one_hot( target.to(Args.device), categories )
        x = sample.to(Args.device) 
        
        with autocast():
            prediction = model( x )
            loss = criterion(prediction, y)
                
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append( loss.item()  )
        
        optimizer.zero_grad()
        scheduler.step()
        
        correct += torch.eq( prediction.argmax(dim=1), y.argmax(dim=1) ).sum().float().item()
        
    return correct/count * 100, np.mean(losses)

def CorrectionLoader( model, loader, categories, intensity ):
    
    model.eval()
    model.to(Args.device)
    
    samples, targets, latents = [], [], []
    for sample, target in loader:
        x = sample.to(Args.device)
        y = F.one_hot( target.to(Args.device), num_classes=categories).float()
        
        with torch.no_grad():
            z = F.normalize( model.backbone(x) )
            
        samples.append( x )
        targets.append( y )
        latents.append( z )
        
    samples = torch.cat(samples, 0).to('cpu')
    targets = torch.cat(targets, 0)
    latents = torch.cat(latents, 0)

    granularity = max( int( intensity * Args.Granularity ), 10 )
    
    datasets, Intensity = [], []
    similarity = torch.einsum('bl, gl -> bg', latents, latents ).exp()
    similarity.fill_diagonal_(0)
    weight, index = torch.topk( similarity, granularity, dim=1 )
    
    SgClabels = (targets[index] * weight.unsqueeze(dim = 2)).sum(1) / weight.sum(1, keepdim = True)
    
    purity, pseudo = SgClabels.max(dim = 1)
    pseudo = F.one_hot( pseudo, num_classes=categories).float()
    intensity = 1 - purity.mean().item()
    
    belief = torch.einsum('bc, bc -> b', SgClabels, targets )
    certainty = belief * ( belief/purity )
    uncertainty = 1 - certainty
    
    Mg_targets = torch.einsum('bc, b -> bc', targets, certainty) + torch.einsum('bc, b -> bc', pseudo, uncertainty) 
    
    loader = DataLoader(
        TensorDataset( samples, Mg_targets.to('cpu') ),
        batch_size = Args.BatchSize, num_workers = Args.num_workers, pin_memory=True, shuffle = True )
    
    return loader, intensity

def GradientUpdateViaCorrectedLabels( model, loader, optimizer, scheduler ):
    
    model.train()
    model.to(Args.device)
    criterion = nn.CrossEntropyLoss()
    criterion.eval()
    criterion.to(Args.device)

    count, correct, losses = 0, 0, []
    scaler = GradScaler()
    
    for sample, target in loader:
        count += target.shape[0]
        
        y = target.to(Args.device)
        x = sample.to(Args.device) 
        
        with autocast():
            p = model(x)
            loss = criterion(p, y)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append( loss.item()  )
        
        optimizer.zero_grad()
        scheduler.step()
        
        correct += torch.eq( p.argmax(dim=1), y.argmax(dim=1) ).sum().float().item()
        
    return correct/count * 100, np.mean(losses)

def GetModelWithWeights( save_root, tra_loader, val_loader, categories ):
    
    training_net = os.path.join( Args.logdir, 'Training net.pt' )
    training_log = os.path.join( save_root, 'Training logs.mat' )
    
    Trained_net = os.path.join( save_root, 'Trained net.pt' )
    if not os.path.exists(training_log):
        model = Get_Model( categories )
        model.to(Args.device)

        Acc, ValAcc, Loss, ValLoss, Intensity, Max_acc = [], [], [], [], [], 0.
        intensity = 0.
        
        for e in range(len(Args.Epochs)):

            epoch = Args.Epochs[e]
            LearningRate = Args.InitialLearningRate
            optimizer = torch.optim.AdamW( model.parameters(), LearningRate )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( 
                optimizer, T_max = epoch * len(tra_loader) + 1 ) 
            
            pbar = tqdm( range(epoch) )
            for _ in pbar:
                if e==0:
                    acc, loss = GradientUpdate( model, tra_loader, optimizer, scheduler, categories )
                else:
                    Mg_Loader, intensity = CorrectionLoader( model, tra_loader, categories, intensity )
                    acc, loss = GradientUpdateViaCorrectedLabels( model, Mg_Loader, optimizer, scheduler )
                    
                val, val_loss = GetAccuracy( model, val_loader, categories )
                
                if val>Max_acc:
                    Max_acc = val
                    torch.save(model.state_dict(), training_net)
                    
                pbar.set_description( 
                    r'{} times training: with {:.2f}% of val acc, {:.2f}% of acc,\
                    {:.3f} of loss, {:.3f} of val loss {:.3f} of intensity.'.format(
                    e + 1, val, acc,  loss, val_loss, intensity * 100
                )
                                     )
                                     
                Acc.append(acc)
                Loss.append(loss)
                
                ValAcc.append(val)
                ValLoss.append(val_loss)
                
                Intensity.append(intensity * 100)
                
            model.load_state_dict( torch.load(training_net) )
            
        log_results = {}
        
        log_results['Accuracy'] = Acc
        log_results['ValAccuracy'] = ValAcc
        log_results['Loss'] = Loss
        log_results['ValLoss'] = ValLoss
        log_results['Intensity'] = Intensity
        
        savemat( training_log, log_results )
        torch.save(model.state_dict(), Trained_net)
        
    else:
        model = Get_Model( categories )
        model.load_state_dict( torch.load(Trained_net) )
        model.to(Args.device)
        
    return model

def Training( training_times = 10 ):
    
    test_results = os.path.join( Args.logdir, 'Test results.txt')
    
    if not os.path.exists(test_results):
        Header = '{}\t{}\t{}\t{}\t{}\t{}\t\n'.format(
            'Dataset', 'Noise type', 'Noise intensity', 'Times', 'Accuracy(%)', 'Loss' )
        with open(test_results, 'a+') as log:
            log.write(Header)

    for dataname in Args.Datasets:
        
        for nt in Args.NoiseType.keys():
            print()
            print('====================================================================')
            for intensity in Args.NoiseType[nt]:
                print(nt, ' noisy        ', intensity, '%')
                
                for times in range(1, training_times + 1):
                    
                    f = open(test_results, encoding='gbk')
                    Results=[]
                    for line in f:
                        Results.append(line.strip().split('\t'))
                        
                    IfResult = [
                        result for result in Results 
                        if result[0] == dataname
                        and result[1] == nt
                        and result[2] == str(intensity)
                        and result[3] == str(times)
                    ]
                    
                    if len(IfResult) > 0:
                        for results in IfResult:
                            print( r'On {} dataset with {} noise of {}% {}-times:\
                            accuracy of {:.3f}% with loss of {:.4f}.'.format(
                                results[0], results[1], results[2], results[3],
                                float(results[4]), float(results[5])
                                )
                            )
                    else:
                        save_root = os.path.join(Args.logdir, dataname, nt, str(intensity), f'{times} times')
                        os.makedirs(save_root, exist_ok=True)
                        
                        tra_loader, val_loader, categories = GetTensorDataset( 'tra', dataname, nt, intensity )
                        tes_loader = GetTensorDataset( 'tes', dataname, nt, intensity )
                        
                        model = GetModelWithWeights( save_root, tra_loader, val_loader, categories )
                        acc, loss = GetAccuracy( model, tes_loader, categories )
                        
                        results = '{}\t{}\t{}\t{}\t{}\t{}\t\n'.format(
                            dataname, nt, intensity, times, acc, loss )
                        with open(test_results, 'a+') as log:
                            log.write(results)
                            
                        results = results.split('\t')
                        print(r'On {} dataset with {} noise of {}% {}-times:\
                        accuracy of {:.3f}% with loss of {:.4f}.'.format(
                            results[0], results[1], results[2], results[3],
                            float(results[4]), float(results[5]) )
                            )
                        
                print()

# Display experimental results

Training(  )