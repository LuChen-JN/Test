from time import time
from torch.utils.data import DataLoader
import torch
import argparse
import warnings, os
import pandas as pd
import logging
logging.disable(logging.WARNING)
from model_glu import *
from utils import graph_collate_func
from dataloader import NTIDataset
from model_glu import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:1')
parser = argparse.ArgumentParser(description="FNA-SpeEvo for NTI prediction")
parser.add_argument('--data', default="fold_1", type=str, metavar='TASK',
                    help='dataset')
args = parser.parse_args()

def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    dataFolder = f'./datasets/{args.data}'

    train_path = os.path.join(dataFolder, 'train.csv')
    val_path=os.path.join(dataFolder, 'val.csv')
    test_path=os.path.join(dataFolder, 'test.csv')

    df_train = pd.read_csv(train_path)
    df_val=pd.read_csv(val_path)
    df_test=pd.read_csv(test_path)

    train_dataset = NTIDataset(df_train.index.values, df_train)
    val_dataset= NTIDataset(df_val.index.values,df_val)
    test_dataset= NTIDataset(df_test.index.values,df_test)

    train_length=len(train_dataset)

    dataset_train=DataLoader(train_dataset,shuffle=True,collate_fn= graph_collate_func,batch_size=64,num_workers=0,drop_last=True)  # True/False
    dataset_dev=DataLoader(val_dataset,shuffle=True,collate_fn= graph_collate_func,batch_size=64,num_workers=0,drop_last=True)      # True/False
    dataset_test=DataLoader(test_dataset,shuffle=True,collate_fn= graph_collate_func,batch_size=64,num_workers=0,drop_last=True)    # True/False
    
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
    lr_decay = 1
    iteration = 100
    kernel_size = 7

    encoder = Encoder(protein_dim, hid_dim, 3, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)
    
    model.to(device)
    trainer = Trainer(model, lr, weight_decay, batch,device)
    tester = Tester(model,device,batch)

    file_model = 'model.pt'

    file_train = 'train_results.txt'
    file_dev = 'dev_results.txt'
    file_test = 'test_results.txt'

    train_headers = ['Epoch', 'Loss_train']
    dev_headers = ['Epoch', 'AUROC_dev', 'Precision', 'AUPRC_dev', 'Val_loss']
    test_headers = ['Epoch', 'AUROC_test', 'Precision', 'AUPRC_test', 'F1_test', 'Sensitivity_test', 'Specificity_test', 'Accuracy_test', 'Threshold_test', 'Test_loss']
    
    with open(file_train, 'w') as f:
        f.write('\t'.join(train_headers) + '\n')

    with open(file_dev, 'w') as f:
        f.write('\t'.join(dev_headers) + '\n')

    with open(file_test, 'w') as f:
        f.write('\t'.join(test_headers) + '\n')
    
    print('Training...')
    max_AUC_test = 0
    epoch_label = 0

    for epoch in range(1, iteration+1):
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] -= 5e-5

        Loss_train = trainer.train(dataset_train, train_length)
        print(f"epoch{epoch} Finish")
        AUROC_dev, Precision,AUPRC_dev, _,_,_,_,_,Val_loss = tester.test(dataset_dev)
        print(f"epoch{epoch} Finish")
        AUROC_test,Precision, AUPRC_test, F1_test, Sensitivity_test, Specificity_test, Accuracy_test, Threshold_test,  Test_loss  = tester.test(dataset_test)
        print(f"epoch{epoch} Finish")

        # Save the results of the training set
        train_metrics = [epoch,  Loss_train]
        tester.save_metrics(train_metrics, file_train)

        # Save the results of the validation set
        dev_metrics = [epoch, AUROC_dev,Precision, AUPRC_dev, Val_loss]
        tester.save_metrics(dev_metrics, file_dev)

        # Save the results of the test set
        test_metrics = [epoch, AUROC_test,Precision, AUPRC_test, F1_test, Sensitivity_test, Specificity_test, Accuracy_test, Threshold_test, Test_loss]
        tester.save_metrics(test_metrics, file_test)

        if AUROC_test > max_AUC_test:
            tester.save_model(model, file_model)
            max_AUC_test = AUROC_test
            epoch_label = epoch
    
    print(f"all right!The best epoch:{epoch_label}")

if __name__ == '__main__':
    main()


