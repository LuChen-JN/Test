import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from dgllife.model.gnn import GCN
import sys
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, f1_score

picture = False
maxtrix = None

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)

        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len_Q, hid dim]

        return x


class Encoder(nn.Module):
    """DNA feature extraction."""
    def __init__(self, dna_dim, hid_dim, n_layers,kernel_size , dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = dna_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim,self.hid_dim)

    def forward(self, dna):
        conv_input = self.fc(dna)

        conv_input = conv_input.permute(0, 2, 1)

        for i, conv in enumerate(self.convs):

            conved = conv(self.dropout(conv_input))

            conved = F.glu(conved, dim=1)

            conved = (conved + conv_input) * self.scale

            conv_input = conved

        conved = conved.permute(0,2,1)

        return conved


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.do(F.relu(self.fc_1(x)))

        x = self.fc_2(x)

        x = x.permute(0, 2, 1)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg


class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)

    def forward(self, trg, src, trg_mask=None,src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        trg = self.ft(trg)
        attention_weights_list = []
        # trg = [batch size, compound len, hid dim]
        for layer in self.layers:
            trg = layer(trg, src)
            trg1 = trg[0, :, :]
            attention_weights_list.append(trg1.squeeze(0))
        global picture
        if picture:
            global maxtrix
            maxtrix = attention_weights_list[-1]
        # trg = [batch size, compound len, hid dim]
        """Use norm to determine which atom is significant. """
        norm = torch.norm(trg, dim=2)
        norm = F.softmax(norm, dim=1)
        sum = torch.zeros((trg.size(0), self.hid_dim)).to(self.device)
        for i in range(norm.size(0)):
            for j in range(norm.size(1)):
                v = trg[i, j,]
                v = v * norm[i, j]
                sum[i,] += v

        label = F.relu(self.fc_1(sum))  # [batch size, fc1_out_dim]
        label = self.fc_2(label)  # [batch size, output_dim]
        return label

class Predictor(nn.Module):
    def __init__(self, encoder, decoder, device, atom_dim=128):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.mole_extractor = MolecularGCN(in_feats=74, dim_embedding=128,
                                           padding=False,
                                           hidden_feats=[128, 128,128])
        self.dna_extractor = DNACNN(128,[128, 128, 128], [3, 6, 9], True)

    def forward(self, graph, dna):

        compound = self.mole_extractor(graph)
        dna = self.dna_extractor(dna)

        enc_src = self.encoder(dna)

        out = self.decoder(compound, enc_src)

        return out

    def __call__(self, temp1,temp2,temp3 ,train=True):

        graph=temp1
        dna=temp2
        correct_interaction=temp3
        Loss = nn.CrossEntropyLoss()
        if train:
            correct_interaction = correct_interaction.long()
            predicted_interaction = self.forward(graph, dna)
            loss = Loss(predicted_interaction, correct_interaction)
            return loss

        else:
            correct_interaction = correct_interaction.long()
            predicted_interaction = self.forward(graph, dna)
            loss = Loss(predicted_interaction, correct_interaction)
            correct_labels = correct_interaction.to('cpu').numpy()
            ys = F.softmax(predicted_interaction, dim=1).to('cpu').numpy()
            predicted_labels = np.argmax(ys, axis=1)  
            predicted_scores = ys[:, 1]  
            return correct_labels, predicted_labels, predicted_scores, loss


class Predictor2(nn.Module):
    def __init__(self, encoder, decoder, device, atom_dim=128):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.mole_extractor = MolecularGCN(in_feats=74, dim_embedding=128,
                                           padding=False,
                                           hidden_feats=[128, 128, 128])
        self.dna_extractor = DNACNN(128, [128, 128, 128], [3, 6, 9], True)

    def forward(self, graph, dna):
        compound = self.mole_extractor(graph)
        dna = self.dna_extractor(dna)

        enc_src = self.encoder(dna)
        # enc_src = [batch size, protein len, hid dim]
        out = self.decoder(compound, enc_src)
        # out = [batch size, 2]
        # out = torch.squeeze(out, dim=0)

        return out

    def __call__(self, temp1, temp2, temp3=None, train=True):
        graph = temp1
        dna = temp2
        predicted_interaction = self.forward(graph, dna)
        ys = F.softmax(predicted_interaction, dim=1).to('cpu').numpy()
        predicted_labels = np.argmax(ys, axis=1)
        return predicted_labels


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=False, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class DNACNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(DNACNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(5, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(5, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.bn1 = nn.BatchNorm1d(in_ch[1])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(v)
        v = v.view(v.size(0), v.size(2), -1)
        return v


class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch, device):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)
        torch.backends.cudnn.benchmark = True 
        self.batch = batch
        self.device = device

    def train(self, dataset, N):
        self.model.train()  # train
        loss_total = 0
        num_batches = len(dataset)
        self.optimizer.zero_grad()

        for idx, (temp1, temp2, temp3) in enumerate(dataset):
            temp1 = temp1.to(self.device)
            temp2 = temp2.to(self.device)
            temp3 = temp3.to(self.device)

            loss = self.model(temp1, temp2, temp3)  # loss

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_total += loss.item()

            print(f"{idx}Finish")

        loss_total /= num_batches
        return loss_total



class Tester(object):
    def __init__(self, model, device, batch):
        self.model = model
        self.device = device
        self.batch = batch

    def test(self, dataset):
        total_loss = 0
        num_batches = len(dataset)
        self.model.eval()
        T, Y, S = [], [], []
        with torch.no_grad():
            for idx, (temp1, temp2, temp3) in enumerate(dataset):
                temp1 = temp1.to(self.device)
                temp2 = temp2.to(self.device)
                temp3 = temp3.to(self.device)
                correct_labels, predicted_labels, predicted_scores, loss = self.model(temp1, temp2, temp3, train=False)

                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)

                total_loss += loss.item()

        total_loss /= num_batches

        # **1. AUROC**
        auroc = roc_auc_score(T, S)

        # **2. Precision **
        precision_1 = precision_score(T, Y)

        # **3. AUPRC **
        auprc = average_precision_score(T, S)

        # **4. F1-Score**
        f1 = f1_score(T, Y)

        # **5. Sensitivity **
        cm = confusion_matrix(T, Y)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0

        # **6. Specificity **
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        # **7. Accuracy **
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # **8. Optimal Threshold **
        fpr, tpr, thresholds = roc_curve(T, S)
        f1_scores = 2 * tpr * (1 - fpr) / (tpr + (1 - fpr) + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]

        return auroc, precision_1, auprc, f1, sensitivity, specificity, accuracy, optimal_threshold, total_loss
    
    
    def save_metrics(self, metric, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, metric)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


