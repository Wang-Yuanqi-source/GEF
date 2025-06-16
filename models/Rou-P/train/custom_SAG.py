import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn import GATConv, JumpingKnowledge, global_mean_pool, SAGEConv, GCNConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import logging
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch.nn as nn
from torch.nn import Linear, ModuleList, BatchNorm1d
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.nn.pool.connect.filter_edges import filter_adj
from torch.nn import Parameter
import torch_geometric

# Assuming GNNDataset_criticalpath is already defined as per your provided script
from routability_gnn import GNNDataset_criticalpath
from torch_geometric.utils import degree


class SAGPool(nn.Module):
    def __init__(self, in_channels, ratio=0.8, Conv=GCNConv, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity
        
        # MLP for calculating attention scores from concatenated node and neighbor features
        self.attention_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        row, col = edge_index 

        src_node_features = x[row]   
        dst_node_features = x[col]   
        concat_features = torch.cat([src_node_features, dst_node_features], dim=1) 

        edge_scores = self.attention_mlp(concat_features).squeeze()  

        node_scores = torch.zeros(x.size(0), device=x.device)
        node_scores = node_scores.index_add_(0, row, edge_scores) 
        node_degree = degree(row, x.size(0), dtype=x.dtype)
        node_scores = node_scores / (node_degree + 1e-6) 

        adjusted_score = node_scores * torch.sqrt(node_degree)

        perm = topk(adjusted_score, self.ratio, batch)

        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=x.size(0))

        x = x[perm] * self.non_linearity(adjusted_score[perm]).view(-1, 1)
        batch = batch[perm]

        return x, edge_index, edge_attr, batch, perm

class SAGE_JK(torch.nn.Module):
    def __init__(self, num_node_features, conv_neurons, conv_type='GAT', num_layers=3):
        super(SAGE_JK, self).__init__()

        jk_mode = 'cat'
        conv_dict = {
            'GCN': GCNConv,
            'GAT': GATConv,
            'SAGE': SAGEConv
        }

        self.convs = ModuleList()
        self.bns = ModuleList()

        for i in range(num_layers):
            in_channels = num_node_features if i == 0 else conv_neurons
            self.convs.append(conv_dict[conv_type](in_channels, conv_neurons))
            self.bns.append(BatchNorm1d(conv_neurons))

        self.jk = JumpingKnowledge(mode=jk_mode, channels=conv_neurons, num_layers=num_layers)

        concatenated_feature_size = conv_neurons * num_layers if jk_mode == 'cat' else conv_neurons
        self.lin1 = torch.nn.Linear(concatenated_feature_size, conv_neurons * 2)
        self.lin2 = torch.nn.Linear(conv_neurons * 2, conv_neurons // 2)
        self.lin3 = torch.nn.Linear(conv_neurons // 2, 1)

        self.reg_lin1 = torch.nn.Linear(272 + concatenated_feature_size + 1, 32)  # 加上170和池化后的特征
        self.reg_lin2 = torch.nn.Linear(32, 2)  # 回归输出为1维


    def forward(self, data):  
        x, edge_index, batch = data.x, data.edge_index, data.batch
        layer_features = []
        
        delay_param = data.x[:, 0]
        # print(delay_param)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            layer_features.append(x)
        
        x = self.jk(layer_features)

        # Classification output
        node_output = F.relu(self.lin1(x))
        node_output = F.relu(self.lin2(node_output))
        node_output = self.lin3(node_output)

        node_delay = node_output * delay_param.view(-1, 1)

        x_class = global_mean_pool(node_delay, batch)
        x_pool = global_mean_pool(x, batch)

        # Regression output
        print(data.other_attrs.shape)
        other_attrs = data.other_attrs.view(-1, 273) 

        # print("Shape of other_attrs:", other_attrs.shape)
        reg_input = other_attrs[:, :272]  # Shape: [batch_size, 170]

        # print("Shape of reg_input:", reg_input.shape)  # [batch_size, 170]
        # print("Shape of x_class:", x_class.shape)  
        # print("Shape of x_pool:", x_pool.shape)  

        last_attr = other_attrs[:, -1].unsqueeze(1)  # Shape: [batch_size, 1]

        reg_input = torch.cat((reg_input, x_class, x_pool), dim=1)  
        print()

        reg_output = F.relu(self.reg_lin1(reg_input))
        reg_output = self.reg_lin2(reg_output)

        return node_output, reg_output, last_attr 

class ModelC(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2, num_conv_layers=3, other_attrs_dim=272):
        super(ModelC, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.conv_layers = nn.ModuleList()
        for i in range(num_conv_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            self.conv_layers.append(GCNConv(in_channels, hidden_dim))

        self.mlp_input_dim = hidden_dim * 3 + other_attrs_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        readouts = []

        for conv in self.conv_layers:
            x = F.relu(conv(x, edge_index))

            layer_readout = global_mean_pool(x, batch)
            readouts.append(layer_readout)

        avg_readout = torch.mean(torch.stack(readouts), dim=0) 
        final_mean = global_mean_pool(x, batch)  
        final_max = global_max_pool(x, batch)   

        x = torch.cat([avg_readout, final_mean, final_max], dim=1)
        
        other_attrs = data.other_attrs.view(-1, 272)
        x = torch.cat([x, other_attrs], dim=1)
        
        return self.mlp(x)

class ModelB(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2, num_conv_layers=3, other_attrs_dim=272):

        super(ModelB, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        for i in range(num_conv_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            self.conv_layers.append(GCNConv(in_channels, hidden_dim))
            self.pool_layers.append(SAGPool(hidden_dim, 0.5))
        
        self.mlp_input_dim = hidden_dim * 3 + other_attrs_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        readouts = [] 
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = F.relu(conv(x, edge_index))
            x, edge_index, edge_attr, batch, perm = pool(x, edge_index, batch=batch)
            
            # calculate and save readout
            layer_mean = global_mean_pool(x, batch)
            readouts.append(layer_mean)

        avg_readout = torch.mean(torch.stack(readouts), dim=0)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        x = torch.cat([avg_readout, x_mean, x_max], dim=1)

        other_attrs = data.other_attrs.view(-1, 272) 
        x = torch.cat([x, other_attrs], dim=1)

        # MLP
        logits = self.mlp(x)
        return logits

def train(train_loader, threshold):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        data.y = data.y.long()
        output = model(data)
        pred = torch.argmax(output, dim=1)
        loss = F.cross_entropy(output, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    average_loss = total_loss / len(train_loader.dataset)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    accuracy = accuracy_score(all_labels, all_preds)  # Calculate accuracy
    return average_loss, precision, recall, f1, accuracy

def test(data_loader, model, device, threshold):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            data.y = data.y.long()
            output = model(data)
            pred = torch.argmax(output, dim=1)
            loss = F.cross_entropy(output, data.y)

            total_loss += loss.item() * data.num_graphs
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.squeeze().cpu().numpy())

    loss = total_loss / len(data_loader.dataset)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    auc = roc_auc_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)  # Calculate accuracy
    return loss, precision, recall, f1, auc, accuracy
    # model.eval()
    # y_real = []
    # y_pred_probs = []
    
    # with torch.no_grad():
    #     for data in data_loader:
    #         data = data.to(device)
    #         outputs = model(data)
    #         probabilities = torch.nn.functional.softmax(outputs, dim=1)[:, 1]  # Assuming class '1' is positive
    #         y_pred_probs.extend(probabilities.cpu().numpy())
    #         y_real.extend(data.y.cpu().numpy())

    # fpr, tpr, _ = roc_curve(y_real, y_pred_probs)
    # roc_auc = auc(fpr, tpr)

    # # Plot ROC curve
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.savefig('roc.png')
    # plt.close()

    # # Return other metrics as needed, e.g., accuracy, loss, etc.
    # return {'auc': roc_auc, 'fpr': fpr, 'tpr': tpr}


def apply_threshold(output, threshold):
    probabilities = torch.softmax(output, dim=1)
    positive_prob = probabilities[:, 1]
    predictions = (positive_prob >= threshold).long()
    return predictions

def save_checkpoint(epoch, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    logging.info(f'Checkpoint saved at {path}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a GAT model on a graph dataset.")

    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--decay", type=float, default=5e-5, help="Weight decay (L2 loss on parameters).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for applying the classification decision.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--neurons", type=int, default=256, help="Number of neurons in the convolution layer.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of graph convolutional layers.")
    parser.add_argument("--conv_type", type=str, default="GCN", choices=["GCN", "GAT", "SAGE"], help="Type of graph convolution layer.")
    parser.add_argument("--dataset_dir", type=str, default='/home/wllpro/llwang/yfdai/HRAE_paper/final_dataset', help="The save director of the dataset.")
    args = parser.parse_args()

    learning_rate = args.lr
    weight_decay = args.decay
    threshold = args.threshold
    hidden_dim = args.neurons
    num_layers = args.num_layers
    conv_type = args.conv_type
    batch_size = args.batch
    dataset_dir = args.dataset_dir

    log_filename = f"modelc_{args.lr}_{args.decay}_{args.threshold}_{args.neurons}_{args.conv_type}_{args.num_layers}_{args.batch}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Weight decay: {args.decay}")
    logging.info(f"Threshold: {args.threshold}")
    logging.info(f"Batch size: {args.batch}")
    logging.info(f"Number of layers: {args.num_layers}")
    logging.info(f"Convolution type: {args.conv_type}")
    logging.info(f"Convolution neurons: {args.neurons}")


    # Load dataset
    dataset = GNNDataset_criticalpath(root=dataset_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split dataset
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.3, random_state=12)
    # train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.125, random_state=72)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model = ModelB(input_dim=103, hidden_dim = hidden_dim, num_conv_layers=num_layers, other_attrs_dim=273)
    # model = ModelB(input_dim=103, hidden_dim = hidden_dim, num_conv_layers=num_layers, other_attrs_dim=272)
    model = ModelC(input_dim=103, hidden_dim = hidden_dim, num_conv_layers=num_layers, other_attrs_dim=272)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_metric = 0.8
    best_epoch = 0

    for epoch in range(100):
        train_loss, train_precision, train_recall, train_f1, train_accuracy = train(train_loader, threshold)
        # test_results = test(test_loader, model, device, threshold)
        # test_fpr = test_results['fpr']
        # test_auc = test_results['auc']
        # test_tpr = test_results['tpr']

        test_loss, test_precision, test_recall, test_f1, test_auc, test_accuracy = test(test_loader, model, device, threshold)

        logging.info(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Precision: {train_precision:.4f},\
            Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, Train Accuracy: {train_accuracy:.4f},\
            Test Loss: {test_loss:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, \
            Test F1: {test_f1:.4f}, Test AUC: {test_auc:.4f}, Test Accuracy: {test_accuracy:.4f}')    

        # Check if the current F1 score is the best, and save the model checkpoint if it is
        if test_f1 > best_metric:
            best_metric = test_f1
            best_epoch = epoch + 1
            save_checkpoint(epoch + 1, model, optimizer, f'modelc_{test_f1}_epoch_{epoch + 1}.pth')

    logging.info(f'Best performance at epoch {best_epoch} with Test F1: {best_metric:.4f}')
