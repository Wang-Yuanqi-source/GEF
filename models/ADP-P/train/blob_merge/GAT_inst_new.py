import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, JumpingKnowledge, global_mean_pool, SAGEConv, GCNConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import logging
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.nn import Linear, ModuleList, BatchNorm1d
import numpy as np
from sklearn.metrics import r2_score

# Assuming GNNDataset_criticalpath is already defined as per your provided script
from get_dataset import GNNDataset_criticalpath

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

        self.reg_lin1 = torch.nn.Linear(272 + concatenated_feature_size + 1, 32)  
        self.reg_lin2 = torch.nn.Linear(32, 1)  


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

        # print("Shape of reg_input:", reg_input.shape)  
        # print("Shape of x_class:", x_class.shape)  
        # print("Shape of x_pool:", x_pool.shape)  

        last_attr = other_attrs[:, -1].unsqueeze(1)  # Shape: [batch_size, 1]

        reg_input = torch.cat((reg_input, x_class, x_pool), dim=1) 
        print()

        reg_output = F.relu(self.reg_lin1(reg_input))
        reg_output = self.reg_lin2(reg_output)

        return node_output, reg_output, last_attr  

def train(train_loader):
    model.train()
    total_loss = 0
    total_node_loss = 0
    total_reg_loss = 0
    total_mape = 0
    all_reg_preds = []
    all_reg_labels = []

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        node_output, reg_output, last_attr = model(data)

        node_loss = F.mse_loss(node_output.view_as(data.y).float(), data.y.float()) 
      
        reg_loss = F.mse_loss(reg_output.squeeze(), last_attr.squeeze())  
        loss = node_loss + 100 * reg_loss  # Total loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_node_loss += node_loss.item() * data.num_graphs
        total_reg_loss += reg_loss.item() * data.num_graphs

        all_reg_preds.extend(reg_output.detach().cpu().numpy())
        all_reg_labels.extend(last_attr.detach().cpu().numpy())

    average_loss = total_loss / len(train_loader.dataset)
    average_node_loss = total_node_loss / len(train_loader.dataset)  
    average_reg_loss = total_reg_loss / len(train_loader.dataset)
    all_reg_preds = np.array(all_reg_preds)
    all_reg_labels = np.array(all_reg_labels)
    mape = np.mean(np.abs((all_reg_labels - all_reg_preds) / all_reg_labels)) * 100
    r = np.corrcoef(all_reg_labels.flatten(), all_reg_preds.flatten())[0, 1]
    r2 = r2_score(all_reg_labels, all_reg_preds)

    return average_loss, average_node_loss, average_reg_loss, mape, r, r2

def test(data_loader):
    model.eval()
    total_loss = 0
    total_node_loss = 0
    total_reg_loss = 0
    total_mape = 0
    all_preds = []
    all_labels = []
    all_reg_preds = []
    all_reg_labels = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)

            node_output, reg_output, last_attr = model(data)

            node_loss = F.mse_loss(node_output.view_as(data.y).float(), data.y.float())  
           
            reg_loss = F.mse_loss(reg_output.squeeze(), last_attr.squeeze()) 
            loss = node_loss + 100 * reg_loss  # Total loss

            total_loss += loss.item() * data.num_graphs
            total_node_loss += node_loss.item() * data.num_graphs
            total_reg_loss += reg_loss.item() * data.num_graphs
            
            all_reg_preds.extend(reg_output.detach().cpu().numpy())
            all_reg_labels.extend(last_attr.detach().cpu().numpy())

    average_loss = total_loss / len(train_loader.dataset)
    average_node_loss = total_node_loss / len(train_loader.dataset)  
    average_reg_loss = total_reg_loss / len(train_loader.dataset)
    all_reg_preds = np.array(all_reg_preds)
    all_reg_labels = np.array(all_reg_labels)
    mape = np.mean(np.abs((all_reg_labels - all_reg_preds) / all_reg_labels)) * 100
    r = np.corrcoef(all_reg_labels.flatten(), all_reg_preds.flatten())[0, 1]
    r2 = r2_score(all_reg_labels, all_reg_preds)

    return average_loss, average_node_loss, average_reg_loss, mape, r, r2

def evaluate(model, test_loader, device, plot_path='true_vs_predicted.png'):
    model.eval()
    all_reg_preds = []
    all_reg_labels = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            _, reg_output, last_attr = model(data)
            all_reg_preds.extend(reg_output.cpu().numpy().flatten())
            all_reg_labels.extend(last_attr.cpu().numpy().flatten())

    all_reg_preds = np.array(all_reg_preds)
    all_reg_labels = np.array(all_reg_labels)

    return all_reg_preds, all_reg_labels

    
    # mape = np.mean(np.abs((all_reg_labels - all_reg_preds) / all_reg_labels)) * 100
    # r = np.corrcoef(all_reg_labels, all_reg_preds)[0, 1]
    # r2 = r2_score(all_reg_labels, all_reg_preds)

    
    # plt.figure(figsize=(8, 6))
    # plt.scatter(all_reg_labels, all_reg_preds, alpha=0.5)
    # plt.plot([all_reg_labels.min(), all_reg_labels.max()],
    #          [all_reg_labels.min(), all_reg_labels.max()], 'k--', lw=2)
    # plt.xlabel('True Values')
    # plt.ylabel('Predicted Values')
    # plt.title(f'True vs Predicted\nR: {r:.2f}, R²: {r2:.2f}, MAPE: {mape:.2f}%')
    # plt.savefig(plot_path)
    # plt.close()

def save_checkpoint(epoch, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)
    logging.info(f'Checkpoint saved at {path}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a GAT model on a graph dataset.")

    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--decay", type=float, default=5e-4, help="Weight decay (L2 loss on parameters).")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--neurons", type=int, default=8, help="Number of neurons in the convolution layer.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of graph convolutional layers.")
    parser.add_argument("--conv_type", type=str, default="GCN", choices=["GCN", "GAT", "SAGE"], help="Type of graph convolution layer.")
    parser.add_argument("--dataset_dir", type=str, default='/home/wllpro/llwang/yfdai/HRAE_paper/final_dataset', help="The save director of the dataset.")
    args = parser.parse_args()

    learning_rate = args.lr
    weight_decay = args.decay
    conv_neurons = args.neurons
    num_layers = args.num_layers
    conv_type = args.conv_type
    batch_size = args.batch
    dataset_dir = args.dataset_dir

    log_filename = f"best_{args.lr}_{args.neurons}_{args.conv_type}_{args.num_layers}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Weight decay: {args.decay}")
    logging.info(f"Batch size: {args.batch}")
    logging.info(f"Number of layers: {args.num_layers}")
    logging.info(f"Convolution type: {args.conv_type}")
    logging.info(f"Convolution neurons: {args.neurons}")


    # Load dataset
    dataset = GNNDataset_criticalpath(root=dataset_dir, type='ADP')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split dataset
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.3, random_state=122)
    # train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.125, random_state=72)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = SAGE_JK(num_node_features=103, conv_neurons = conv_neurons, num_layers=num_layers, conv_type=conv_type)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_metric = 0.7
    best_epoch = 0

    for epoch in range(300):
        
        train_loss, train_node_loss, train_reg_loss, train_mape, train_r, train_rr = train(train_loader)
        test_loss, test_node_loss, test_reg_loss, test_mape, test_r, test_rr = test(test_loader)

        logging.info(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Node Loss: {train_node_loss:.4f}, '
                    f'Train Reg Loss: {train_reg_loss:.4f}, Train MAPE: {train_mape:.4f}, Train R: {train_r:.4f}, '
                    f'Train R2: {train_rr:.4f}, Test Loss: {test_loss:.4f}, Test Node Loss: {test_node_loss:.4f}, '
                    f'Test Reg Loss: {test_reg_loss:.4f}, Test MAPE: {test_mape:.4f}, Test R: {test_r:.4f}, '
                    f'Test R2: {test_rr:.4f}')

        if test_rr > best_metric:
            best_metric = test_rr
            best_epoch = epoch + 1
            save_checkpoint(epoch + 1, model, optimizer, f'wo_inter_{args.lr}_{args.conv_type}_{args.num_layers}_{test_rr:.4f}_epoch_{epoch + 1}.pth')

    logging.info(f'Best performance at epoch {best_epoch} with Test MAPE: {best_metric:.4f}')
