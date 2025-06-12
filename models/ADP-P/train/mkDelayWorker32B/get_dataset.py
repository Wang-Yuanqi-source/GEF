import os
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from extract_xml import *
from get_segment import *
from path_count import *

class GNNDataset_criticalpath(InMemoryDataset): 
    def __init__(self, root, type='', transform=None, pre_transform=None):
        self.type = type
        super(GNNDataset_criticalpath, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir('/home/wllpro/llwang/yfdai/HRAE_paper/raw_crit_paths/mkDelayWorker32B/') 

    @property
    def processed_file_names(self):
        if self.type == 'wo_inter':
            return ['mkDelayWorker32B_wo_inter.pt']
        elif self.type == 'rrg':
            return ['mkDelayWorker32B_rrg.pt']
        elif self.type == 'clb':
            return ['mkDelayWorker32B_clb.pt']
        else:
            return ['mkDelayWorker32B.pt']
        
    def download(self):
        pass

    def process(self):
        report_df = pd.read_csv('/home/wllpro/llwang/yfdai/HRAE_paper/raw_dataset_results/mkDelayWorker32B.csv')
        data_list = []

        for json_file in tqdm(self.raw_file_names, desc='Processing JSON Files'):
            if json_file.endswith('.json'):
                json_file_path = f'/home/wllpro/llwang/yfdai/HRAE_paper/raw_crit_paths/mkDelayWorker32B/{json_file}'
                arch_name = json_file.replace('.json', '')
               
                if f'{arch_name}.xml' in os.listdir('/home/wllpro/llwang/yfdai/HRAE_paper/raw_dataset_archs/'):
                    label_row = report_df[report_df['Architecture'] == arch_name]
                    if (not label_row.empty and not pd.isna(label_row['Critical Path'].iloc[0])):
                        
                        label = torch.tensor([label_row['Critical Path'].iloc[0] * label_row['Total Routing Area'].iloc[0]], dtype=torch.float) * 1e-7
                       
                        node_labels = count_node_names(json_file_path)

                        xml_file_path = f'/home/wllpro/llwang/yfdai/HRAE_paper/raw_dataset_archs/{arch_name}.xml'
                        data = process_xml_to_data(xml_file_path, label, node_labels)  # 传入节点标签
                        data_list.append(data)
        
        print(data_list)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    dataset = GNNDataset_criticalpath(root='/home/wllpro/llwang/yfdai/HRAE_paper/final_dataset')

# Print information about the first 20 graphs in the dataset
# for graph_data in dataset[:10]:
#     edge_sample = graph_data.edge_index[:, 0]
#     node_sample = graph_data.x[100:200]
#     label = graph_data.y
#     print(f'Edge Sample: {edge_sample}, Node Sample: {node_sample}, Label: {label}')
