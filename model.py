from typing import Union, Tuple, Dict, List

import torch
import networkx as nx
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, GATConv, SAGEConv, global_max_pool, GlobalAttention
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import GCNConv
from data_process.dataSet import MyData
from layer import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding
from static import CONSTANT
import torch.nn.functional as F


def floyd_warshall_source_to_all(G, source, cutoff=None):
    "Floyd-Warshall算法查询最短路径(BFS遍历图)"
    if source not in G:
        raise nx.NodeNotFound("Source {} not in G".format(source))

    edges = {edge: i for i, edge in enumerate(G.edges())}

    level = 0  # the current level
    nextlevel = {source: 1}  # list of nodes to check at next level
    node_paths = {source: [source]}  # paths dictionary  (paths to key from source)
    edge_paths = {source: []}

    while nextlevel:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in G[v]:
                if w not in node_paths:
                    node_paths[w] = node_paths[v] + [w]
                    edge_paths[w] = edge_paths[v] + [edges[tuple(node_paths[w][-2:])]]
                    nextlevel[w] = 1

        level = level + 1

        if (cutoff is not None and cutoff <= level):
            break
    return node_paths, edge_paths

memory_dict = {}
def all_pairs_shortest_path(G,max_path_distance,node_shift,edge_shift):
    graph_hashable = frozenset(G.edges())
    #src_tensor,dst_tensor,path_tensor = None
    if graph_hashable in memory_dict:
        src_tensor = memory_dict[graph_hashable][0]
        dst_tensor = memory_dict[graph_hashable][1]
        path_tensor = memory_dict[graph_hashable][2]
    else:
        #print("###create_new###")
        paths = {n: floyd_warshall_source_to_all(G, n) for n in G}
        #node_paths = {n: paths[n][0] for n in paths}
        edge_paths = {n: paths[n][1] for n in paths}
        src_list,dst_list,path_indices = [],[],[]
        for src, q in edge_paths.items():
            for dst, path in q.items():
                src_list.append(src)
                dst_list.append(dst)
                path_indices.append(path[:max_path_distance])  # Truncate to max_path_distance
        src_tensor = torch.tensor(src_list)
        dst_tensor = torch.tensor(dst_list)
        path_indices = [sublist + [-1] * (max_path_distance - len(sublist)) for sublist in path_indices]
        path_tensor = torch.tensor(path_indices)
        memory_dict[graph_hashable] = [src_tensor,dst_tensor,path_tensor]
    # do shift
    src_tensor = src_tensor + node_shift
    dst_tensor = dst_tensor + node_shift
    mask = path_tensor != -1
    path_tensor = torch.where(mask, path_tensor + edge_shift, path_tensor)
    return src_tensor,dst_tensor,path_tensor


def shortest_path_distance(data: Data) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    G = to_networkx(data)
    node_paths, edge_paths = all_pairs_shortest_path(G)
    return node_paths, edge_paths


def batched_shortest_path_distance(data) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    graphs = [to_networkx(sub_data) for sub_data in data.to_data_list()]
    relabeled_graphs = []
    shift = 0
    for i in range(len(graphs)):
        num_nodes = graphs[i].number_of_nodes()
        relabeled_graphs.append(nx.relabel_nodes(graphs[i], {i: i + shift for i in range(num_nodes)}))
        shift += num_nodes

    paths = [all_pairs_shortest_path(G) for G in relabeled_graphs]
    node_paths = {}
    edge_paths = {}

    for path in paths:
        for k, v in path[0].items():
            node_paths[k] = v
        for k, v in path[1].items():
            edge_paths[k] = v

    return node_paths, edge_paths

def batched_shortest_path(data, max_path_distance):
    graphs = [to_networkx(sub_data) for sub_data in data.to_data_list()]
    edge_nums = []
    node_nums = []
    for i in range(len(graphs)):
        node_nums.append(graphs[i].number_of_nodes())
        edge_nums.append(graphs[i].number_of_edges())

    all_src,all_dst,all_path = None,None,None
    j = 0
    node_shift = 0
    edge_shift = 0
    for G in graphs:
        src, dst, paths = all_pairs_shortest_path(G,max_path_distance,node_shift,edge_shift)
        if(j == 0):
            all_src, all_dst, all_path = src,dst,paths
        else:
            all_src = torch.cat([all_src,src],dim=0)
            all_dst = torch.cat([all_dst,dst],dim=0)
            all_path = torch.cat([all_path,paths],dim=0)
        node_shift += node_nums[j]
        edge_shift += edge_nums[j]
        j += 1

    return [all_src,all_dst,all_path]

class SpatialEncodingPlus(nn.Module):
    def __init__(self, max_path_distance):
        super().__init__()
        self.max_path_distance = max_path_distance
        self.b = nn.Parameter(torch.randn(self.max_path_distance))

    def forward(self, edge_index, num_nodes):
        device = edge_index.device

        # 构造稀疏邻接矩阵，添加自环
        values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device,requires_grad=False)
        adj_matrix = torch.sparse_coo_tensor(
            edge_index, values, (num_nodes, num_nodes), device=device
        )
        adj_matrix = adj_matrix + torch.eye(num_nodes, device=device).to_sparse()

        # 初始化路径矩阵
        spatial_matrix = torch.zeros((num_nodes, num_nodes), device=device)
        adj_last = torch.zeros((num_nodes, num_nodes), device=device, requires_grad=False).to_sparse()

        for i in range(self.max_path_distance):
            # 当前路径长度贡献
            spatial_matrix += (adj_matrix-adj_last) * self.b[i]
            adj_last = adj_matrix

            # 更新下一次路径
            adj_matrix = torch.sparse.mm(adj_matrix, adj_matrix)
            adj_matrix = torch.sparse_coo_tensor(
                adj_matrix.indices(),
                torch.clamp(adj_matrix.values(), max=1),  # 保持二值化
                adj_matrix.size(),
                device=device,
            )
        return spatial_matrix

class SEBlock(nn.Module):
    def __init__(self, channels):
        """
        Squeeze-and-Excitation Block
        :param channels: 输入通道数
        :param reduction: 通道压缩比例
        """
        super(SEBlock, self).__init__()
        self.fc = nn.Linear(channels, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_pool = x.mean(dim=-1).to(x.device)  # (batch_size, channels)
        x_pool = self.fc(x_pool)
        x_pool = self.sigmoid(x_pool).unsqueeze(2)  # (batch_size, channels, 1)

        # Scale: 按通道加权
        return torch.sum(x * x_pool,dim=1)


class MultiHeadLinear(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8):
        """
        :param input_dim: 输入特征维度
        :param output_dim: 输出特征维度
        :param num_heads: 头的数量
        """
        super(MultiHeadLinear, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear_layers = nn.ModuleList([
            nn.Linear(self.input_dim, self.output_dim) for _ in range(num_heads)
        ])
        self.se_block = SEBlock(num_heads)

    def forward(self, x):
        head_outputs = [linear(x).unsqueeze(1) for linear in self.linear_layers]
        head_outputs = torch.cat(head_outputs, dim=1)
        return self.se_block(head_outputs)

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)

class OmicsCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x1, x2):  # x1 attends to x2
        Q = self.query(x1)
        K = self.key(x2)
        V = self.value(x2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        return out

class CrossAttentionOmicsFeatureEncoder(nn.Module):
    def __init__(self, gexpr_dim=CONSTANT.GEXPR_DIM, methy_dim=CONSTANT.METHY_DIM):
        super().__init__()
        self.gexpr_proj1 = nn.Linear(gexpr_dim, 512)
        self.gexpr_proj2 = nn.Linear(512, 256)
        self.methy_proj1 = nn.Linear(methy_dim, 512)
        self.methy_proj2 = nn.Linear(512, 256)

        self.att = SemanticAttention(256)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, gexpr, methy):
        g = self.relu(self.gexpr_proj1(gexpr))
        g = self.dropout(g)
        g = self.relu(self.gexpr_proj2(g))

        m = self.relu(self.methy_proj1(methy))
        m = self.dropout(m)
        m = self.relu(self.methy_proj2(m))

        g_m = torch.stack((g, m), dim=1)
        g_m = self.att(g_m)

        return g_m, g, m

class OmicsFeatureEncoder(nn.Module):
    def __init__(self, gexpr_dim=697, mut_dim=34673, methy_dim=808):
        super(OmicsFeatureEncoder, self).__init__()
        # for gexpr_feature and methy_feature
        #self.mulfc_gexpr1 = MultiHeadLinear(gexpr_dim,256)
        self.mulfc_gexpr1 = nn.Linear(gexpr_dim, 256)
        self.fc_gexpr2 = nn.Linear(256,128)

        #self.mulfc_methy1 = MultiHeadLinear(methy_dim,256)
        self.mulfc_methy1 = nn.Linear(methy_dim, 256)
        self.fc_methy2 = nn.Linear(256,128)

        self.fc = nn.Linear(256,256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    def forward(self, gexpr, methy):
        """
        Args:
            gexpr(697 dim), mut(34673 dim), methy(808 dim)
        Returns:
            Tensor: 低维表示，形状为 (batch_size, output_dim)。
        """
        gexpr = self.relu(self.mulfc_gexpr1(gexpr))
        gexpr = self.dropout(gexpr)
        gexpr = self.relu(self.fc_gexpr2(gexpr))

        methy = self.relu(self.mulfc_methy1(methy))
        methy = self.dropout(methy)
        methy = self.relu(self.fc_methy2(methy))

        return self.relu(self.fc(torch.cat([gexpr,methy],dim=1))), gexpr, methy

class EmbeddingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate = 0.3):
        super(EmbeddingLayer, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, output_dim)
        self.conv2 = GCNConv(output_dim, output_dim)
        self.prelu = torch.nn.PReLU()
        self.dropout = nn.Dropout(dropout_rate)  # 增加 Dropout

    def forward(self, x, edge_index):
        x = self.prelu(self.linear(x))
        x = self.prelu(self.conv1(x, edge_index))
        x = self.prelu(self.conv2(x, edge_index))
        return x

# class BaseEmbeddingLayer(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate = 0.3):
#         super(EmbeddingLayer, self).__init__()
#         self.linear = nn.Linear(input_dim, hidden_dim)
#         self.conv1 = SAGEConv(hidden_dim, hidden_dim)
#         self.conv2 = SAGEConv(hidden_dim, output_dim)
#         self.prelu = torch.nn.PReLU()
#
#     def forward(self, x, edge_index):
#         x = self.prelu(self.linear(x))
#         x = self.prelu(self.conv1(x, edge_index))
#         x = self.prelu(self.conv2(x, edge_index))
#         return x

class GraphFpLin(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(192, 192)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(192, 192)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out1 = self.lin1(x)
        out = self.relu(out1)
        out = self.dropout(out)
        out = self.lin2(out)
        out = self.relu(out)
        return out + out1


class GramDRP(nn.Module):
    def __init__(self, args, num_node_features, num_edge_features):
        """
        :param num_layers: number of Graphormer layers
        :param input_node_dim: input dimension of node features
        :param node_dim: hidden dimensions of node features
        :param input_edge_dim: input dimension of edge features
        :param edge_dim: hidden dimensions of edge features
        :param output_dim: number of output node features
        :param n_heads: number of attention heads
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max out degree of nodes
        :param max_path_distance: max pairwise distance between two nodes
        """
        super().__init__()

        self.num_layers = args.num_layers
        self.input_node_dim = num_node_features
        self.node_dim = args.node_dim
        self.input_edge_dim = num_edge_features
        self.edge_dim = args.edge_dim
        self.output_dim = args.output_dim
        self.num_heads = args.num_heads
        self.max_in_degree = args.max_in_degree
        self.max_out_degree = args.max_out_degree
        self.max_path_distance = args.max_path_distance
        self.fp_dim = CONSTANT.MORGAN_FP_DIM

        self.node_embedding = EmbeddingLayer(self.input_node_dim, self.node_dim//2, self.node_dim)

        # embedding_dict = torch.load("./pretrained_attrmask_encoder.pth")
        # new_state_dict = {}
        # for k, v in embedding_dict.items():
        #     if k.startswith("encoder."):
        #         new_key = k[len("encoder."):]  # 去掉前缀
        #     else:
        #         new_key = k
        #     new_state_dict[new_key] = v
        #
        # self.node_embedding.load_state_dict(new_state_dict)

        self.edge_in_lin = nn.Linear(self.input_edge_dim, self.edge_dim)


        self.spatial_encoding = SpatialEncodingPlus(
            max_path_distance=self.max_path_distance,
        )

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                num_heads=self.num_heads,
                max_path_distance=self.max_path_distance,
                dropout_rate = 0.3) for _ in range(self.num_layers)
        ])

        # use morgan fingerprint or embedding
        self.fp_layers = nn.ModuleList([
            nn.Linear(self.fp_dim, 256),  # 第一层
            #nn.Linear(512, 256),  # 第二层
            nn.Linear(256, 64)  # 第三层
        ])

        self.graph_fp_lin = nn.Linear(128+64,128+64)

        # 导入预训练模型
        self.omics_encoder = OmicsFeatureEncoder()
        encoder_dict = torch.load("./pretrained_omics_encoder.pth")

        self.omics_encoder.load_state_dict(encoder_dict)

        self.node_out_lin1 = nn.Linear(self.node_dim+256+64, 512)
        self.node_out_lin3 = nn.Linear(512, 128)
        self.node_out_lin4 = nn.Linear(128, self.output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.batchnorm_node = nn.BatchNorm1d(num_features=self.input_node_dim)
        self.batchnorm_edge = nn.BatchNorm1d(num_features=self.input_edge_dim)

    @staticmethod
    def init_gate_nn(m):
        if isinstance(m, nn.Linear):
            if m.out_features == 1:
                # 输出是 attention 权重时，用较小 gain 防止过大权重
                nn.init.xavier_uniform_(m.weight, gain=0.1)
            else:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_parameter_groups(self):
        return [
            {'params': self.omics_encoder.parameters(), 'lr': 1e-4},
            #{'params': self.node_embedding.parameters(), 'lr': 1e-4},
            {'params': [
                p for n, p in self.named_parameters()
                if not n.startswith('omics_encoder')
            ]}
        ]

    def forward(self, data: Union[MyData]) -> torch.Tensor:
        """
        :param data: input graph of batch of graphs
        :return: torch.Tensor, output node embeddings
        """
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_index.requires_grad = False
        edge_attr = data.edge_attr.float()
        nodeNum = x.shape[0]
        gexpr_feature = data.gexpr
        methy_feature = data.methylation
        fingerPrint = data.fingerPrint
        # ptr图的位置指针([ 0, 12, 22, 43, 62, 73, 95,..., 899])
        ptr = data.ptr
        batch = data.batch

        edge_paths = batched_shortest_path(data,self.max_path_distance)
        del data

        # node and edge encode
        x = self.batchnorm_node(x.to(self.edge_in_lin.weight.device))
        x = self.node_embedding(x, edge_index)


        edge_attr = self.edge_in_lin(edge_attr)

        b = self.spatial_encoding(edge_index, nodeNum)

        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths, ptr)

        x = global_mean_pool(x, batch)

        fingerPrint = fingerPrint.reshape(-1, self.fp_dim)
        x_fp = fingerPrint
        for i, layer in enumerate(self.fp_layers):
            fingerPrint = layer(fingerPrint)
            if i < len(self.layers) - 1:
                fingerPrint = self.relu(fingerPrint)
                fingerPrint = self.dropout(fingerPrint)


        x = self.graph_fp_lin(torch.cat([x,fingerPrint],dim=1))


        # deal with cell_line features
        gexpr_feature = gexpr_feature.reshape(x.shape[0],-1)
        #mutation_feature = data.mutation.reshape(x.shape[0],-1)
        methy_feature = methy_feature.reshape(x.shape[0],-1)
        omics_in = torch.cat([gexpr_feature, methy_feature], dim=1)
        omics, gexpr_view, methy_view = self.omics_encoder(gexpr_feature, methy_feature)

        # predict
        x = self.relu(self.node_out_lin1(torch.cat([x, omics], dim=1)))
        x = self.dropout(x)
        x = self.relu(self.node_out_lin3(x))
        x = self.dropout(x)
        x = self.node_out_lin4(x)
        return x



"""
m = OmicsFeatureEncoder()
a = torch.randn((5,697))
b = torch.randn((5,34673))
c = torch.randn((5,808))
print(m(a,b,c).shape)

edge_index = torch.tensor(
        [[0, 2, 1, 2,3, 1, 3, 4],  # 起点
         [1, 3, 2, 1,4, 0, 2, 3]],  # 终点
        dtype=torch.long
    )

model = SpatialEncodingPlus(4)
spatial_matrix = model(edge_index, 5)

"""
