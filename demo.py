import torch, random
import pickle
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange
import torch.nn as nn
from utils import *
from model import Grammy
from parameter import parse_args, IOStream


# Calculate mean and standard deviation
def calculate_stats(values):
    mean = np.mean(values)
    std = np.std(values)
    return f"{mean:.3f} ± {std:.3f}"


def augment(gexpr, methy):
    with torch.no_grad():
        # 高斯噪声 + 随机mask
        gexpr = gexpr + torch.randn_like(gexpr) * 0.1
        methy = methy + torch.randn_like(methy) * 0.1
    return gexpr, methy


class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, true):
        alpha = 0.5
        rmse = torch.sqrt(self.mse(pred, true))
        mae = self.mae(pred, true)
        return alpha * rmse + (1 - alpha) * mae


# 对比损失（InfoNCE Loss）
def contrastive_loss(gexpr_embed, methy_embed, temperature=0.1):
    sim_matrix = torch.matmul(gexpr_embed, methy_embed.T) / temperature
    labels = torch.arange(gexpr_embed.size(0)).to(gexpr_embed.device)
    loss = nn.CrossEntropyLoss()(sim_matrix, labels)
    return loss

def run(test_loader, model, criterion, device):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader), desc="Test_Loader", leave=False):
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)
            loss = criterion(outputs, data.y)
            test_loss += loss.item()
            all_preds.append(outputs.detach().cpu().numpy().flatten())
            all_labels.append(data.y.detach().cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    pcc, scc, rmse = get_valuation(all_preds, all_labels)
    print(f"pcc:{pcc} ;scc:{scc} ;rmse:{rmse} ")


# a demo to run with our provied data and model dict
if __name__ == '__main__':
    # prase
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 加载到模型中
    device = torch.device('cuda:0')
    state_dict = torch.load("/zoo/model.pth", map_location=device)

    # load model
    model = GramDRP(args, 21, 11).to(device)
    model.load_state_dict(state_dict)
    criterion = ComboLoss()

    # load testSet
    with open("/zoo/test_dataset.pkl", "rb") as f:
        testSet = pickle.load(f)
    test_loader = DataLoader(testSet, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # run
    run(test_loader, model, criterion, device)
