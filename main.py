import os, torch, random, copy, csv
import pickle

from tqdm import tqdm, trange
import torch.nn as nn
from utils import *
from data import loadDrugCellData, getDrugBldDataLoader, loadExitAndNanDrugCellData, getCellBldDataLoader
from model import SADRP
from parameter import parse_args, IOStream, table_printer
from collections import defaultdict


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


def train(train_loader, model, optimizer, criterion, scheduler, device, epoch):
    model.train()  # 训练模式
    train_loss = 0.0  # 一个epoch，所有样本损失总和
    all_preds = []
    all_labels = []

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc="Train_Loader", leave=False,
                        mininterval=60 * 5):
        data = data.to(device)
        # data.gexpr, data.methylation = augment(data.gexpr, data.methylation)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data.y)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)  # 剪裁可迭代参数的梯度范数，防止梯度爆炸
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

        # 收集预测值和真实值以计算指标
        all_preds.append(outputs.detach().cpu().numpy().flatten())
        all_labels.append(data.y.detach().cpu().numpy().flatten())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    pcc, scc, rmse = get_valuation(all_preds, all_labels)

    return train_loss / len(train_loader.dataset), pcc, scc, rmse


def test(test_loader, model, criterion, device, epoch, save_path=None):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    for i, data in tqdm(enumerate(test_loader), total=len(test_loader), desc="Test_Loader", leave=False,
                        mininterval=60):
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
    if save_path:
        plot_predict_and_label(all_preds, all_labels, save_path)
    return test_loss / len(test_loader.dataset), pcc, scc, rmse


def final_test_by_drug(loader, model, args, device):
    model.eval()

    all_preds = []
    all_labels = []

    # 每个药物一个结果容器
    drug_results = defaultdict(lambda: {
        "pred": [],
        "label": []
    })

    for i, data in tqdm(enumerate(loader), total=len(loader), desc="Test_Loader", leave=False, mininterval=60):
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)

            preds = outputs.detach().cpu().numpy().flatten()
            labels = data.y.detach().cpu().numpy().flatten()
            drugId_list = data.drugId  # data.drugId是 list

            all_preds.append(preds)
            all_labels.append(labels)

            for id, p, l in zip(drugId_list, preds, labels):
                drug_results[id]["pred"].append(p)
                drug_results[id]["label"].append(l)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    pcc, scc, rmse = get_valuation(all_preds, all_labels)

    # 计算每个药物的指标，只保留样本数 >= 20 的药物
    drug_metrics = []  # 收集用于写入 CSV 的行
    for id, res in drug_results.items():
        preds = np.array(res["pred"])
        labels = np.array(res["label"])
        if len(preds) < 20:
            continue  # 跳过样本数太少的药物
        res["pcc"], res["scc"], res["rmse"] = get_valuation(preds, labels)
        drug_metrics.append([id, res["pcc"], res["scc"], res["rmse"]])

    # 按 RMSE 升序排序
    drug_metrics.sort(key=lambda x: x[3])
    with open(f"./outputs/{args.exp_name}/result_{args.seed}.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["DrugID", "PCC", "SCC", "RMSE"])
        writer.writerows(drug_metrics)

    return pcc, scc, rmse

def final_test_by_cell(loader, model, args, device):
    model.eval()

    all_preds = []
    all_labels = []

    # 每个药物一个结果容器
    drug_results = defaultdict(lambda: {
        "pred": [],
        "label": []
    })

    for i, data in tqdm(enumerate(loader), total=len(loader), desc="Test_Loader", leave=False, mininterval=60):
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)

            preds = outputs.detach().cpu().numpy().flatten()
            labels = data.y.detach().cpu().numpy().flatten()
            cellId_list = data.cellId  # data.drugId是 list

            all_preds.append(preds)
            all_labels.append(labels)

            for id, p, l in zip(cellId_list, preds, labels):
                drug_results[id]["pred"].append(p)
                drug_results[id]["label"].append(l)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    pcc, scc, rmse = get_valuation(all_preds, all_labels)

    # 计算每个药物的指标，只保留样本数 >= 20 的药物
    cell_metrics = []  # 收集用于写入 CSV 的行
    for id, res in drug_results.items():
        preds = np.array(res["pred"])
        labels = np.array(res["label"])
        if len(preds) < 5:
            continue  # 跳过样本数太少的药物
        res["pcc"], res["scc"], res["rmse"] = get_valuation(preds, labels)
        cell_metrics.append([id, res["pcc"], res["scc"], res["rmse"]])

    # 按 RMSE 升序排序
    cell_metrics.sort(key=lambda x: x[3])
    with open(f"./outputs/{args.exp_name}/cell_result_{args.seed}.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["CellID", "PCC", "SCC", "RMSE"])
        writer.writerows(cell_metrics)

    return pcc, scc, rmse


def predict_nan_response(loader, model, args, device):
    model.eval()

    all_pedicts = []  # 存储每一个 (drug, cell, ic50)

    # 每个药物的结果
    drug_results = defaultdict(lambda: {
        "ic50_pred": [],
    })

    for i, data in tqdm(enumerate(loader), total=len(loader), desc="Test_Loader", leave=False, mininterval=60):
        data = data.to(device)
        with torch.no_grad():
            outputs = model(data)
            preds = outputs.detach().cpu().numpy().flatten()
            drugId_list = data.drugId  # list
            cellId_list = data.cellId  # list

            for d_id, c_id, p in zip(drugId_list, cellId_list, preds):
                # 存入所有组合记录
                all_pedicts.append({
                    "drugId": d_id,
                    "cellId": c_id,
                    "ic50": p
                })

                # 存入按药物分类的预测值
                drug_results[d_id]["ic50_pred"].append(p)

    # 1、所有 drug-cell 对的预测值
    df_all = pd.DataFrame(all_pedicts)
    df_all_sorted = df_all.sort_values(by="ic50", ascending=True)  # 药效强（低ic50）排前面
    df_all_sorted.to_csv(f"./outputs/{args.exp_name}/predictions.csv", index=False)

    # 2、 每个药物的统计信息
    summary_list = []
    for drug_id, result in drug_results.items():
        preds = result["ic50_pred"]
        summary_list.append({
            "drugId": drug_id,
            "ic50_list": preds,
            "avg_ic50": sum(preds) / len(preds),
            "num": len(preds)
        })

    # 创建 DataFrame，将 ic50_list 转为字符串形式以便保存
    df_summary = pd.DataFrame(summary_list)
    df_summary["ic50_list"] = df_summary["ic50_list"].apply(lambda x: ";".join([f"{v:.4f}" for v in x]))
    df_summary = df_summary.sort_values(by="avg_ic50", ascending=True)  # 可以按平均药效排序
    df_summary.to_csv(f"./outputs/{args.exp_name}/drug_statistics.csv", index=False)


def exp_init():
    # init
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    if not os.path.exists('outputs/' + args.exp_name):
        os.mkdir('outputs/' + args.exp_name)

    # make backup
    output_dir = f"outputs/{args.exp_name}"
    os.system(f'cp main.py {output_dir}/main.py.backup')
    os.system(f'cp data_process/dataSet.py {output_dir}/dataSet.py.backup')
    os.system(f'cp model.py {output_dir}/model.py.backup')
    os.system(f'cp model_ori.py {output_dir}/model_ori.py.backup')
    os.system(f'cp model_cpy.py {output_dir}/model_cpy.py.backup')
    os.system(f'cp parameter.py {output_dir}/parameter.py.backup')
    os.system(f'cp layer.py {output_dir}/layer.py.backup')
    os.system(f'cp layer.py {output_dir}/layer.py.backup')
    os.system(f'cp data.py {output_dir}/data.py.backup')


def main(args):
    model_path = "your/path/model.pth"
    TRAIN_FROM_CHEKPOINT = False
    best_model_dict = None
    random.seed(args.seed)  # 设置Python随机种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机种子
    patience = 25
    counter = 0
    best_loss = 999

    IO = IOStream('outputs/' + args.exp_name + '/run.log')
    IO.cprint("--------------------------------------------------------------")
    IO.cprint(str(table_printer(args)))  # 参数可视化

    IO.cprint("your topic")
    device = f'cuda:{args.gpu_index}'
    print(f"Using GPU {args.gpu_index}")

    # mix test
    train_loader, val_loader, test_loader, _ , num_node_features, num_edge_features, _ \
        = loadDrugCellData(args, "save", "single")

    # blind test
    # train_loader, val_loader, test_loader, _ , num_node_features, num_edge_features, _ \
    #     = getDrugBldDataLoader(args, "save", "single")
    # train_loader, val_loader, test_loader, _ , num_node_features, num_edge_features, _ \
    #     = getCellBldDataLoader(args, "save", "single")

    # predict NAN response
    # train_loader, val_loader, test_loader, _ , num_node_features, num_edge_features, _ \
    #     = loadExitAndNanDrugCellData(args, "save", "single")

    # model
    model = SADRP(args, num_node_features, num_edge_features)
    if TRAIN_FROM_CHEKPOINT:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        IO.cprint("train from check point")
    model = model.to(device)
    IO.cprint(str(model))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    IO.cprint('Model Parameter: {}'.format(total_params))

    # optimizer
    optimizer = torch.optim.AdamW(model.get_parameter_groups(), lr=args.lr, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    IO.cprint('Using AdamW...')

    # loss
    criterion = ComboLoss()
    losses, test_losses = [], []
    IO.cprint('Using MSELoss...')

    # train
    epochs = trange(args.epochs, leave=True, desc="Epochs")
    for epoch in epochs:
        train_loss, pcc, scc, rmse = train(train_loader, model, optimizer, criterion, scheduler, device, epoch)
        losses.append(train_loss)
        IO.cprint(
            'Epoch #{:03d},Train_Loss:{:.4f},pcc{:.3f},scc{:.3f},rmse{:.3f}'.format(epoch, train_loss, pcc, scc, rmse))

        if (epoch + 1) >= 70:
            # do val
            test_loss, pcc, scc, rmse = test(val_loader, model, criterion, device, epoch)
            test_losses.append(test_loss)
            if (epoch + 1) % 5 == 0:
                IO.cprint(
                    '[##VAL##]Epoch #{:03d},Test_Loss:{:.4f},pcc:{:.3f},scc:{:.3f},rmse:{:.3f}'.format(epoch, test_loss,
                                                                                                   pcc, scc, rmse))
            # early stop
            if test_loss < best_loss:
                best_model_dict = copy.deepcopy(model.state_dict())
                best_loss = test_loss
                counter = 0
            else:
                counter = counter + 1
            if counter == patience:
                IO.cprint("[info] **** early stop ****")
                break

        # save check point
        if epoch % 50 == 0 and epoch > 10:
            torch.save(best_model_dict, 'outputs/%s/model.pth' % args.exp_name)
            IO.cprint('[INFO]save checkpoint: epoch {} model saved in: {}'.format(epoch,
                                                                                  '***outputs/%s/model.pth ***' % args.exp_name))

    # load best model
    model.load_state_dict(best_model_dict)
    torch.save(model.state_dict(), f'outputs/{args.exp_name}/model_{args.seed}.pth')
    IO.cprint(f'model saved in: outputs/{args.exp_name}/model_{args.seed}.pth')

    IO.cprint("\n******* on test dataset *******")
    # if you want to predict NaN response
    #predict_nan_response(test_loader, model, args, device)

    pcc, scc, rmse = final_test_by_drug(test_loader, model, args, device)
    IO.cprint(f"test pcc: {pcc}")
    IO.cprint(f"test scc: {scc}")
    IO.cprint(f"test rmse: {rmse}")

    final_test_by_cell(test_loader, model, args, device)



if __name__ == '__main__':
    args = parse_args()
    exp_init()
    main(args)


