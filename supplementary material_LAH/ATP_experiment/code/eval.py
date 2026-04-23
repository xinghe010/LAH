import os
import torch
import argparse
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from trainer import train
from model import PremiseSelectionModel
from dataset import FormulaGraphDataset, formula_graph_collate
from utils import set_recorder, dump_pickle_file

def hyper_parameters():
    params = argparse.ArgumentParser()
    params.add_argument(
        "--model_save",
        type=str,
        default="model_save",
        help="the directory to save models"
    )
    params.add_argument(
        "--root_dir",
        type=str,
        default="dataset",
        help="the directory to save data")
    params.add_argument("--node_out_channels",
                        type=int,
                        default=512,
                        help="the dimension of node")
    params.add_argument("--layers",
                        type=int,
                        default=1,
                        help="the number of message passing steps")
    params.add_argument("--device",
                        type=str,
                        default="cuda:0",
                        help="device name")
    params.add_argument("--epochs",
                        type=int,
                        default=50,
                        help='Number of training episodes')
    params.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Initial learning rate for Adam")
    params.add_argument("--weight_decay",
                        type=float,
                        default=1e-4,
                        help="L2 normalization penality")
    params.add_argument("--batch_size",
                        type=int,
                        default=64,
                        help="Batch Size")
    params.add_argument("--seed",
                        type=int,
                        default=24,
                        help="Random seed")
    args = params.parse_args()
    return args

def main():
    args = hyper_parameters()

    if not os.path.exists(args.model_save):
        os.makedirs(args.model_save)
    recorder = set_recorder("LAH-all",
                            os.path.join(args.model_save, "record.log"))

    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.cuda.manual_seed_all(args.seed)

    params_info = ''
    for key, value in vars(args).items():
        params_info += '\n{}: {}'.format(key, value)
    recorder.info(params_info)

    model = PremiseSelectionModel(793,
                                  args.node_out_channels,
                                  args.layers).to(device=args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = ReduceLROnPlateau(optimizer)
    recorder.info('------DATA LOADING------')

    train_dataset = FormulaGraphDataset(os.path.join(args.root_dir, "train"),
                                        "train",
                                        os.path.join(args.root_dir,
                                                     "statements"),
                                        os.path.join(args.root_dir,
                                                     "node_dict.pkl"),
                                        rename=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True, collate_fn=formula_graph_collate,
                              follow_batch=["x_s", "x_t"])

    recorder.info(f'------DATA LOADED: {len(train_dataset)} samples------')

    recorder.info('------PROCESS START------')
    history = {"train_loss": [], "train_acc": []}

    best_epoch = -1
    best_state_dict = {"model": None}
    best_train_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        recorder.info('------learning rate is {}------'.format(
            optimizer.param_groups[0]["lr"]))

        train_loss, train_acc, train_f1, train_recall, train_precision = train(
            epoch, train_loader, model, optimizer, args.device, recorder)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if best_train_loss > train_loss:
            best_epoch = epoch
            best_state_dict["model"] = {k: v.clone() for k, v in model.state_dict().items()}
            best_train_loss = train_loss

        lr_scheduler.step(train_loss)

    torch.save(best_state_dict, os.path.join(args.model_save, "best.pt"))
    recorder.info('------the best epoch is {}------'.format(best_epoch))

    dump_pickle_file(history, os.path.join(args.model_save, "history.pkl"))

    recorder.info('------PROCESS FINISH------')

if __name__ == "__main__":
    main()
