from models import PepBAN
from time import time
from utils import set_seed,mkdir
from configs import get_cfg_defaults
from dataloader import PepPIDataset, MultiDataLoader
from torch.utils.data import DataLoader
from trainer import Trainer
from domain_adaptator import Discriminator
import torch
import argparse
import warnings, os
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="PepBAN for PepPI prediction")
parser.add_argument('--cfg', required=True, help="path to config file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK', help='dataset')
parser.add_argument('--result', required=True, type=str)
args = parser.parse_args()

def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.SOLVER.SEED)
    cfg.RESULT.OUTPUT_DIR=args.result
    mkdir(cfg.RESULT.OUTPUT_DIR)
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'./datasets/{args.data}'

    train_source_path = os.path.join(dataFolder, 'source_train.csv')
    train_target_path = os.path.join(dataFolder, 'target_train.csv')
    test_target_path = os.path.join(dataFolder, 'target_test.csv')

    df_train_source = pd.read_csv(train_source_path)
    df_train_target = pd.read_csv(train_target_path)
    df_test_target = pd.read_csv(test_target_path)

    train_dataset = PepPIDataset(df_train_source.index.values, df_train_source)
    train_target_dataset = PepPIDataset(df_train_target.index.values, df_train_target)
    test_target_dataset = PepPIDataset(df_test_target.index.values, df_test_target)

    hyper_params = {"LR": cfg.SOLVER.LR, "Output_dir": cfg.RESULT.OUTPUT_DIR, "DA_use": cfg.DA.USE, "DA_task": cfg.DA.TASK}

    if cfg.DA.USE:
        da_hyper_params = {
            "DA_init_epoch": cfg.DA.INIT_EPOCH,
            "Use_DA_entropy": cfg.DA.USE_ENTROPY,
            "Random_layer": cfg.DA.RANDOM_LAYER,
            "Original_random": cfg.DA.ORIGINAL_RANDOM,
            "DA_optim_lr": cfg.SOLVER.DA_LR
        }
        hyper_params.update(da_hyper_params)

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True}

    if cfg.DA.USE:
        source_train_generator = DataLoader(train_dataset, **params)
        target_train_generator = DataLoader(train_target_dataset, **params)
        n_batches = max(len(source_train_generator), len(target_train_generator))
        multi_generator = MultiDataLoader(dataloaders=[source_train_generator, target_train_generator], n_batches=n_batches)
        params['shuffle'] = False
        params['drop_last'] = False
        test_generator = DataLoader(test_target_dataset, **params)
        target_generator=DataLoader(target_dataset, **params)
    else:
        training_generator = DataLoader(train_dataset, **params)
        params['shuffle'] = False
        params['drop_last'] = False
        test_generator = DataLoader(test_target_dataset, **params)

    model = PepBAN(**cfg).to(device)

    if cfg.DA.USE:
        if cfg["DA"]["RANDOM_LAYER"]:
            domain_dmm = Discriminator(input_size=cfg["DA"]["RANDOM_DIM"], n_class=cfg["DECODER"]["BINARY"]).to(device)
        else:
            domain_dmm = Discriminator(input_size=cfg["DECODER"]["IN_DIM"] * cfg["DECODER"]["BINARY"],
                                       n_class=cfg["DECODER"]["BINARY"]).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
        opt_da = torch.optim.Adam(domain_dmm.parameters(), lr=cfg.SOLVER.DA_LR)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    torch.backends.cudnn.benchmark = True

    if cfg.DA.USE:
        trainer = Trainer(model, opt, device, multi_generator, test_generator, opt_da=opt_da,
                          discriminator=domain_dmm, **cfg)
    else:
        trainer = Trainer(model, opt, device, training_generator,test_generator, opt_da=None,
                          discriminator=None,**cfg)

    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")