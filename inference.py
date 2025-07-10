import os
import torch
import yaml
import argparse
from utils.logger import set_logging
import model
from data import create_datasets
from metrics import Evaluator
from data import create_datasets, AfifiAWBDataset, ColorConstancyDataset, CubePlusDataset

def main(cfg: dict):
    # Prepare logging
    logger, log_dirpath = set_logging()

    # Create model
    net = model.TransformerWB(device=cfg['device'], P=cfg['model']['P'])
    net.to(cfg['device'])
    net.load_state_dict(torch.load(cfg['model']['ckpt_path']))
    logger.info(f'Model created with {str(sum([p.numel() for p in net.parameters() if p.requires_grad]))} parameters')
    logger.info(f"Model loaded from {cfg['model']['ckpt_path']}")

    _, _, test_dataset = create_datasets(data_args=cfg['dataset_settings']['data'])
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)
    evaluator = Evaluator(criterion=None, data_loader=test_dataset, split_name='test', log_dirpath=log_dirpath)

    _ = evaluator.evaluate(net, 'test')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the CURL neural network on image pairs")

    parser.add_argument(
        "--config_path", "-c", help="The location of curl config file", default='./config.yaml')
    parser.add_argument(
        "--device", "-d", help="Device to use", default='0')
    parser.add_argument(
        "--ckt_path", "-ckpt", help="Path to the checkpoint to load")
 
    args = parser.parse_args()
    config_path = args.config_path

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    else:
        assert args.device == 'cpu', "Only CPU is available"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    config_file.update({'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
    if config_file['model']['ckpt_path'] is None and args.ckt_path is None:
        raise ValueError("No checkpoint path provided")
    
    config_file['model']['ckpt_path'] = args.ckt_path
    main(config_file)