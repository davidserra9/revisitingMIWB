import os
import torch.utils.data
import yaml
from logger import set_logging
import argparse
from model import TransformerWB
from data import create_datasets
import metrics
from glob import glob
import omegaconf

def main(cfg: dict):
    # Prepare logging
    logger, log_dirpath = set_logging()

    # Create model
    net = TransformerWB(device=cfg['device'])

    if cfg['model']['pretrained'] is not None:
        net.load_state_dict(torch.load(cfg['model']['pretrained']))

    net.to(cfg['device'])
    logger.info(f'Model created with {str(sum([p.numel() for p in net.parameters() if p.requires_grad]))} parameters')

    # Print config file
    logger.info(f"Config file: {omegaconf.OmegaConf.to_yaml(cfg)}")

    # Create Loss Function and Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3, betas=(0.9, 0.999),
                               eps=1e-08)

    # Create training datasets and dataloaders
    train_dataset, val_dataset, test_dataset = create_datasets(data_args=cfg['dataset_settings']['data'])

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=cfg['dataset_settings']['train_batch_size'],
                                                   num_workers=cfg['dataset_settings']['num_workers'],
                                                   shuffle=True)

    # Create LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=500 * len(train_dataloader),
                                                           eta_min=1e-5)
    
    # Create validation and test evaluators
    if val_dataset is not None:
        valid_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=cfg['dataset_settings']['valid_batch_size'],
                                                     num_workers=cfg['dataset_settings']['num_workers'],
                                                     shuffle=False)

        valid_evaluator = metrics.Evaluator(criterion=criterion,
                                            data_loader=valid_dataloader,
                                            split_name='valid',
                                            log_dirpath=log_dirpath)

    else:
        valid_evaluator = None

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=cfg['dataset_settings']['valid_batch_size'],
                                                  num_workers=cfg['dataset_settings']['num_workers'],
                                                  shuffle=False)

    test_evaluator = metrics.Evaluator(criterion=criterion,
                                       data_loader=test_dataloader,
                                       split_name='valid',
                                       log_dirpath=log_dirpath)

    best_valid_de = 1e10

    for epoch in range(cfg['training_settings']['epochs']):
        running_loss = 0.0
        for batch_num, data_dict in enumerate(train_dataloader):
            output_img = net(data_dict)
            loss = criterion(output_img, data_dict['target'].to(cfg['device']))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch < 500:
                scheduler.step()

            running_loss += loss.item()

        logger.info(f"Epoch {epoch} loss: {running_loss / len(train_dataset):.10f} - lr: {scheduler.get_last_lr()[0]:.10f}")


        if epoch % cfg['training_settings']['valid_every'] == 0:
            valid_metrics = valid_evaluator.evaluate(net, split="valid", epoch=epoch)
            valid_de = valid_metrics['valid/e00_mean']

            if valid_de < best_valid_de:
                print(f'Validation Delta E00 improved: {valid_de:.2f}')

                previous_models = glob(os.path.join(log_dirpath, f'MIWB_*.pth'))
                if len(previous_models) == 1:
                    os.remove(previous_models[0])
                elif len(previous_models) > 1:
                    for model_path in previous_models:
                        os.remove(model_path)

                logger.info(f"Previous models removed.")

                for path in previous_models:
                    logger.info(f"Removed model: {path}")

                best_valid_de = valid_de
                test_metrics = test_evaluator.evaluate(net, split="test", epoch=epoch)
                test_de = test_metrics['test/e00_mean']

                save_path = os.path.join(log_dirpath, f'MIWB_testde00_{test_de:.2f}_epoch_{epoch}.pth')
                
                torch.save(net.state_dict(), save_path)
                logger.info(f"Model saved @ {save_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the CURL neural network on image pairs")

    parser.add_argument(
        "--config_path", "-c", help="The location of curl config file", default='./config.yaml')
    parser.add_argument(
        "--device", "-d", help="Device to use", default='0')

    args = parser.parse_args()
    config_path = args.config_path

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    else:
        assert args.device == 'cpu', "Only CPU is available"

    with open(config_path) as file:
        config_file = yaml.full_load(file)

    config_file.update({'device': 'cuda' if torch.cuda.is_available() else 'cpu'})

    main(config_file)
