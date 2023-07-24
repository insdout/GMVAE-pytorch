from utils import get_model, plot_id_history, plot_training_curves
import hydra
from hydra.utils import instantiate
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import logging
from train import Trainer, flatten_mnist


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(config):
    """_summary_

    Args:
        config (dict): Configuration dictionary containing experiment parameters.
    """
    # current hydra output folder
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    log.info(f"Output direcory: {output_dir}")

    model, criterion = get_model(**config.get_model)

    # Set up data loaders
    # Define the transformation
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, **config.dataloaders.train_loader)
    test_loader = DataLoader(dataset=test_dataset, **config.dataloaders.test_loader)

    # Instantiating the optimizer:
    optimizer = instantiate(config.optimizer, params=model.parameters())

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        transform_fn=flatten_mnist,
        path=output_dir,
        **config.trainer.init_params)

    log.info(f"Training started. Epochs: {config.trainer.epochs}")
    trainer.train(config.trainer.epochs)
    log.info("Training finished.")

    history = trainer.history
    ids_history = trainer.ids_history

    plot_training_curves(history, output_dir)
    plot_id_history(ids_history, output_dir)


if __name__ == "__main__":
    main()
