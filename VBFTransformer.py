# Hydra - for CLI configuration management
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError

# For pretty printing
from rich import print
from rich.syntax import Syntax
from loguru import logger

# Import local modules
from src.utils.Train import train
from src.utils.Predict import predict
from src.utils.Performance import testing
from src.data.VBFTransformerDataModule import VBFTransformerDataModule

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    try: 
        # Print the job configuration
        syntax = Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="monokai", line_numbers=False)
        logger.info("Configuration:")
        print(syntax)

        # Load the data module
        datamodule = VBFTransformerDataModule(cfg.dataset.signal_path, cfg.dataset.background_path, n_particles=cfg.model.n_particles)

        # Run the different modes based on the configuration
        if cfg.general.mode == 'train':
            train(datamodule, cfg)

        if cfg.general.mode == 'predict':
            predict(datamodule, cfg)

        if cfg.general.mode == 'performance':
            testing(datamodule, cfg)

    except ConfigAttributeError:
        logger.error("Configuration error: Please check your configuration file. Possibly a missing attribute is needed.")
        raise

if __name__ == "__main__":
    # Run
    main()