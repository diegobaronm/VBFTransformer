# Hydra - for CLI configuration management
import hydra
from omegaconf import DictConfig, OmegaConf

# Import local modules
from src.utils.Train import train
from src.utils.Predict import predict
from src.utils.Performance import testing
from src.data.VBFTransformerDataModule import VBFTransformerDataModule

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    datamodule = VBFTransformerDataModule(cfg.dataset.signal_path, cfg.dataset.background_path, n_particles=7)

    if cfg.opts.mode == 'train':
        train(datamodule, cfg)

    if cfg.opts.mode == 'predict':
       predict(datamodule)

    if cfg.opts.mode == 'performance':
        testing(datamodule)

if __name__ == "__main__":
    # Run
    main()