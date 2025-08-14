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

# Import DataModules
from src.data.VBFDNNDataModule import VBFDNNDataModule
from src.data.VBFTransformerDataModule import VBFTransformerDataModule
from src.data.DNNRegressionDataModule import VBFDNNRegressionDataModule
from src.data.TransformerRegressionDataModule import VBFTransformerRegressionDataModule

# Import models
from src.models.TransformerModel import VBFTransformer
from src.models.DNNModel import VBFDNN
from src.models.DNNRegressionModel import VBFDNNRegression
from src.models.TransformerRegression import VBFTransformerRegression
from src.models.ModelArchitectures import ExtraFeatureTransformer

# Model and datamodule dictionaries
g_datamodule_dict = {
    'DNN': VBFDNNDataModule,
    'Transformer': VBFTransformerDataModule, 
    'DNNRegression': VBFDNNRegressionDataModule,
    'TransformerRegression': VBFTransformerRegressionDataModule,
}

g_model_dict = {
    'DNN': VBFDNN,
    'Transformer': VBFTransformer, 
    'DNNRegression': VBFDNNRegression,
    'TransformerRegression': VBFTransformerRegression,
}

# Entry point for the application
@hydra.main(version_base=None, config_path="", config_name="config")
def main(cfg: DictConfig):
    try: 
        # Print the job configuration
        syntax = Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="monokai", line_numbers=False)
        logger.info("Configuration:")
        print(syntax)

        cfg = cfg.BatchConfigs

        
        if cfg.model.type not in ['DNN', 'Transformer', 'DNNRegression', 'TransformerRegression']:
            logger.error("Invalid model type specified in the configuration. Please choose either 'DNN', 'Transformer', 'DNNRegression' or 'TransformerRegression'.")
            raise ValueError()
        
        model = g_model_dict[cfg.model.type] # This is configured inside each step of the pipeline, so just passing the type here.
        datamodule = g_datamodule_dict[cfg.model.type](cfg)

        # Run the different modes based on the configuration
        if cfg.general.mode == 'train':
            train(datamodule, model, cfg)

        if cfg.general.mode == 'predict':
            predict(datamodule, model, cfg)

        if cfg.general.mode == 'performance':
            testing(datamodule, model, cfg)

    except ConfigAttributeError:
        logger.error("Configuration error: Please check your configuration file. Possibly a missing attribute is needed.")
        raise

if __name__ == "__main__":
    main()