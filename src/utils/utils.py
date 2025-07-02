from loguru import logger
import torch
import os

# Set the execution device
def set_exection_device(cfg_device : str):
    if cfg_device == None:
        logger.warning("No device specified in the configuration, defaulting to CPU.")
        return "cpu"

    is_accelerator_available = torch.accelerator.is_available()
    
    # Detect the accelerator name
    accelerator_name = 'cpu'
    if is_accelerator_available:
        logger.info("Accelerator is available, checking the configuration for the device.")
        has_mps = torch.backends.mps.is_available()
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            logger.info("CUDA is available.")
        if has_mps:
            logger.info("MPS is available.")
        if has_cuda and has_mps:
            logger.warning("Both CUDA and MPS are available. Using CUDA by default.")
            accelerator_name = 'cuda'
        elif has_cuda:
            accelerator_name = 'cuda'
        else:
            accelerator_name = 'mps'
    
    if accelerator_name == 'cpu':
        logger.info("No accelerator available, using CPU.")
        return 'cpu'

    allowed_devices = ['cpu', accelerator_name]
    if cfg_device not in allowed_devices:
        logger.error(f"Provided device {cfg_device} is not in the allowed devices {allowed_devices}.")
        raise ValueError()
    
    if accelerator_name != cfg_device:
        logger.warning(f"Using provided device {cfg_device} instead of the detected accelerator {accelerator_name}.")
        logger.warning("You should consider chaging the configuration to speed up the processing.")
    else:
        logger.info(f"Using the provided accelerator {cfg_device}.")

    return cfg_device
    
# Get latest checkpoint path
def get_latest_checkpoint_path(checkpoint_dir):

    # Check if is a directory
    if not os.path.isdir(checkpoint_dir):
        logger.info("Provided a direct file path instead of a directory.")
        # Check if the file exists, if not error out
        if not os.path.isfile(checkpoint_dir):
            logger.error(f"The provided path {checkpoint_dir} is not a file or directory.")
            raise FileNotFoundError()
        else:
            logger.info(f"Using the provided file {checkpoint_dir} as the checkpoint.")
            return checkpoint_dir

    # If it is a directory, find the checkpoint files
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if len(checkpoints) != 1:
        logger.error(f"Expected exactly one checkpoint file in the directory, found {len(checkpoints)}.")
        raise ValueError()
    
    return os.path.join(checkpoint_dir, checkpoints[0])

# Check if a result already exists, given a path, if so, ask the user if they want to overwrite it.
# If not, generate a timestamped version of the path.
def check_and_overwrite_result_path(result_path):
    if os.path.exists(result_path):
        logger.warning(f"Result file {result_path} already exists.")
        logger.info("If you want to overwrite it, please type 'y'. If you want to create a new file, type 'n'.")
        overwrite = input().strip().lower()
        while overwrite not in ['y', 'n']:
            logger.error("Invalid input. Please type 'y' to overwrite or 'n' to create a new file.")
            overwrite = input().strip().lower()
        if overwrite == 'y':
            logger.warning(f"Overwriting the result file {result_path}.")
            return result_path
        else:
            logger.info(f"Creating a new result file with a timestamp. Original file: {result_path}")
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            old_result_name = os.path.basename(result_path)
            old_result_extension = os.path.splitext(old_result_name)[1]
            new_result_name = os.path.join(os.path.dirname(result_path), f"{old_result_name}_{timestamp}.{old_result_extension}")
            return new_result_name
    else:
        logger.info(f"Result file {result_path} does not exist. Creating it.")
        return result_path