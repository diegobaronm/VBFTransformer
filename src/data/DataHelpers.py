import h5py
import torch
import numpy as np
import numpy.lib.recfunctions as rfn

from itertools import combinations
from sklearn.utils import resample
from sklearn.neighbors import KernelDensity
from torch.utils.data import WeightedRandomSampler

from src.data.DataFormat import particle_feature_dict, MetadataIndex


def read_h5_data(file_name, max_features=42):
    with h5py.File(file_name, "r") as file:
        particles = file["INPUTS"]["PARTICLES"][:, :max_features]
        metadata = file["METADATA"]["EVENT_DATA"][:]

        signal_inputs = rfn.structured_to_unstructured(particles)
        particles_keys = list(file["INPUTS"]["PARTICLES"].dtype.names)[:max_features]
        input_keys_dict = {key: idx for idx, key in enumerate(particles_keys)}

    return signal_inputs, metadata, input_keys_dict

def load_multiple_h5(files, max_features=42):
    results = [read_h5_data(f, max_features) for f in files]
    signals, metadata, keys_dicts = zip(*results)

    combined_signals = np.vstack(signals)
    combined_metadata = np.vstack(metadata)

    return combined_signals, combined_metadata, keys_dicts[0]

def load_and_filter_data(files):
    signal_inputs, metadata_inputs, _ = load_multiple_h5(files)
    valid_mask = metadata_inputs[:, MetadataIndex.MMC_MZ.value] != 0
    return signal_inputs[valid_mask], metadata_inputs[valid_mask]

        
def flat_inputs(input_array, n_particles, feature_indices=None):
    if feature_indices is None:
        feature_indices = list(range(input_array.shape[2]))  # Use all features if none specified
        
    output_array = input_array[:, :n_particles, feature_indices]
    
    # Transpose to (samples, features, particles)
    output_array = output_array.transpose(0, 2, 1)
    
    # Flatten to (samples, n_particles * n_features_selected)
    return output_array.reshape(-1, n_particles * len(feature_indices))

def add_features(starting_df, arr_of_features):
    data_df = starting_df

    for feature in arr_of_features:
        new_feature = feature.reshape(-1, 1)
        data_df = np.hstack((data_df, new_feature))

    return data_df


def prepare_input_data(particles, metadata, n_particles, features, extra_features, combine=True):
    base = flat_inputs(particles, n_particles, features)
    extras = [metadata[:, i] for i in extra_features]

    if combine: 
        return np.array(add_features(base, extras))
    else: 
        return np.array(base), np.array(extras)

    
def get_particle_feature_index_ranges(feature_names, n_particles):
    feature_index_ranges = {}
    current_index = 0
    for name in feature_names:
        indices = list(range(current_index, current_index + n_particles))
        feature_index_ranges[name] = indices
        current_index += n_particles
    return feature_index_ranges
    
def get_full_feature_index_ranges(particle_feature_names, extra_feature_names, n_particles):
    feature_index_ranges = get_particle_feature_index_ranges(particle_feature_names, n_particles)
    current_index = len(particle_feature_names) * n_particles
    for name in extra_feature_names:
        feature_index_ranges[name] = [current_index]
        current_index += 1
    return feature_index_ranges

def to_float_tensor(X,  NAN_PAD_VALUE=-1):
    return torch.nan_to_num(torch.tensor(X, dtype=torch.float32), nan=NAN_PAD_VALUE)

def calculate_KDE_sampler(y_train, KDE_width, Min_Dens_cap):
    # Ensure NumPy array
    if isinstance(y_train, torch.Tensor):
        y_train_np = y_train.cpu().numpy()
    else:
        y_train_np = np.array(y_train)

    # Fit KDE on a resampled subset of y_train
    # Note for future devs: using multiple threads to compute kde to speed it up runs into sampling issues easily
    if len(y_train_np) > 30_000:
        idx = np.random.choice(len(y_train_np), size=30_000, replace=False)
        y_kde_train = y_train_np[idx]
    else:
        y_kde_train = y_train_np
        
    kde = KernelDensity(kernel='gaussian', bandwidth=KDE_width).fit(y_kde_train.reshape(-1, 1))

    log_dens = kde.score_samples(y_train_np.reshape(-1, 1))
    density = np.exp(log_dens)

    # Inverse density weighting
    inv_density = 1.0 / (density + 1e-8)
    inv_density = np.minimum(inv_density, Min_Dens_cap)

    # Normalize to get probabilities
    weights = inv_density / np.sum(inv_density)

    # Convert to tensor for sampler
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)

    return density, sampler


# Calculates pair-wise interaction tokens as first defined in DOI: 10.1088/1674-1137/ad7f3d
def calculate_interaction_inputs(tensor_data, N_particles): 
    # Expecting data [event : feature_particle
    # Indices are: energy, eta, phi, pt
    N_events, _, _ = tensor_data.shape

    # Generate all unique (i, j) pairs with i < j
    pair_indices = list(combinations(range(N_particles), 2))  # e.g. [(0,1), (0,2), (1,2), ...]
    N_pairs = len(pair_indices)

    # Prepare output arrays
    DELTAs = np.zeros((N_events, N_pairs))
    K_Ts = np.zeros((N_events, N_pairs))
    Zs = np.zeros((N_events, N_pairs))
    M2s = np.zeros((N_events, N_pairs))

    particle_feature_dict = pdf

    # Loop over each pair
    for idx, (i, j) in enumerate(pair_indices):
        # Extract features
        E_i, eta_i, phi_i, pt_i = tensor_data[:, i, pdf['energy']], tensor_data[:, i, pdf['eta']], tensor_data[:, i, pdf['phi']], tensor_data[:, i, pdf['pt']]
        E_j, eta_j, phi_j, pt_j = tensor_data[:, j, pdf['energy']], tensor_data[:, j, pdf['eta']], tensor_data[:, j, pdf['phi']], tensor_data[:, j, pdf['pt']]

        # Cartesian 3-momentum
        px_i, py_i, pz_i = pt_i * np.cos(phi_i), pt_i * np.sin(phi_i), pt_i * np.sinh(eta_i)
        px_j, py_j, pz_j = pt_j * np.cos(phi_j), pt_j * np.sin(phi_j), pt_j * np.sinh(eta_j)

        # 3D vectors: shape (N_events, 3)
        p_vec_i = np.stack([px_i, py_i, pz_i], axis=1)
        p_vec_j = np.stack([px_j, py_j, pz_j], axis=1)

        # Rapidity
        y_i = 0.5 * np.log((E_i + pz_i) / (E_i - pz_i))
        y_j = 0.5 * np.log((E_j + pz_j) / (E_j - pz_j))

        # ΔR
        delta = np.sqrt((y_i - y_j)**2 + (phi_i - phi_j)**2)
        pt_min = np.minimum(pt_i, pt_j)
        pt_sum = pt_i + pt_j

        # K_T, Z
        K_T = pt_min * delta
        Z = pt_min / pt_sum

        # Invariant mass squared M2
        p_total = p_vec_i + p_vec_j  # shape (N_events, 3)
        p_mag = np.linalg.norm(p_total, axis=1)
        M2 = (E_i + E_j)**2 - p_mag**2

        # Log transforms as described in the paper
        DELTAs[:, idx] = np.log(delta)
        K_Ts[:, idx] = np.log(K_T)
        Zs[:, idx] = np.log(Z)
        M2s[:, idx] = np.log(M2)


    tokens = np.stack([DELTAs, K_Ts, Zs, M2s], axis=-1)
    return tokens, pair_indices