import h5py               
import numpy as np          
import numpy.lib.recfunctions as rfn

def read_h5_data(file_name):
    with h5py.File(file_name, 'r') as file:
        particles = file['INPUTS']['PARTICLES'][:, :42]  # First 43 columns/features
        metadata = file['METADATA']['EVENT_DATA'][:]

        signal_inputs = rfn.structured_to_unstructured(particles)

        particles_keys = list(file['INPUTS']['PARTICLES'].dtype.names)[:42]
        input_keys_dict = {key: idx for idx, key in enumerate(particles_keys)}

    return signal_inputs, metadata, input_keys_dict

def load_multiple_h5(files):
    all_signals = []
    all_metadata = []
    input_keys_dict = None

    for f in files:
        signals, metadata, keys_dict = read_h5_data(f)
        all_signals.append(signals)
        all_metadata.append(metadata)
        if input_keys_dict is None:
            input_keys_dict = keys_dict

    combined_signals = np.vstack(all_signals)
    combined_metadata = np.vstack(all_metadata)

    return combined_signals, combined_metadata, input_keys_dict


def flat_inputs(input_array, n_particles, feature_indices=None):
    if feature_indices is None:
        feature_indices = list(range(input_array.shape[2]))  # Use all features if none specified

    # Select specific particles and features
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