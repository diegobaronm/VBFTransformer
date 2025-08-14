# This file is used for general formatting of the data inputs, here you need to specify which indices hold which variables in 
# the input files. This is done such that in the config files, variables can be input out of order, and the model can simply
# use the lookup tables below and still obtain correct results.
from enum import Enum  

class MetadataIndex(Enum):
    # EVENT_NUMBER = 0 No need to assing this 
    OMEGA = 1
    MZ_RECO = 2
    MZ_TRUTH = 3
    MMC_MZ = 4
    OPENING_ANGLE = 5
    LEP_MET_ANGLE_SIGNED = 6
    IS_MET_INSIDE = 7
    IS_EVENT_TPG_HADEL = 8
    TAU_GOOD_ID = 9
    TAU_GOOD_N_TRACKS = 10
    TAU_NN_DECAY = 11
    THREE_BODY_TRAN_MASS = 12
    M3_STAR = 13

particle_feature_dict = {
    'energy': 0,
    'eta': 1,
    'phi': 2,
    'pt': 3,
    'btag': 4,
    'charge': 5,
    'type': 6
}

extra_feature_dict = {
    'omega': MetadataIndex.OMEGA.value, 
    'mz_reco': MetadataIndex.MZ_RECO.value, 
    'mz_mmc': MetadataIndex.MMC_MZ.value, 
    'opening_angle': MetadataIndex.OPENING_ANGLE.value, 
    'lep_met_angle_signed': MetadataIndex.LEP_MET_ANGLE_SIGNED.value, 
    'is_met_inside': MetadataIndex.IS_MET_INSIDE.value, 
    'is_event_tpg_hadel': MetadataIndex.IS_EVENT_TPG_HADEL.value, 
    'tau_good_id': MetadataIndex.TAU_GOOD_ID.value, 
    'tau_good_n_tracks':  MetadataIndex.TAU_GOOD_N_TRACKS.value,
    'tau_nn_decay':  MetadataIndex.TAU_NN_DECAY.value,
    'transverse_mass': MetadataIndex.THREE_BODY_TRAN_MASS.value,
    'm3_star': MetadataIndex.M3_STAR.value,
}

pretty_label_dict = {
    'energy': r"$E$",
    'eta': r"$\eta$",
    'pt': r"$p_{T}$",
    'phi': r"$\phi$",
    'btag': r"btag", 
    'charge': r"charge",
    'type': r"type",
    'phi__cos': r"$\cos(\phi)$",
    'phi__sin': r"$\sin(\phi)$",
}
    
extra_feature_label_dict = {
    'opening_angle':         r"$\Delta \phi_U(lep, tau)$",
    'lep_met_angle_signed': r"$\Delta \phi_S(lep, MET)$",
    'omega': r"$\Omega$", 
    'mz_reco': r"Mz-reco",
    'mz_mmc': r"Mz-mmc",   
    'is_event_tpg_hadel': r"Is Event TPG HadEL", 
    'tau_good_id': r"Tau Good ID", 
    'tau_good_n_tracks': r"Tau Good N Tracks", 
    'tau_nn_decay': r"Tau NNDecay Mode",
    'transverse_mass':r"Transverse Mass",
    'm3_star': r"$M_3^*$",
}
