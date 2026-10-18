BASE_FEATURES = (
    "Px",
    "Py",
    "Pz",
    "E",
    "Pt",
)

DENSITY_AND_EVENT_FEATURES = (
    "R02",
    "R05",
    "sumR02",
    "sumR05",
    "sumpT02",
    "sumpT05",
    "w_numPU",
    "TruthMultiplicity",
    "eventPt",
)

def get_feature_names(use_densities):
    features = list(BASE_FEATURES)

    if use_densities:
        features.extend(DENSITY_AND_EVENT_FEATURES)

    return features

