"""Registry mapping experiment names to their concrete classes."""

from experiments.baseline import BaselineExperiment
from experiments.docel import (
    DocElExperiment,
    DocElCotV1Experiment,
    DocElCotV2Experiment,
    DocElCotV3Experiment,
    DocElCotV4Experiment,
    DocElCotNumVREExperiment,
)
from experiments.nlp_tag import NLPTagExperiment, NLPTagCotExperiment
from experiments.nlp_list import (
    NLPListExperiment,
    NLPListCotExperiment,
    NLPListOCRExperiment,
    NLPListOCRCotExperiment,
)
from experiments.layout import (
    LayoutV1Experiment,
    LayoutV2Experiment,
    LayoutV3Experiment,
    LayoutV4Experiment,
)

EXPERIMENTS = {
    # Baseline
    "baseline": BaselineExperiment,
    "baseline_ocr": BaselineExperiment,  # Uses the same class, requires ocr_enabled=True in config
    
    # DocEl
    "docel": DocElExperiment,
    "docel_cot_v1": DocElCotV1Experiment,
    "docel_cot_v2": DocElCotV2Experiment,
    "docel_cot_v3": DocElCotV3Experiment,
    "docel_cot_v4": DocElCotV4Experiment,
    "docel_cot_numvre": DocElCotNumVREExperiment,
    
    # NLP Tag
    "nlp_tag": NLPTagExperiment,
    "nlp_tag_cot": NLPTagCotExperiment,
    
    # NLP List
    "nlp_list": NLPListExperiment,
    "nlp_list_cot": NLPListCotExperiment,
    "nlp_list_ocr": NLPListOCRExperiment,
    "nlp_list_ocr_cot": NLPListOCRCotExperiment,
    
    # Layout
    "layout_v1": LayoutV1Experiment,
    "layout_v2": LayoutV2Experiment,
    "layout_v3": LayoutV3Experiment,
    "layout_v4": LayoutV4Experiment,
}

def get_experiment_class(name):
    """Retrieve the experiment class by name."""
    if name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment: '{name}'. "
            f"Available experiments: {', '.join(EXPERIMENTS.keys())}"
        )
    return EXPERIMENTS[name]
