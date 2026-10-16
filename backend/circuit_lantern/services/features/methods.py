"""Method descriptions recorded with every dictionary artifact."""

from circuit_lantern.services.features.autoencoder import TrainingConfig

REFERENCES = [
    {
        "title": "Towards Monosemanticity",
        "authors": "Bricken et al.",
        "year": 2023,
        "url": "https://transformer-circuits.pub/2023/monosemantic-features/index.html",
    },
    {
        "title": "Scaling Monosemanticity",
        "authors": "Templeton et al.",
        "year": 2024,
        "url": "https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html",
    },
    {
        "title": "Scaling and evaluating sparse autoencoders",
        "authors": "Gao et al.",
        "year": 2024,
        "url": "https://arxiv.org/abs/2406.04093",
    },
]


def method_record(config: TrainingConfig) -> dict:
    return {
        "implementation": "circuit_lantern_sae_v1",
        "capture": "Raw residual stream immediately after the selected transformer block.",
        "preprocessing": "Divide by one RMS scalar fitted to training activations only. Subtract a learned centering bias shared by the encoder input and decoder output.",
        "initialization": "Seeded random unit decoder directions, transposed initial encoder directions, and a geometric median centering bias fitted on training activations.",
        "activation": "ReLU"
        if config.method == "relu"
        else "ReLU of the largest k encoder preactivations. At most k features are active.",
        "objective": (
            "Mean squared reconstruction error per coordinate in normalized activation units plus the L1 coefficient times mean per-token feature L1."
            if config.method == "relu"
            else "Mean squared reconstruction error in normalized units divided by a fixed training centered variance, plus a residual-normalized AuxK loss on inactive features."
        ),
        "optimizer": {
            "name": "Adam",
            "betas": [0.9, 0.999],
            "epsilon": 6.25e-10,
            "gradient_norm_limit": 1.0,
            "decoder_gradient": "Remove the component parallel to each unit decoder column before the optimizer step.",
            "decoder_normalization": "Normalize every decoder column after each optimizer step.",
            "batching": "Seeded shuffled full batches. Incomplete epoch tails are reshuffled. A corpus smaller than a batch uses every training row per step.",
        },
        "auxiliary": (
            {
                "coefficient": config.auxiliary_coefficient,
                "inactive_after_optimizer_token_exposures": config.dead_after_tokens,
                "k": "Logarithmically nearest power of two to half the input width, capped by the inactive feature count.",
                "scale_adaptation": "The default inactivity window is shortened for interactive runs. Repeated optimizer exposures and unique captured tokens are reported separately.",
            }
            if config.method == "topk"
            else None
        ),
        "reproducibility": "Initialization and sample order use local seeded generators. CUDA kernels and tied TopK scores may not be bitwise reproducible across devices or runtime versions.",
        "relu_neuron_resampling": False,
        "exponential_weight_averaging": False,
        "evaluation": "MSE uses original activation units per coordinate. Normalized error divides total squared error by the error of the training mean on the same evaluation set. Explained variance is one minus that ratio and can be negative. A zero baseline variance remains unavailable.",
        "scope": "Interactive dictionary learning on the recorded corpus. Reconstruction and sparsity metrics alone do not establish a semantic interpretation or a circuit.",
        "references": REFERENCES,
    }
