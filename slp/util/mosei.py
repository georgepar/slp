# this script contains custom argument parsers
# - dataset: [mosei, mosi]
# - models: [rnn, transformer]
from argparse import ArgumentParser

def get_mosei_parser():
    parser = ArgumentParser("Transformer-MOSEI example")

    # model hyperparameters
    parser.add_argument(
        "--max_length",
        dest="model.max_length",
        type=int,
        default=1024,
        help="Sequence max length for transformer layers",
    )
    parser.add_argument(
        "--kernel_size",
        dest="model.kernel_size",
        type=int,
        default=33,
        help="Kernel Size used for residual trick in transformer layers",
    )
    parser.add_argument(
        "--use_nystrom",
        dest="model.nystrom",
        default=False,
        action="store_true",
        help="Use nystrom self-attention approximation",
    )
    parser.add_argument(
        "--num_landmarks",
        dest="model.num_landmarks",
        type=int,
        default=32,
        help="Number of landmarks used for nystrom approximation",
    )
    parser.add_argument(
        "--num_layers",
        dest="model.num_layers",
        type=int,
        default=3,
        help="Number of Transformer Layers",
    )
    parser.add_argument(
        "--num_heads",
        dest="model.num_heads",
        type=int,
        default=4,
        help="Number of Transformer Heads per Layer",
    )
    parser.add_argument(
        "--dropout",
        dest="model.dropout",
        type=float,
        default=0.3,
        help="Drop probability for each Transformer Layer",
    )
    parser.add_argument(
        "--hidden_size",
        dest="model.hidden_size",
        type=int,
        default=100,
        help="Hidden Size for each Transformer Layer",
    )
    parser.add_argument(
        "--inner_size",
        dest="model.inner_size",
        type=int,
        default=200,
        help="Inner Size for each Transformer Layer",
    )
    parser.add_argument(
        "--prenorm",
        dest="model.prenorm",
        default=False,
        action="store_true",
        help="Use the normalization before residual."
        "Default value is False as in Vaswani.",
    )
    parser.add_argument(
        "--scalenorm",
        dest="model.scalenorm",
        default=False,
        action="store_true",
        help="Use scalenorm (L2) instead of LayerNorm (Vaswani).",
    )
    parser.add_argument(
        "--mmdrop",
        dest="model.multi_modal_drop",
        type=str,
        default="dropout",
        choices=["dropout", "mmdrop_hard", "mmdrop_soft", "none"],
        help="Which dropout is applied to the late fusion stage",
    )
    parser.add_argument(
        "--p-mmdrop",
        dest="model.p_mmdrop",
        type=float,
        default=0.33,
        help="probability of droping 1/3 modlities",
    )
    parser.add_argument(
        "--p-drop-modalities",
        dest="model.p_drop_modalities",
        default=[0.33, 0.33, 0.33],
        help="Per modal drop rate",
    )

    # dataset specifications
    parser.add_argument(
        "--pad-front",
        dest="preprocessing.pad_front",
        action="store_true",
        help="Handles front padding. Default is True",
    )
    parser.add_argument(
        "--pad-back",
        dest="preprocessing.pad_back",
        action="store_true",
        help="Handles back padding. Default is False",
    )
    parser.add_argument(
        "--remove-pauses",
        dest="preprocessing.remove_pauses",
        action="store_true",
        help="When used removes pauses from dataset",
    )
    parser.add_argument(
        "--crop-length",
        dest="preprocessing.max_length",
        default=-1,
        type=int,
        help="Crop feature sequences during data ingestion from cmusdk"
    )
    parser.add_argument(
        "--already-unaligned",
        dest="preprocessing.already_aligned",
        action="store_true",
        help="When used indicates unaligned data otherwise"
        " aligned scenario is used.",
    )
    parser.add_argument(
        "--do-align",
        dest="preprocessing.align_features",
        action="store_true",
        help="When used automatically aligns the features.",
    )

    return parser


