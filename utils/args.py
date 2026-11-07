import argparse
from typing import DefaultDict


def get_args():

    parser = argparse.ArgumentParser()

    # directories
    parser.add_argument(
        "--path",
        type=str,
        default="data",
        help="Base directory containing data folders (MC-overlay_{sample}/ and Track-overlay_{sample}/). Default: 'data'",
    )
    # TODO: Eventually users would only be expected to pass the path, and not the sample name separately.
    parser.add_argument(
        "-sn",
        "--sample",
        type=str,
        default="JZ7W",
        help="This is the sample that will be used",
    )

    # Reproducibiltiy
    parser.add_argument(
        "-s", "--seed", type=int, default=23, help="Random seed for consistency"
    )

    parser.add_argument(
        "--downsample",
        action="store_true",
        default=False,
        help="Randomly downsample the NoDiff training sample to the size of the Diff training sample.",
    )
    
    # Stats
    parser.add_argument(
        "--trainsplit",
        type=float,
        default=0.7,
        help="Fraction of full dataset used for training; rest for held-out test.",
    )

    # Model
    parser.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=[45,35,30],
        help="This is the layer definition for the classifier",
    )
    parser.add_argument(
        "-lr", type=float, default=0.001, help="Learning rate for the optimiser"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train the classifier",
    )
    parser.add_argument(
        "-bs", "--batchsize", type=int, default=80, help="Size of batches"
    )
    parser.add_argument(
        "-p",
        "--patience",
        type=int,
        default=20,
        help="Number of epochs over which the classifier will be\
        monitored for change",
    )

    parser.add_argument(
        "-rltr",
        "--rouletter",
        type=str,
        default="smart",
        help="Type of roulette to be used. Defaults to hard cuts.",
    )

    parser.add_argument(
        "-temp",
        "--temperature",
        type=float,
        default=0.00005,
        help="Amount of smoothing to be applied for the roulette. Used if rouletter is set to smart.",
    )


    parser.add_argument(
        "-xs",
        "--xscore",
        type=str,
        default="simpleratio",
        help="definition of the roulette score.",
    )

    parser.add_argument(
        "-em",
        "--evtmix",
        type=str,
        default="None",
        help="How to combine the density feature?",
    )
    # Data:
    parser.add_argument(
        "-ud",
        "--usedensity",
        type=int,
        default=1,
        help="Use del R densities as features for training.",
    )
    
    # Evaluation:
    parser.add_argument(
        "--eval_sample",
        type=str,
        default=None,
        help="Sample to use for evaluation (if different from training sample). Defaults to --sample.",
    )

    return parser.parse_args()
