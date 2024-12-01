from argparse import ArgumentParser
from os import path

parser = ArgumentParser()
parser.add_argument("--approach", type=str,
                    help="approach to use for training the model (blank, pretrained, pretrained_part_deriv)")
parser.add_argument("--path", type=str, default="",
                    help="path to load/store the trained model")

parser.add_argument("--epochs", type=int, default=10,
                    help="number of epochs to train the model")
parser.add_argument("--stop_early_after", type=int, default=2,
                    help="number of epochs without improvement to stop training")
parser.add_argument("--batch_size", type=int, default=1024,
                    help="batch size for training")

parser.add_argument("--debug", action="store_true",
                    help="run in debug mode")
parser.add_argument("--sample", action="store_true",
                    help="use sample data instead of full data")
parser.add_argument("--skip_training", action="store_true",
                    help="skip training the model")
parser.add_argument("--clear", action="store_true",
                    help="clear the cache and output directories")
args = parser.parse_args()

OUT = path.join(path.dirname(path.dirname(__file__)), ".out")
MODEL_PATH = args.path if args.path else OUT

BLANK = args.approach == "blank"
INCLUDE_PART_DERIV = args.approach == "pretrained_part_deriv"

EPOCHS = args.epochs
STOP_EARLY_AFTER = args.stop_early_after
BATCH_SIZE = args.batch_size

DEBUG = args.debug
SAMPLE = args.sample
TRAIN = not args.skip_training

if args.clear:
    from shutil import rmtree
    rmtree(OUT, ignore_errors=True)
    if DEBUG:
        print("Cleared cache and output directories.")
