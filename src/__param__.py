from argparse import ArgumentParser
from os import path

parser = ArgumentParser()
parser.add_argument("--blank", action="store_true",
                    help="use blank model instead of pretrained")
parser.add_argument("--part_deriv", action="store_true",
                    help="include partial derivatives in the model (only for pretrained)")
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
args = parser.parse_args()

BLANK = args.blank
INCLUDE_PART_DERIV = False if BLANK else args.part_deriv
EPOCHS = args.epochs
STOP_EARLY_AFTER = args.stop_early_after
BATCH_SIZE = args.batch_size
DEBUG = args.debug
SAMPLE = args.sample
TRAIN = not args.skip_training

OUT = path.join(path.dirname(path.dirname(__file__)), ".out")
