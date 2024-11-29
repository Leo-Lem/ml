from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--blank', action='store_true',
                    help='Blank model instead of pretrained')
parser.add_argument('--include_part_deriv', action='store_true',
                    help='Include partial derivatives in the model (only for pretrained)')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Batch size for training')
parser.add_argument('--debug', action='store_true',
                    help='Run in debug mode')
args = parser.parse_args()

BLANK = args.blank
INCLUDE_PART_DERIV = args.include_part_deriv if not BLANK else False
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
DEBUG = args.debug
