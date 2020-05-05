import argparse
import project.localization
import numpy as np



def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--vis', default=True,
                        help='visualization')

def main(args):
    pass


if __name__ == '__main__':
    args = cli()
    main(args)