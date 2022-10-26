import os
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--result', '-r',
                        type=str,
                        default=None,
                        help='Configuration file to use')
    args = parser.parse_args()

    