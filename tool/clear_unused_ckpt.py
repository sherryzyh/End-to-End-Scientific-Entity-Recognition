import os
import shutil
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--experiment', '-e',
                        type=str,
                        default=None,
                        help='Configuration file to use')
    args = parser.parse_args()

    source = os.path.join("results", args.experiment)
    dest_root = "results/confirmed/"
    dest = os.path.join(dest_root, args.experiment)

    if not os.path.exists(dest):
        os.mkdir(dest)

    for file in os.listdir(source):
        if not os.path.isdir(os.path.join(source, file)):
            shutil.copy2(os.path.join(source, file), dest)
            continue

        if file == "best_f1_model":
            if not os.path.exists(os.path.join(dest, file)):
                shutil.copytree(os.path.join(source, file), os.path.join(dest, file))
            continue
            
        if file == "best_loss_model":
            if not os.path.exists(os.path.join(dest, file)):
                shutil.copytree(os.path.join(source, file), os.path.join(dest, file))
            continue

    shutil. rmtree(source)