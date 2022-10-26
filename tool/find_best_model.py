import os
import shutil
import json
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--experiment', '-e',
                        type=str,
                        default=None,
                        help='Experiment Name')
    parser.add_argument('--find_only', '-f',
                        action='store_true',
                        help='No need to save the best models')
    args = parser.parse_args()

    result_path = os.path.join("results", args.experiment)
    trainer_state_path = os.path.join(result_path, "trainer_state.json")

    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)
    
    log_history = trainer_state["log_history"]
    epoch = int(trainer_state["epoch"])

    min_eval_loss = [10000, None, None]
    max_f1_score = [0, None, None]
    for e in range(epoch):
        eval_log = log_history[e * 2 + 1]
        if eval_log["eval_loss"] < min_eval_loss[0]:
            min_eval_loss = [eval_log["eval_loss"], eval_log["step"], e + 1]
        
        if eval_log["eval_f1"] > max_f1_score[0]:
            max_f1_score = [eval_log["eval_f1"], eval_log["step"], e + 1]
    
    if not args.find_only:
        print("Save best models")
        shutil.copytree(os.path.join(result_path, f"checkpoint-{min_eval_loss[1]}"), os.path.join(result_path, "best_loss_model"))
        shutil.copytree(os.path.join(result_path, f"checkpoint-{max_f1_score[1]}"), os.path.join(result_path, "best_f1_model"))
    
    print(f"Best f1 score: {max_f1_score[0]}, at step {max_f1_score[1]}, epoch {max_f1_score[2]}")
    print(f"Min eval loss: {min_eval_loss[0]}, at step {min_eval_loss[1]}, epoch {min_eval_loss[2]}")