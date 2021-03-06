import argparse
from pathlib import Path
import json
import shutil
import sys
from termcolor import colored, cprint
from pprint import pprint
import pdb

def command_interface(title=None):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--config', '-cf', default=None, help='training configs json file')
    parser.add_argument('--devices', '-d', nargs='+', default=None, type=int, help='CUDA devices. Use CPU if None')
    parser.add_argument('--rand_seed', '-r', default=1, type=int, help='random seed initialization')
    # parser.add_argument('--name', '-n', default='exp', help='name of this experiment')
    parser.add_argument('--lr', '-lr', default=0.04, help='learning_rate')
    parser.add_argument('--mode', '-m', default='new', choices=['new', 'resume', 'test', 'pretrained', 'finetune'], help='running mode')
    parser.add_argument('--iters', '-i', default=1, type=int, help='number of iterations to run the experiment')
    parser.add_argument('--omniscient', '-o', action='store_true', help='if specified, set validation set = test set')
    parser.add_argument('--overwrite', '-ow', action='store_true', help='if specified, overwrite existing folder without asking')
    parser.add_argument('--workers', '-w', default=12, type=int, help='number of workers for the dataloader')
    parser.add_argument('--amp', '-a', action='store_true', help='if specified, turn amp on')
    # parser.add_argument('--eval_sel', '-es', default=None, type=str, help='at test mode, select eval1 or eval2')
    #for consistency_loss
    parser.add_argument('--p_cutoff', type=float, default=0.95)
    parser.add_argument('--temperature', type=float, default=0.5)
    args = parser.parse_args()
    pprint(vars(args))

    config = json.load(open(args.config))
    #save_root_name comes_from config_name
    args.name = args.config.split('/')[-1].replace(".json","")+ f'[{args.lr}]'
    config['train']['lr'] = float(args.lr)
    assert config['train']['lr'] == float(args.lr)
    save_root = Path('weights')/args.name

    # Deprecated
    if args.mode == 'finetune':
        fine_tune_folder = args.name +"_finetune"
        if Path(save_root).exists():
            fine_tune_folder = fine_tune_folder + "_" + str(int(fine_tune_folder[-1]) + 1)
        save_root = Path('weights')/fine_tune_folder

    if args.mode == 'new' and Path(save_root).exists():

        if not args.overwrite and args.name != 'exp':
            txt = input(colored(f'[WARNING] {save_root} exists. Overwrite [Y/N]? ', color='yellow', attrs=['bold']))
        else:
            txt = 'y'

        if txt.lower() == 'y':
            cprint(f'Overwrite {save_root} folder...', color='yellow', attrs=['bold'])
            shutil.rmtree(save_root)
        else:
            cprint('Abort...', color='yellow', attrs=['bold'])
            sys.exit()

    return args, config, save_root


if __name__ == '__main__':
    command_interface()
