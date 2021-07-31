import argparse
import pdb
from pathlib import Path
import json
import shutil
import sys
from termcolor import colored, cprint
from pprint import pprint
import configparser


def config_parser(config):
    for key1, value1 in config.items():
        dictTemp = config[key1]
        for key2, value2 in dictTemp.items():
            if value2 == 'True':
                config[key1][key2] = True
                print("config[{}][{}]'s value is changed from yes to True".format(key1,key2))
            elif value2 == 'False':
                config[key1][key2] = False
                print("config[{}][{}]'s value is changed from no to False".format(key1,key2))
    return config
def command_interface(title=None):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--config', '-cf', default=None, help='training configs json file')
    parser.add_argument('--devices', '-d', nargs='+', default=None, type=int, help='CUDA devices. Use CPU if None')
    parser.add_argument('--rand_seed', '-r', default=1, type=int, help='random seed initialization')
    parser.add_argument('--lr', '-lr', default=0.04, help='learning_rate')
    parser.add_argument('--name', '-n', default='exp', help='name of this experiment')
    parser.add_argument('--mode', '-m', default='new', choices=['new', 'resume', 'test', 'finetune'], help='running mode')
    parser.add_argument('--iters', '-i', default=1, type=int, help='number of iterations to run the experiment')
    parser.add_argument('--omniscient', '-o', action='store_true', help='if specified, set validation set = test set')
    parser.add_argument('--overwrite', '-ow', action='store_true', help='if specified, overwrite existing folder without asking')
    parser.add_argument('--workers', '-w', default=12, type=int, help='number of workers for the dataloader')
    parser.add_argument('--amp', '-a', action='store_true', help='if specified, turn amp on')
    args = parser.parse_args()
    pprint(vars(args))

    config = json.load(open(args.config))
    '''
    config_parser (JW)
    '''
    config = config_parser(config)

    #save_root_name comes_from config_name
    args.name = args.config.split('/')[-1].replace(".json","")+ f'[{args.lr}]'
    config['train']['lr'] = float(args.lr)
    assert config['train']['lr'] == float(args.lr)

    save_root = Path('weights')/args.name
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
