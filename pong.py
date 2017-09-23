#! /usr/bin/env python3
import agent as rl
import sys
import utils
import os
import time
import numpy as np #remove
import argparse
import logging

"""

Front end for training PONG reinforcement algorthims.  Has various modes.

train: trains a given model
make: makes a new model with given parameters


"""

def train(model, episodes = 10000):
    """ Train specific model. """

    agent = rl.Agent(name = model)

    agent.lock()

    try:
        while agent.episode < episodes:
            agent.train()
            agent.apply()
    except Exception as e:
        print("Error:",e)
        agent.close()
        return

    agent.unlock()

    agent.close()

    print("Finished.")


def info(model):
    """ Display information about given model. """
    utils.show_model_header()
    utils.show_model_info(model)


def info_list(model_names):
    """ Display information about given models. """
    utils.show_model_header()
    for name in model_names:
        utils.show_model_info(name)

def is_backup_file(filename):
    """ Returns if this filename is a backup file or not. """
    # really should have thought this through before.  Backup files end with a number then k"
    return filename[-3:] == 'k.p'

def get_models(filter = ''):
    """ Returns a list of models. """
    files = os.listdir('models')
    files = [file[:-2] for file in files if filter in file and file[-2:] == '.p' and not is_backup_file(file)]
    return files


def restart(model_name, pickup_point):
    """ Restart the model from given point. """
    k = int(pickup_point) // 1000

    print("-"*40)
    print("Restarting {0} from {1}".format(model_name, k*1000))
    print("-" * 40)

    backup_name = "{0}_{1}k".format(model_name, k)
    agent = rl.Agent(name = backup_name, silent = True)
    agent.name = model_name
    agent.save_filename = model_name+".p"
    agent.save()
    print("Model {0} saved.  Use pong.py train {0} to train".format(model_name))
    print("Model RMSprop cache norm = {0:.1f}".format(np.linalg.norm(agent.rmsprop_cache['W1'])))
    print()
    #train(model_name)



def view(filter = ''):
    """ Continiously monitor model progress. """
    spinner = ['-','\\','|','/']
    i = 0

    while True:
        # build a list of model names
        files = get_models(filter)
        info_list(files)
        print(spinner[i % len(spinner)])
        i += 1
        time.sleep(10)
        print("\033["+str(len(files)+2)+"A", end='')


def requires_work(model_name):
    """ Returns if this model requires more processing or not.
        Models are considered to need work if they episodes <= 10000 and they 
        are either unassigned, or have not been updated recently (30 min)
    """

    # first check that we have not completed.
    model = rl.Agent(model_name, silent = True)
    if model.episode >= 10000:
        return False

    # next check if someone is working on this
    if model.worker_name != '' and model.recent():
        return False


    # wait 2 minutes before starting...
    if model.recent(2):
        return False

    return True



def work(filter = ''):

    print('Scanning....')
    if filter != '':
        print(" - applying filter '{0}'".format(filter))

    ignore_jobs = []

    while True:

        # look for potential models to work on
        files = get_models(filter)

        if len(files) == 0:
            time.sleep(60)
            continue

        jobs = []

        for model in files:
            if requires_work(model) and model not in ignore_jobs:
                jobs.append(model)

        if len(jobs) == 0:
            continue

        print(" - found {0} jobs".format(len(jobs)))

        job = jobs[0]

        try:
            print("Starting job {0}".format(job))
            train(job)
            print("Finished job {0}".format(job))
        except Exception as e:
            print("Error: ",e)
            ignore_jobs.append(job)

        # we restart the search as a lot may be changed by the time we finish training...
        time.sleep(2)



def make(model_name, params):
    
    config = rl.Config()   
    
    for k,v in params.items():
        try:
            config.set_attr(k, v)
        except:
            # probably fine, just an unused argument.
            pass

    agent = rl.Agent(name = model_name, config=config, make_new = True)
    agent.save()


def evaluate(model, points = 'missing', num_trials = 100, deterministic = False):

    agent = rl.Agent(model, silent = True)

    # get a lock
    agent.save(with_lock = True)

    if points.lower() == 'missing':
        # re-evaluate any missing data points.
        print("Evaluating {0} at missing datapoints (n={1}).".format(model, num_trials))
        max_epoch = agent.episode // 1000
        for epoch in range(1, max_epoch+1):
            episode = epoch * 1000
            if not agent.has_evaluation(episode, deterministic):
                print("Processing episode {0}:".format(episode))
                agent.reevaluate(episode = episode, episodes = num_trials, deterministic = deterministic)
                agent.save(with_lock = True)

        print("Done.")

    elif points.lower() == 'all':
        # re-evaluate all data points.
        print("Re-evaluating {0} at all datapoints (n={1}).".format(model, num_trials))
        max_epoch = agent.episode // 1000
        for epoch in range(1, max_epoch+1):
            agent.reevaluate(episode = epoch*1000, episodes = num_trials, deterministic = deterministic)
            agent.save(with_lock=True)
        print("Done.")

    else:
        # assume points is episode number.
        episode = int(points)
        print("Evaluating {0} at episode {1} (n={2})".format(model, episode, num_trials))
        agent.reevaluate(episode = episode, deterministic = deterministic, episodes = num_trials)

    # release lock
    agent.save(with_lock = False)

    print("Finished.")
    
    
def usage_error():
    print("Usage pong [train] [model-name]")
    exit()

def parse_params():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='subparser')

    parser_train = subparsers.add_parser('train', help = "Train model")
    parser_train.add_argument('model', help = 'Name of the model.')
    parser_train.add_argument('--episodes', default = '10000', type = int, help = 'Number of episodes to train to')
    parser_train.set_defaults(func=lambda args: train(args.model, args.episodes))

    parser_info = subparsers.add_parser('info', help = "Info on model")
    parser_info.add_argument('model', help='Name of the model.')
    parser_info.set_defaults(func=lambda args: info(args.model))

    parser_make = subparsers.add_parser('make', help = "Make a new model")
    parser_make.add_argument('model', help='Name of the model.')
    parser_make.add_argument('--H', default='100', type=int, help='Number of hidden units.')
    parser_make.add_argument('--batch_size', default='5', type=int, help='Batch size.')
    parser_make.add_argument('--learning_rate', default=3e-3, type=float, help='Learning rate.')
    parser_make.add_argument('--gamma', default=0.99, type=float, help='Gamma value for RMSprop.')
    parser_make.add_argument('--weight_decay', default=1e-2, type=float, help='Weight decay.')
    parser_make.add_argument('--discount_rate', default=0.99, type=float, help='Discount rate.')
    parser_make.set_defaults(func=lambda args: make(args.model, vars(args)))

    parser_work = subparsers.add_parser('work', help = "Enter worker mode, processes any pending jobs.")
    parser_work.add_argument('--filter', default='', help='Worker will only work on files matching the given pattern.')
    parser_work.set_defaults(func=lambda args: work(args.filter))

    parser_view = subparsers.add_parser('view', help = "Enter view mode, showing list of all models.")
    parser_view.add_argument('--filter', default='', help='View only files matching given pattern.')
    parser_view.set_defaults(func=lambda args: view(args.filter))

    parser_restart = subparsers.add_parser('restart', help = "Restart model from given point.")
    parser_restart.add_argument('model', help='Name of the model.')
    parser_restart.add_argument('--episode', default='0', help='Episode to restart from.')
    parser_restart.set_defaults(func=lambda args: restart(args.model, args.episode))

    parser_eval = subparsers.add_parser('eval', help = "Evaluate given model.")
    parser_eval.add_argument('model', help='Name of the model.')
    parser_eval.add_argument('point', default='missing', help='Which point to evaluate [missing|all|n] where n is the episode number.')
    parser_eval.add_argument('--trials', default='100', type=int, help='Number of trials to perform.')
    parser_eval.add_argument('--deterministic', dest='deterministic', action='store_true', help="If enabled evaluates with deterministic actions.")
    parser_eval.set_defaults(func=lambda args: evaluate(args.model, points = args.point, num_trials = args.trials, deterministic = args.deterministic))

    args = parser.parse_args()
    args.func(args)


def main():
    # disable gym logging.
    logger = logging.getLogger('gym.envs.registration')
    logger.setLevel(logging.CRITICAL)

    parse_params()


main()
"""
    
if sys.argv[1].lower() == 'train':
    if len(sys.argv) <= 2:
        usage_error()
    train(sys.argv[2])
elif sys.argv[1].lower() == 'eval':
    if len(sys.argv) <= 2:
        usage_error()
    evaluate(sys.argv[2])
elif sys.argv[1].lower() == 'make':
    make(sys.argv[2], sys.argv[3:])
elif sys.argv[1].lower() == 'info':
    info(sys.argv[2:])
elif sys.argv[1].lower() == 'work':
    worker(sys.argv[2] if len(sys.argv) >= 3 else "")
elif sys.argv[1].lower() == 'restart':
    restart(sys.argv[2], sys.argv[3] if len(sys.argv) >= 4 else "0")
elif sys.argv[1].lower() == 'watch':
    watch(sys.argv[2] if len(sys.argv) == 3 else '')
else:
    print("Usage pong [train] [model-name]")
    exit()
    
"""
    