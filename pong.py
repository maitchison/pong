#! /usr/bin/env python3
import agent as rl
import sys
import utils
import os
import time

"""

Front end for training PONG reinforcement algorthims.  Has various modes.

train: trains a given model
make: makes a new model with given parameters


"""

def train(model_name):
    
    """ Train specific model. """

    # todo: check file exists...
    
    agent = rl.Agent(name = model_name)

    training_episodes = 10000
    
    try:
        while agent.episode < training_episodes:
            agent.train()
            agent.apply()
    except Exception as e:
        print(e)

    agent.env.close()

    print("Finished.")


def info(model_names):
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


def watch(filter = ''):
    """ Continiously monitor model progress. """
    spinner = ['-','\\','|','/']
    i = 0

    while True:
        # build a list of model names
        files = get_models(filter)
        info(files)
        print(spinner[i % len(spinner)])
        i += 1
        time.sleep(10)
        print("\033["+str(len(files)+2)+"A", end='')


def worker(filter = '*'):

    while True:

        # look for potential models to work on
        files = get_models(filter)

        # make sure model hasn't been worked on in a while (30 minutes...)

        # see how much progress we need to make

        # start working on this model
        # >> can we spin this up in another thread, so that we can have multiple workers

        print("Starting job {0}")
        print("Finished job {0}")


def make(model_name, params):
    
    config = rl.Config()   
    
    for param in params:
        k,v = param.split("=")     
        config.set_attr(k, v)        

    agent = rl.Agent(name = model_name, config=config, make_new = True)
    

# todo: move
def reevaluate(model_name):
    
    """ Re-runs the evaluation tests for this model.  Requires the backup
        models [model-name]_[x]k.p to be present."""  

    num_trials = 1
            
    agent = rl.Agent(model_name)
    x = 1

    model_fmt = "{0}_{1}k.p"
    
    print("Re-evaluating the model {0}".format(model_name))
    
    # check peformance at each checkpoint.
    while agent.exists(model_fmt.format(model_name, x)):
        agent.load(model_fmt.format(model_name, x))
        scores = agent.evaluate(episodes = num_trials)
        mean = np.mean(scores)
        error = np.std(scores) / np.sqrt(num_trials)
        print("[{2}k]: Score = {0:.2f} (±{1:.4f})".format(mean, error, x))
        x += 1
        
        # perform the save
        master = rl.Agent(model_name)
        master.score_history[x*1000] = (mean, error)
        master.save()
        master.env.close()
        
    print("Finished.")
    
    
def usage_error():
    print("Usage pong [train] [model-name]")
    exit()

    
if len(sys.argv) <= 1:
    usage_error()
    
if sys.argv[1].lower() == 'train':
    if len(sys.argv) <= 2:
        usage_error()
    train(sys.argv[2])
elif sys.argv[1].lower() == 'eval':
    if len(sys.argv) <= 2:
        usage_error()
    reevaluate(sys.argv[2])
elif sys.argv[1].lower() == 'make':
    make(sys.argv[2], sys.argv[3:])
elif sys.argv[1].lower() == 'info':
    info(sys.argv[2:])
elif sys.argv[1].lower() == 'watch':
    watch(sys.argv[2] if len(sys.argv) == 3 else '')
else:
    print("Usage pong [train] [model-name]")
    exit()
    