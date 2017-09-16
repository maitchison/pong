import pickle
import os
from datetime import datetime


def show_model_header():
    MODEL_INFO_HEADER = "Name                H      LR        WD        BS        IT         TIME     (ETA)      MACHINE"
    print(MODEL_INFO_HEADER)

def show_model_info(model_name):
    """ Prints some information about given model. """
    try:
        path = 'models/' + model_name + ".p"
        save_package = pickle.load(open(path, 'rb'))
    except:
        print("{0:20}           ---------------- missing ----------------".format(model_name))
        return

    format_str = "{0:16} {1:>4}      {2:<6}    {3:<5} {4:6}   {5:7}{9}{6:>10.1f} hrs   [{8:>4.1f}]     {7}"
    config = save_package['config']

    iterations = save_package['episodes']

    if 'total_training_time' in save_package['stats']:
        cooking_time = save_package['stats']['total_training_time'] / (60 * 60)
    else:
        cooking_time = 0

    if 'last_batch_time' in save_package['stats']:
        batch_time_hours = save_package['stats']['last_batch_time'].total_seconds() / (60.0 * 60.0)
        it_time = save_package['config'].batch_size / batch_time_hours
    else:
        it_time = (iterations / cooking_time) if cooking_time != 0 else 0

    machine_name = save_package['stats']['machine_name'] if 'machine_name' in save_package['stats'] else "n/a"

    last_update = datetime.fromtimestamp(os.path.getmtime(path))
    recent = (datetime.now() - last_update).total_seconds() < 30 * 60

    if it_time > 0:
        eta = (10000 - iterations) / (it_time)
    else:
        eta = 0

    print(format_str.format(
        model_name, config.H, config.learning_rate, config.weight_decay,

        config.batch_size, iterations, cooking_time, machine_name, eta, "*" if recent else " "
    ))