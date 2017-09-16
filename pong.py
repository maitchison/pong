import agent as rl
import sys
import os    


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
    
    
def make(model_name, params):
    
    config = rl.Config()   
    
    for param in params:
        k,v = param.split("=")     
        config.set_attr(k, v)        

    agent = rl.Agent(name = model_name, config=config, make_new = True)
    
    
    
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
else:
    print("Usage pong [train] [model-name]")
    exit()
    