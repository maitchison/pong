import agent as rl
import sys

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
   
def create_model(model_name):
    #todo
    pass
    
def reevaluate(mode_name):
    pass
    
def usage_error():
    print("Usage pong [train] [model-name]")
    exit()

    
if len(sys.argv) <= 1:
    usage_error()

if sys.argv[1].lower() == 'train':
    if len(sys.argv) <= 2:
        usage_error()
    train(sys.argv[2])