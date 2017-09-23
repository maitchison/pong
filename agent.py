import numpy as np
import gym
import pickle
import os
from datetime import datetime

# just for host name
import socket

"""
Changes:
added total training time.
"""

MODEL_FOLDER = 'models/'

# Helper functions.

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

    
def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# helper functions for training and analysis.

class History:
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.x = []
        self.h = []
        self.logp_grad = []
        self.reward = []

        
class Config:
    
    def __init__(self):
        
        # default values
        self.D = 80 * 80
        self.H = 100
        self.batch_size = 5    
        self.learning_rate = 3e-3 
        self.gamma = 0.99             # gamma for RMS prop
        self.discount_rate = 0.99     # discount rate for algorithm  
        self.weight_decay = 0.01
        self.data_type = 'float32'
        
    def set_attr(self, k, v):
        
        if k == 'D':
            self.D = int(v)
        elif k == 'H':
            self.H = int(v)
        elif k == 'batch_size':
            self.batch_size = int(v)
        elif k == 'learning_rate':
            self.learning_rate = float(v)
        elif k == 'weight_decay':
            self.weight_decay = float(v)
        elif k == 'discount_rate':
            self.discount_rate = float(v)
        elif k == 'gamma':
            self.gamma = float(v)
        else:
            raise Exception("Invalid attribute {0}".format(k))
        
        
    def display(self, one_line = False):
        if one_line:
            format_str =  "{0:10} {1:10} {2:10} {3:10}\n"
            
        else:
            format_str =  "Hidden states:   {0}\n"+\
                          "Learning rate:   {1} / {5}\n"+\
                          "Weight decay:    {2:.4f}\n"+\
                          "Batch size:      {3}\n"+\
                          "Discount rate:   {4}\n"
            
        print(format_str.format(self.H, self.learning_rate, self.weight_decay, self.batch_size, self.discount_rate, self.gamma))
        
       
class Agent:
    
    def __init__(self, name = "model", config = None, make_new = False, silent = False):
        """ Initialise a model.  Name will be used when saving. """
        self.name = name                # name of model
        self.params = {}                # model params
        self.ema = None                 # exponential moving average
        self.rmsprop_cache = None 
        self.grad_buffer = None

        self.config = Config() if config is None else config
        
        self.env = None

        self.episode = 0
        self.worker_name = socket.gethostname()

        self.score_history = {}
        self.stats = {}
        self.lock_key = {}

        self.save_filename = "{0}.p".format(self.name)
        
        if self.exists(self.name):
            if make_new:
                if not silent: print("Model {0} already exists.".format(self.name))
                exit()
            self.load()
            if not silent: print("\nPickling up '{0}' from episode {1}...\n".format(self.name, self.episode))
        else:
            
            if not make_new:
                print("Model {0} not found.".format(self.name))
                exit()

            if not silent: print("Starting new model '{0}'".format(self.name))
            self.params = {}
            self.params['W1'] = np.random.randn(self.config.H,self.config.D).astype(self.config.data_type) / np.sqrt(self.config.D) # "Xavier" initialization
            self.params['W2'] = np.random.randn(self.config.H).astype(self.config.data_type) / np.sqrt(self.config.H)
            self.rmsprop_cache = { k : np.zeros_like(v) for k,v in self.params.items() }    
            self.save()

        self.grad_buffer = { k : np.zeros_like(v) for k,v in self.params.items() } # update buffers that add up gradients over a batch
    
        self.batch_start_time = datetime.now()

        if not silent: self.config.display()
    
        # history over an entire episode (this will be big!)    
        self.history = History()


    def get_lock(self):
        """ Returns information on the current owner of this file. """
        save_package = pickle.load(open(MODEL_FOLDER+self.save_filename, 'rb'))
        if 'lock' not in save_package:
            #old version of save file, assume not locked....
            lock, worker, lock_key = (False, '', 0)
        else:
            lock, worker, lock_key = save_package['lock']

        return lock, worker, lock_key

    def exists(self, model_name):
        """ Returns if model exists or not. """
        path = MODEL_FOLDER+"{0}.p".format(self.name)
        return os.path.isfile(path)


    def last_updated(self, model_name = None):
        """ Returns date model was last updated. """
        if model_name is None: model_name = self.name
        path = MODEL_FOLDER+"{0}.p".format(model_name)
        return datetime.fromtimestamp(os.path.getmtime(path))


    def recent(self, delay = 30):
        """ Returns if this model was updated within 'delay' minutes. """
        return (datetime.now() - self.last_updated()).total_seconds() < delay * 60

    def prepro(self, I):
        """ preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(self.config.data_type).ravel()


    def policy_forward(self, x):
        h = np.dot(self.params['W1'], x)
        h[h<0] = 0 # ReLU nonlinearity
        logp = np.dot(self.params['W2'], h)
        p = sigmoid(logp)
        return p, h # return probability of taking action 2, and hidden state


    def close(self):
        """ Frees memory for this agent. """
        self.env.close()


    def policy_backward(self, eph, epx, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.params['W2'])
        dh[eph <= 0] = 0 # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1':dW1, 'W2':dW2}


    def lock(self):
        """ Force a lock on this file. """
        self.save(with_lock = True)


    def unlock(self):
        """ Release lock from this file. """
        self.save(with_lock = False)
                
        
    def save(self, filename = None, with_lock = False):
        """ Saves model parameters to disk.  If 'with_lock' is set this model will be marked as being locked
            by this worker for 30 minutes.  """
        save_package = {}
        if filename is None: filename = self.save_filename
        save_package['params'] = self.params
        save_package['episodes'] = self.episode
        save_package['rmsprop'] = self.rmsprop_cache
        save_package['history'] = self.score_history
        save_package['stats'] = self.stats
        save_package['config'] = self.config

        # record the last person to write to this file, and if the file should be considered locked or not.
        # locks last for 30 minutes.
        self.lock_key[filename] = np.random.randint(1e6,1e8)
        save_package['lock'] = (with_lock, self.worker_name, self.lock_key[filename])

        pickle.dump(save_package, open(MODEL_FOLDER+filename, 'wb'))
        
        
    def load(self, filename = None):
        """ Loads the parameters for the model from disk. """
        if filename is None: filename = self.save_filename
        save_package = pickle.load(open(MODEL_FOLDER+filename, 'rb'))
        self.params = save_package['params']
        self.episode = save_package['episodes']
        self.rmsprop_cache = save_package['rmsprop']
        self.score_history = save_package['history'] if 'history' in save_package else {}
        self.stats = save_package['stats'] if 'stats' in save_package else {}

        if 'config' in save_package:
            self.config = save_package['config']
            # some of these used to have different names
            if not hasattr(self.config, 'discount_rate'):
                self.config.discount_rate = self.config.decay_rate
                del self.config.decay_rate
            
            
        # convert to correct datatype.
        for v in self.params.keys():
            self.params[v] = self.params[v].astype(self.config.data_type) 
            
        if 'ema_history' in self.stats:
            # pick up ema from where we left off. 
            self.ema = self.stats['ema_history'][-1][1]
       
    def render(self):
        """ Render one game. """
        pass
        #global episode
        #episode = 0
        #while episode < 1:
        #    train_model_step(render = True)    
        

    def apply_batch(self):
        """ Updates weights with one batch of episodes worth of data. """
        # perform rmsprop parameter update every batch_size episodes
        for k,v in self.params.items():
            g = self.grad_buffer[k] # gradient
            self.rmsprop_cache[k] = self.config.gamma * self.rmsprop_cache[k] + (1 - self.config.gamma) * g**2
            self.params[k] += self.config.learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)            
            self.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
        self.params['W1'] -= self.params['W1'] * self.config.weight_decay * self.config.learning_rate
        self.stats['machine_name'] = self.worker_name

    def evaluate(self, deterministic = False, episodes = 100):
        """ 
            Evaluate the models performance by running specified number of episodes. 
            Returns a list of scores and frame counts for each episode.
        """
        
        scores = []
        frames = []
        for i in range(episodes):
            self.train(deterministic = deterministic)
            scores.append(np.sum(self.history.reward))
            frames.append(len(self.history.reward))
        return scores, frames
            
    
    def apply(self):
        """ Applies what was learned during training episode. """

        lock, worker_name, lock_key = self.get_lock()
        if (lock and (worker_name != self.worker_name or lock_key != self.lock_key[self.save_filename])):
            # this means someone modified the file since our last save.
            print("Expected lock: ", (self.worker_name, self.lock_key[self.save_filename]))
            print("Found lock:    ", (worker_name, lock_key))
            raise Exception("File was modified by another worker.  Terminating.")

        self.ema = self.ema * 0.95 + 0.05 * np.sum(self.history.reward) if self.ema is not None else np.sum(self.history.reward)

        # save on 0k
        if self.episode == 0:
            self.save(self.name + "_" + str(self.episode // 1000) + "k.p")

        self.episode += 1

        apply_start_time = datetime.now()

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(self.history.x)
        eph = np.vstack(self.history.h)
        epdlogp = np.vstack(self.history.logp_grad)
        epr = np.vstack(self.history.reward)
    
        # compute the discounted reward backwards through time
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr = discount_rewards(epr, self.config.discount_rate)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        # modulate the gradient with advantage (PG magic happens right here.)
        epdlogp *= discounted_epr 
        grad = self.policy_backward(eph, epx, epdlogp)
        for k in self.params: self.grad_buffer[k] += grad[k] # accumulate grad over batch

        apply_finish_time = datetime.now()

        self.stats['last_episode_apply_time'] = (apply_finish_time - apply_start_time).total_seconds()

        # apply batch
        if self.episode % self.config.batch_size == 0:
            self.apply_batch()
            batch_rewards = self.stats['this_batch_rewards']
            self.stats['this_batch_rewards'] = []
            self.stats['last_batch_rewards'] = batch_rewards
            self.stats['last_batch_score_mean'] = np.mean(batch_rewards)
            self.stats['last_batch_score_std'] = np.std(batch_rewards)
            self.stats['last_batch_time'] = datetime.now() - self.batch_start_time
            if 'total_training_time' not in self.stats:
                self.stats['total_training_time'] = 0.0
            self.stats['total_training_time'] = self.stats['total_training_time'] + self.stats['last_batch_time'].total_seconds()
            self.batch_start_time = datetime.now()

        # save backup
        if self.episode % 1000 == 0:
            self.save(self.name+"_"+str(self.episode//1000)+"k.p")
            print("Performing evaluation...")
            num_trials = 100
            scores, frames = self.evaluate(episodes = num_trials)

            score_mean = np.mean(scores)
            score_std = np.std(scores)
            score_error = np.std(scores) / np.sqrt(num_trials)
            n = num_trials
            frame_mean = np.mean(frames)
            frame_std = np.std(frames)

            # previous versions uses the 'score_history' field, we're now moving to stats['history'],
            # but I'll keep this here for backwards compatability for a while.
            self.score_history[self.episode] = (score_mean, score_error)

            if 'history' not in self.stats:
                self.stats['history'] = {}

            self.stats['history'][self.episode] = {
                'score_mean': score_mean,
                'score_std': score_std,
                'score_error': score_error,
                'n': n,
                'frame_mean': frame_mean,
                'frame_std': frame_std
            }

            print("Score = {0:.2f} (±{1:.4f})".format(score_mean, score_error))
            self.save(with_lock = True)

        # print
        if self.episode % 10 == 0:
            frames = self.stats['last_episode_frames']
            train_time = 1000.0 * self.stats['last_episode_train_time'] / frames
            apply_time = 1000.0 * self.stats['last_episode_apply_time'] / frames
            if 'ema_history' not in self.stats: self.stats['ema_history'] = []
            self.stats['ema_history'].append((self.episode, self.ema))
            print("[{5}] Ep {0} had {4} steps with (EMA) reward {2:.2f} [{3:.2f}ms / frame]".format(
                self.episode, np.sum(self.history.reward), self.ema, train_time + apply_time, frames, self.name
            ))

        # saving
        if self.episode % 100 == 0:
            print("Saving state.")
            self.save(with_lock = True)


    def train(self, render = False, deterministic = False):
        """ 
            Train the current model for one episode. 
            Normally actions are selected randomly depending on the action probability,
            however during evaulation deterinistic actions can be enabled.
        """

        if self.env == None: self.env = gym.make("Pong-v0")

        done = False
        
        # timing
        episode_start_time = datetime.now()
        prepro_time = 0.0
        forward_time = 0.0
        step_time = 0.0        
    
        # clean reset.
        self.history.reset()                
        observation = self.env.reset()
        prev_x = None            
        
        frames = 0
        
        while not done:                        
            
            frames += 1
                                
            if render: self.env.render()

            # preprocess the observation, set input to network to be difference image
            start_time = datetime.now()
            cur_x = self.prepro(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros(self.config.D, dtype = self.config.data_type)
            prev_x = cur_x
            prepro_time += (datetime.now() - start_time).total_seconds() * 1000

            # forward the policy network and sample an action from the returned probability
            # note, we pick an action non-deterministically.  This allows the agent to explore
            # actions that may, at first, see to be poor.
            start_time = datetime.now()
            action_prob, h = self.policy_forward(x)
            if deterministic:
                if abs(action_prob - 0.5) < 0.2:
                    action = 1
                else:
                    action = 2 if action_prob > 0.5 else 3 
            else:
                action = 2 if np.random.uniform() < action_prob else 3 

            forward_time += (datetime.now() - start_time).total_seconds() * 1000

            # calculate a gradient that encourages the action that was taken to be taken
            # we will apply this gradient positively for positive rewards, and negative for negative rewards.
            y = 1 if action == 2 else 0 

            # record a history of what happened during the episode.
            self.history.x.append(x)
            self.history.h.append(h)
            self.history.logp_grad.append(y - action_prob)

            # apply our action and sample the new environment.
            start_time = datetime.now()
            observation, reward, done, info = self.env.step(action)
            step_time += (datetime.now() - start_time).total_seconds() * 1000

            # we record the reward after the step as this will be the reward for the previous action.
            self.history.reward.append(reward)

        episode_finish_time = datetime.now()

        # calculate some handy stats
        self.stats['last_episode_train_time'] = (episode_finish_time - episode_start_time).total_seconds()
        self.stats['last_episode_frames'] = frames
        self.stats['last_episode_score'] = np.sum(self.history.reward)
        self.stats['last_episode_prepro_time'] = prepro_time
        self.stats['last_episode_forward_time'] = forward_time
        self.stats['last_episode_step_time'] = step_time
        if 'this_batch_rewards' not in self.stats:
            self.stats['this_batch_rewards'] = []
        self.stats['this_batch_rewards'].append(np.sum(self.history.reward))
