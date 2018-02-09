# OpenGym CartPole-v0
# -------------------
#
# This code demonstrates use of a basic Q-network (without target network)
# to solve OpenGym CartPole-v0 problem.
#
# Made as part of blog series Let's make a DQN, available at: 
# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
# 
# author: Jaromir Janisch, 2016


#--- enable this to run on GPU
# import os    
# os.environ['THEANO_FLAGS'] = "device=gpu,floatX=float32"  

import random, numpy, math, gym, sys

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.25
set_session(tf.Session(config=config))

#----------
HUBER_LOSS_DELTA = 1.0
LEARNING_RATE = 0.00025
MEMORY_CAPACITY = 100000
BATCH_SIZE = 128

UPDATE_TARGET_FREQUENCY = 100

#----------
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel() 

    def _createModel(self):
        model = Sequential()

        model.add(Dense(256, activation='relu', input_dim=stateCnt))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(actionCnt, activation='linear'))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer='adam')

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=BATCH_SIZE, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)        

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)
    
    def isFull(self):
        return len(self.samples) >= self.capacity

#-------------------- AGENT ---------------------------

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)        

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # debug the Q function in poin S
        if self.steps % 100 == 0:
            S = numpy.array([-0.01335408, -0.04600273, -0.00677248, 0.01517507])
            pred = agent.brain.predictOne(S)
            print(pred[0])
            sys.stdout.flush()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=True)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))
        
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)
        
class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    def act(self, s):
        return random.randint(0, self.actionCnt-1)

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

    def replay(self):
        pass        

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        self.env = self.env.unwrapped

    def run(self, agent, render=False):
        s = self.env.reset()
        R = 0 

        while True: 
            if(render):        
                self.env.render()

            a = agent.act(s)

            s_, r, done, info = self.env.step(a)
            
            # From Cart pole env, check https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/6_OpenAI_gym/run_CartPole.py
            x, x_dot, theta, theta_dot = s_
            r1 = (self.env.x_threshold - abs(x))/self.env.x_threshold - 0.8
            r2 = (self.env.theta_threshold_radians - abs(theta))/self.env.theta_threshold_radians - 0.5
            r = r1 + r2

            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_) )
            agent.replay()            

            s = s_
            R += r

            if done:
                break

        print("Total reward:", R,)

#-------------------- MAIN ----------------------------
PROBLEM = 'CartPoleExtraLong-v0'
gym.envs.register(
    id=PROBLEM,
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=200,
)
env = Environment(PROBLEM)

stateCnt  = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.n

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)

try:
    while randomAgent.memory.isFull() == False:
        env.run(randomAgent)

    agent.memory.samples = randomAgent.memory.samples
    randomAgent = None
    for i in range(1, 100):
        print("Episod:", i)
        if(i < 10):
            env.run(agent)
        else:
            env.run(agent, True)
finally:
    agent.brain.model.save("cartpole-basic.h5")