#!/usr/bin/python3
from network_models import d_net, Policy_net
from algo import Discriminator, PPO, GAIL_wrapper
import numpy as np
import tensorflow as tf
import gym
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='log directory', default='log/')
parser.add_argument('--gen_save', help='save directory', default='trained_models/')
parser.add_argument('--disc_save', help='save directory', default='trained_models/')
parser.add_argument('--iters', default=int(1e4), type=int)
args = parser.parse_args()

obs_dims = 4
n_actions = 2

env = gym.make("CartPole-v0")

agent = PPO(args.gen_save, Policy_net, obs_dims, 2)
D = Discriminator(args.disc_save, obs_dims, n_actions)
trainer = GAIL_wrapper(agent, D, env, args.logdir)

trainer.train(args.iters)