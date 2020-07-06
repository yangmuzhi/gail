#!/usr/bin/python3
"""
wrapper for gail
"""

import numpy as np
import tensorflow as tf
import tqdm
import numpy
import pickle

class GAIL_wrapper:
    def __init__(self, agent, discriminator, env):
        
        self.agent = agent
        self.D = discriminator
        self.saver = tf.train.Saver()
        self.sess = tf.Session() 
        self.sess.run(tf.global_variables_initializer())
        self.env = env

    def train(self, episodes, save_model_dir, log_dir):
        train_fq = 1
        rewards_his = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            expert_observations = np.genfromtxt('trajectory/observations.csv')
            expert_actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)

            for i in tqdm.tqdm(range(episodes)):
                observations = []
                actions = []
                rewards = []
                v_preds = []
                r = 0
                obs = self.env.reset()
                done = False
                while not done:
                    act, v_pred = self.agent.Policy.act(obs.reshape(-1,4))
                    act = act[0]
                    next_obs, reward, done, info = self.env.step(act)
                    observations.append(obs)
                    actions.append(act)
                    rewards.append(reward)
                    v_preds.append(v_pred)
                    obs = next_obs
                    r += reward

                next_obs = np.stack([next_obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                _, v_pred = self.agent.Policy.act(obs=next_obs, stochastic=True)
                v_preds_next = v_preds[1:] + [np.asscalar(v_pred)]
                rewards_his.append(r)

                # update
                # train discriminator
                if (i+1) % train_fq == 0:
                    for _ in range(2):
                        self.D.train(expert_s=expert_observations,
                        expert_a=expert_actions,
                        agent_s=observations,
                        agent_a=actions)
                        d_rewards = self.D.get_rewards(agent_s=observations, agent_a=actions).reshape(-1)
                        gaes = self.agent.get_gaes(rewards=d_rewards, v_preds=v_preds,v_preds_next=v_preds_next)

                        gaes = np.array(gaes).reshape(-1)
                        v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

                    # train policy
                    inp = [observations, actions, gaes, d_rewards, v_preds_next]
                    self.agent.assign_policy_parameters()
                    for _ in range(6):
                        sample_indices = np.random.randint(low=0, high=len(observations), size=32)  
                        sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                        self.agent.train(obs=sampled_inp[0],
                                            actions=sampled_inp[1],
                                            gaes=sampled_inp[2],
                                            rewards=sampled_inp[3],
                                            v_preds_next=sampled_inp[4])
                if (i+1) % (train_fq * 1000) == 0:
                    observations = []
                    actions = []
                    rewards = []
                    v_preds = []
                    with open(log_dir + "/r_{}.pkl".format(i), "wb") as f:
                        pickle.dump(rewards_his, f)
                    rewards_his = []
                    
            self.saver.save(sess, save_model_dir + '/model.ckpt')
            print("saved model")

