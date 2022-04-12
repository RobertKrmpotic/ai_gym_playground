import os
import sys
from pathlib import Path
import random
import gym
import time
import copy
import numpy as np
import pickle
from generations import Generation, Species

def list_unique_vals(rewards_list:list) ->int:
    return len(set(rewards_list))

def run_simulation( model, episode_len_limit:int =1600 ,name:str = "BipedalWalker-v3")->int:
    env = gym.make(name)
    observation = env.reset()
    len_output = env.action_space.shape[0]
    len_input = len(observation)
    rewards_list = []
    fitness = 0
    unique_rewards=10
    #run each step
    for t in range(episode_len_limit):
        env.render()
        action = model.spit_output( observation, len_input).flatten() #random generation np.random.uniform(-1,1,size=4)
        observation, reward, done, info = env.step(action)
        fitness += reward
        #track last 100 rewards
        if len(rewards_list)>=100:
            unique_rewards = list_unique_vals(rewards_list)
            del rewards_list[0]
        rewards_list.append(round(reward,3))

        #if agents stalls and just stands
        if unique_rewards < 3:
            #calculate how much it would get by the end while doing this
            left_over_reward = rewards_list[0] * (episode_len_limit-t)
            #punish this type of behaviour
            stalling_punishment = 50
            final_fitness = fitness - 40 + left_over_reward - stalling_punishment
            #print("kicked out staller")
            break


        #if game says done or time limit expires
        if done or t>=episode_len_limit :
            #print("Episode finished after {} timesteps, fitness = {}".format(t+1, (fitness - 40*(t/episode_len_limit))))
            final_fitness = (fitness - 40*(t/episode_len_limit))
            done=False
            break
    env.close()
    return final_fitness
 
def save_logs(gen_counter, dictionary, location):
    Path(f"logs/{location}").mkdir(exist_ok=True)
    with open(f'logs/{location}/gen_{gen_counter}.pickle', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_warm_start(gen=5, location="model_weights"):
    with open(f'logs/{location}/gen_{str(gen)}.pickle', 'rb') as handle:
        weights_list = pickle.load(handle)
        top_performers = []
        for net_weights in weights_list:
            top_performers.append(Species(weights=net_weights).brain)
        print("loaded warm start...")
        return top_performers

def main_loop(number_of_generations=2, warm_start_gen=0):
    global gen_counter,top_performers
    gen_counter = 1
    top_performers = []
    if warm_start_gen>0:
        gen_counter = warm_start_gen+1
        top_performers = load_warm_start(gen=warm_start_gen)
    while number_of_generations >= gen_counter: 
        #create generation
        current_generation = Generation(gen_counter,top_performers)
        neural_net_dict = current_generation.neural_net_dict

        fitness_dict={}
        #for brain of each species
        for key in neural_net_dict:
            net_fitness = run_simulation(neural_net_dict[key])
            neural_net_dict[key].set_fitness(net_fitness)
            fitness_dict[key] = net_fitness
        top_performers_int = sorted(fitness_dict, key=fitness_dict.get, reverse=True)[:8]
        top_performers = []
        save_logs(gen_counter, fitness_dict, "stats")
        
        print(f"Generation: {gen_counter}  Avg fitness: {np.array(list(fitness_dict.values())).mean()}, Max fitness: {np.array(list(fitness_dict.values())).max()}")

        for val in top_performers_int:
            top_performers.append(neural_net_dict[val].brain)

        if gen_counter % 5 == 0:
            weights_list = []
            for brain in top_performers:
                weights_list.append(brain.weights)
            save_logs(gen_counter, weights_list, "model_weights")

        gen_counter +=1

if __name__ == "__main__":

    main_loop(3)

