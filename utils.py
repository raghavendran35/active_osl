import numpy as np
from time import time
import random
# def standard_np_sample(num_classes, classes_per_episode):
#     start_time = time()
#     res = random.sample(range(num_classes), classes_per_episode)
#     end_time = time()
#     seconds_elapsed = end_time - start_time
#     assert len(res) == classes_per_episode
#     #print(res)
#     print("%.5f seconds: " % seconds_elapsed)
# def possibly_faster_sampling(num_classes, classes_per_episode):
#     start_time = time()
#     classes_array = np.arange(0, num_classes)
#     np.random.shuffle(classes_array)
#     res = classes_array[:classes_per_episode]
#     end_time = time()
#     seconds_elapsed = end_time - start_time
#     assert len(res) == classes_per_episode
#     #print(res)
#     print("%.5f seconds: " % seconds_elapsed)
#Modified Uncertainty Volume Minimization
def returnUVMImages( classes, train_dataset, K = 30, num_timesteps=30, number_of_classes = 1200):
    # K is number of trajectories
    #classes is the array of classes in question (for now, just 3)
    # Input: Image indices with their corresponding labels (implicitly or explicitly)
    # Output: Resultant image indices that are optimal (hopefully!)
    #Assume "optimal policy" (as described in paper) is baseline, basically random selection
    #Kind of weird to try to outpace this, but it works reasonably after nearly 100,000 time steps,
    #so let's try to accelerate it!
    #Initialize Demonstrations
    Demo_indices = []
    Demo_indices_average = None
    #Initialize gamma_best (best of demos in trajectory?), as average demo
    gamma_best = None
    #best_labels = []
    length_of_demo_indices = 0
    #for now, take 3 classes and look at each 100 times
    S_0 = [classes[0]]*100 + [classes[1]]*100 + [classes[2]]*100
    for init_state in S_0:
    #generate the trajectory and receive the image and true label in question
    #check if the uncertainty measure (average) with new sample greater than old sample, then keep it as the optimal
    #if it is greater, than add index to demo_indices and corresponding label to best_labels
        class_result, ex_number = gen_trajectory_res(init_state, classes)
        curr_image = train_dataset[class_result][ex_number]
        if length_of_demo_indices >= 1:
            curr_demo_average = 1/(length_of_demo_indices + 1)*(length_of_demo_indices*Demo_indices_average + curr_image)
            #print(curr_demo_average)
        else:
            curr_demo_average = None
        if gamma_best is None:
            gamma_best = curr_image
            Demo_indices.append([class_result, ex_number])
            Demo_indices_average = gamma_best
            length_of_demo_indices+=1
        elif np.sum(curr_demo_average) >= np.sum(Demo_indices_average):
            gamma_best = curr_image
            Demo_indices.append([class_result, ex_number])
            Demo_indices_average = curr_demo_average
            length_of_demo_indices+=1
    #when all is said and done, return the last params.timesteps labels
    if len(Demo_indices)<num_timesteps:
        return (Demo_indices*30)[:num_timesteps]
    else:
        return Demo_indices[:num_timesteps]
#generate a trajectory of actions with random sampling, but let's see if we can't weight it
#1st run: preference for the similar class
###########################
#2nd run: class_props = misclassification error for each (Omniglot survey analysis)
def gen_trajectory_res(init_state, classes, class_props = np.array(1200 * [1 / 1200]), length_traj = 10):
    #basically list of the classes in question, take the last one and send it up to above along with a
    #random index which will correspond to the drawing for the class
    #generate num_classes_trajectories (think of a way to set up transition probabilities)
    #Trajectories are independent of each other anyway, because each class is independent of each other
    #with respect to the current context, we are just running it to get some behavior
    traj = [init_state] + [np.random.choice(classes)]
    #return the class, drawing example
    return traj[-1], np.random.randint(0, 20)
#below is unnecessary for first example
#def is_BEC(class_props):


#Goal is to generate 30 demonstrations from above algorithm w/ correspondingly, random selected index
#also return 30 random labels







    #predict labels when exploiting is lumped together
    #eps_actions = [[0, 1, 2], [0, 2]]
    #Baseline Policy is as follows:
    #Epsilon Greedy Exploration w/ eps=0.05,
    #           -exploring: 2/3 predict, 1/3 request label
    #           -exploiting: either request with 0.08 (baseline beginning prob-play w/ this) or
    #                        predict with 0.92