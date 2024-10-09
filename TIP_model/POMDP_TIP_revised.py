import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from matplotlib import gridspec
import re
import torch

random.seed(11) # Setting the random seed

wf_human_robot_direct, ws_human_robot_direct = 10, 20 # Predetermined trust update parameters
wf_human_human, ws_human_human = 5, 20
wf_human_robot_indirect, ws_human_robot_indirect = 10, 20
ratio = 0.7

# Parameters for the SAR mission
q = np.array([1,0.2]) # weight of health/time costs
h1, h2 = 1, 100 # Possible Health cost
c1, c2, c3, c4 = 300, 50, 250, 30 # Possible Time cost
ryy, ryn, rny, rnn = -np.dot(q, [h1, c1]), -np.dot(q, [h2, c2]), -np.dot(q, [0,c3]), -np.dot(q,[0,c4]) # Loss for different danger/action choice 61/110/50/6

discount_factor = 0.9

kappa1, kappa2 = 2, 50 

# Initial alpha, beta values for initial trust
alpha_human_human_0, beta_human_human_0 = 100, 30 
alpha_human_robot_0, beta_human_robot_0 = 100, 50 

class Human():
    def __init__(self, name, wf_hr_direct=wf_human_robot_direct, ws_hr_direct=ws_human_robot_direct, wf_hh=wf_human_human, 
                 ws_hh=ws_human_human, wf_hr_indirect=wf_human_robot_indirect, ws_hr_indirect=ws_human_robot_indirect, 
                  alpha_hh_0=alpha_human_human_0, beta_hh_0=beta_human_human_0, alpha_hr_0=alpha_human_robot_0, 
                  beta_hr_0=beta_human_robot_0, kappa=kappa1, actualTB=0):
        self.name = name
        self.wf_hr_direct = wf_hr_direct
        self.ws_hr_direct = ws_hr_direct
        self.wf_hh = wf_hh
        self.ws_hh = ws_hh
        self.wf_hr_indirect = wf_hr_indirect
        self.ws_hr_indirect = ws_hr_indirect
        self.alpha_hh_0 = alpha_hh_0
        self.beta_hh_0 = beta_hh_0
        self.alpha_hr_0 = alpha_hr_0
        self.beta_hr_0 = beta_hr_0
        self.kappa = kappa
        self.actualTB = actualTB

    def initialize_trust(self, human_list, robot_list):
        # Initialize human trust for each human and robot
        # This function should be called for each human every trial
        self.trust = {}
        for human in human_list:
            if human.name != self.name:
                self.trust[human.name] = (self.alpha_hh_0, self.beta_hh_0)
        for robot in robot_list:
            self.trust[robot.name] = (self.alpha_hr_0, self.beta_hr_0)

    def initialize(self):
        # Some important logistics to keep track of for the performance of each session
        # This function should be called at the beginning of every session
        self.reward = 0
        # when human makes the same decision as dangerPresence
        self.score = 0
        # when robot makes the same decision as dangerPresence
        self.robot_score = 0
        self.experienced_sites = 0
        self.performance = 0
        self.history = []
    
    def assign(self, working_robot):
        self.working_robot = working_robot

    def assess_danger(self, d):
        # acquire some prior information about the danger level using a low kappa value
        self.d_tilde = np.random.beta(d*self.kappa, (1-d)*self.kappa)

    def generate_human_action(self, siteIndex, recommendation):
        # Generate the possible human action based on robot recommendation and human trusting behavior
        alpha, beta = self.trust[self.working_robot]
        trust = alpha / (alpha + beta)
        if trust > np.random.rand():
            humanAction = recommendation
        elif self.actualTB == 0:
            humanAction = 1 - recommendation
        elif self.actualTB == 1:
            humanAction = self.d_tilde[siteIndex] > np.random.rand()
        else:
            print("unknown human behavior")
        return humanAction

    def update_trust(self, threatPresence, recommendation):
        # This function updates the direct trust towards the current partner robot during the session
        # According to the TIP paper, this update is 
        
        if threatPresence - recommendation == 0:
            self.robot_score += 1
            self.trust[self.working_robot] = (self.trust[self.working_robot][0] + self.ws_hr_direct, self.trust[self.working_robot][1])
        else:
            self.trust[self.working_robot] = (self.trust[self.working_robot][0], self.trust[self.working_robot][1] + self.wf_hr_direct)
        '''
        if threatPresence & recommendation:
            self.robot_score += 1
            self.trust[self.working_robot] = (self.trust[self.working_robot][0] + ratio * self.ws_hr_direct, self.trust[self.working_robot][1])
        elif threatPresence & not recommendation:
            self.trust[self.working_robot] = (self.trust[self.working_robot][0], self.trust[self.working_robot][1] + ratio * self.wf_hr_direct)
        elif not threatPresence & not recommendation:
            self.robot_score += 1
            self.trust[self.working_robot] = (self.trust[self.working_robot][0] + (1 - ratio) * self.ws_hr_direct, self.trust[self.working_robot][1])
        else:
            self.trust[self.working_robot] = (self.trust[self.working_robot][0], self.trust[self.working_robot][1] + (1 - ratio) * self.wf_hr_direct)
        '''
     
    def update_reward(self, threatPresence, humanAction):
        # This function updates the reward we gained during the session
        # Note: the reward here is actually negative. So, it is actually a loss we want to avoid
        if threatPresence == humanAction:
            self.score += 1
        self.experienced_sites += 1
        self.performance = self.score / self.experienced_sites

        if threatPresence:
            self.reward = (self.reward + ryy) if humanAction else (self.reward + ryn)
        else:
            self.reward = (self.reward + rny) if humanAction else (self.reward + rnn)
    

    def update_human_human_trust(self, other_h):
        # This function updates the interhuman trust after each session ends
        # According to the TIP paper, this update is:
        # alpha_k = alpha_(k-1) + ws * p_k, beta_k = beta_(k-1) + wf * (1-p_k)
        # Currently, I am using the score to represent the variable p_k
        self.trust[other_h.name] = (self.trust[other_h.name][0] + self.ws_hh * other_h.score, 
                                    self.trust[other_h.name][1] + self.wf_hh * (1-other_h.score))
    
    def update_indirect_trust(self, other_h, r):
        # This function updates the indirect trust towards another robot after each session ends
        # alpha_k = alpha_(k-1) + ws * t_k_hh * max(0, t_k_direct - t_(k-1)_indirect)
        # beta_k = beta_(k-1) + wf * t_k_hh * max(0, - t_k_direct + t_(k-1)_indirect)
        # This update happens after other human have reported their direct trust to the robot and 
        # also after we have renewed our trust towards other human

        # The indirect trust from previous time
        alpha_indirect, beta_indirect = self.trust[r.name]
        indirect_trust = alpha_indirect / (alpha_indirect + beta_indirect)
        # The interhuman trust
        alpha_human_human, beta_human_human = self.trust[other_h.name]
        human_human_trust = alpha_human_human / (alpha_human_human + beta_human_human)
        # The direct trust 
        alpha_direct, beta_direct = other_h.trust[r.name]
        direct_trust = alpha_direct / (alpha_direct + beta_direct)

        self.trust[r.name] = (alpha_indirect + self.ws_hr_indirect * human_human_trust * np.maximum(0, direct_trust - indirect_trust), 
            beta_indirect + self.wf_hr_indirect * human_human_trust * np.maximum(0, indirect_trust - direct_trust))

    def report(self, f):
        # To report after each session ends
        f.write(f"Human {self.name} works with Robot {self.working_robot}\n")
        f.write(f"Reward: {self.reward}, score: {self.score}, robot score:{self.robot_score}\n")
        for target, value in self.trust.items():
            f.write(f"Trust from {self.name} to {target} is {(value[0]/(value[0] + value[1])):.2f}, where alpha, beta = ({value[0]:.2f},{value[1]:.2f})\n")


class Robot():
    def __init__(self, name, objective=0, kappa=kappa2, sensedTB=0):
        self.name = name
        self.kappa = kappa
        self.objective = objective
        self.sensedTB = sensedTB
       
    def assess_danger(self, d):
        # At each site, the robot could scan the site for better danger level information using a higher kappa
        self.d_hat = np.random.beta(d*self.kappa, (1-d)*self.kappa)

    
    def immediate_reward(self, pFollow, pNotFollow, siteIndex, d_hat, d_tilde):
        # use the robot sensed danger level for the current site
        # use reported prior for the future sites
        if self.objective == 0:
            trustReward = 0
        else:
            trustReward = 80 / (1 + np.exp(0.5 * siteIndex))

        if self.sensedTB == 0:             # reverse
            if siteIndex == 0:
                d_k = d_hat[siteIndex]
            else:
                d_k = d_tilde[siteIndex]

            VY = \
            pNotFollow * d_k     * (ryn + trustReward) + \
            pFollow    * d_k     * (ryy + trustReward) + \
            pNotFollow * (1-d_k) * rnn + \
            pFollow    * (1-d_k) * rny

            VN = \
            pFollow    * d_k     * ryn + \
            pNotFollow * d_k     * ryy + \
            pFollow    * (1-d_k) * (rnn + trustReward) + \
            pNotFollow * (1-d_k) * (rny + trustReward)

        elif self.sensedTB == 1:           # disuse    
            d_tilde_k = d_tilde[siteIndex]       # unless at the current (1st) house, the robot will use d_tilde_k to estimate d_k
            if siteIndex == 0:
                d_hat_k = d_hat[siteIndex]
            else:
                d_hat_k = d_tilde_k
            
            VY = \
            (ryn + trustReward) * pNotFollow * (1 - d_tilde_k) * d_hat_k + \
            (ryy + trustReward) * (pFollow + pNotFollow * d_tilde_k) * d_hat_k + \
            rnn                 * pNotFollow  * (1 - d_tilde_k) * (1 - d_hat_k) + \
            rny                 * (pFollow + pNotFollow * d_tilde_k) * (1 - d_hat_k)
            
            VN = \
            ryn                 * ( pFollow + pNotFollow * (1 - d_tilde_k)) * d_hat_k + \
            ryy                 * pNotFollow * d_tilde_k * d_hat_k + \
            (rnn + trustReward) * (pFollow + pNotFollow * (1 - d_tilde_k)) * (1 - d_hat_k) + \
            (rny + trustReward) * pNotFollow * d_tilde_k * (1 - d_hat_k)

        else:
            print('unseen behavior')
        return VY, VN

    def calculate_best_action(self, h, N, siteIndex):
        # Calculate the best action at the current site by solving a sub POMDP
        # Rename the index of the current site as 1, thus the index of the best
        # Site is n = N - siteIndex, and convert other variables into a subproblem from site siteIndex to N
        n = N - siteIndex # n = N - siteIndex + 1
        # We need two matrix coding all the possible alpha and beta values during the process
        alpha_matrix = np.zeros((n+1, n+1))
        beta_matrix = np.zeros((n+1, n+1))
        a_matrix = np.zeros((n+1, n+1))
        v_matrix = np.zeros((n+1, n+1))
        # Update the value/action reversely: subSiteIndex from n to 1
        # First we let the last column of the value function be zero
        # v_matrix[:, -1] = np.zeros(n+1, 1)
        # Second we update the value function and action function reversely
        for k in range(n):
            subSiteIndex = n - k - 1 #subSiteIndex = n - k + 1
            # Update the column of the experience Matrix
            for j in range(subSiteIndex + 1):
                alpha_matrix[j, subSiteIndex] = h.trust[self.name][0] + (subSiteIndex - j) * h.ws_hr_direct
                beta_matrix[j, subSiteIndex] = h.trust[self.name][1] + j * h.wf_hr_direct
            pFollow = np.divide(alpha_matrix[:subSiteIndex + 1, subSiteIndex],
            (alpha_matrix[:subSiteIndex + 1, subSiteIndex] + beta_matrix[:subSiteIndex + 1, subSiteIndex]))
            pNotFollow = 1 - pFollow
            # Calculate immediate reward
            VY, VN = self.immediate_reward(pFollow, pNotFollow, subSiteIndex, self.d_hat, h.d_tilde)
            # Update current value function and action function
            d_k = self.d_hat[subSiteIndex] if subSiteIndex == 0 else h.d_tilde[subSiteIndex]
            # Case Y:
            # action Y danger Y: prob pk    , next state (alpha + ws, beta)
            # action Y danger N: prob (1-pk), next state (alpha, beta + wf)
            VY = VY + d_k * v_matrix[:subSiteIndex + 1, subSiteIndex + 1] + (1-d_k) * v_matrix[1: subSiteIndex + 2, subSiteIndex + 1]
            # Case N:
            # action N danger Y: prob dk,   , next state (alpha, beta + wf)
            # action N danger N: prob 1-dt  , next state (alpha + ws, beta)
            VN = VN + (1 - d_k) * v_matrix[:subSiteIndex + 1, subSiteIndex + 1] + d_k * v_matrix[1:subSiteIndex + 2, subSiteIndex + 1]
            v_matrix[:subSiteIndex + 1, subSiteIndex] = np.maximum(VY, VN)
            a_matrix[:subSiteIndex + 1, subSiteIndex] = [True if VY[i]> VN[i] else False  for i in range(VY.shape[0])]

        bestAction = a_matrix[0, 0]
        return bestAction

class Session():
    def __init__(self, session_num, human_list, robot_list, numSites=15):
        self.session_num = session_num
        self.human_list = human_list
        self.robot_list = robot_list
        self.numSites = numSites
    
    def assign_robot(self):
        robot_indices = np.arange(len(self.robot_list))
        np.random.shuffle(robot_indices)
        self.groups = []
        for i in range(len(self.human_list)):
            self.groups.append([self.human_list[i], self.robot_list[int(robot_indices[i])]])
            self.human_list[i].assign(self.robot_list[int(robot_indices[i])].name)

    def simulate(self, f):
        
        f.write(f"Starting Session {self.session_num} \n")
        # Initialize the danger levels and danger presence for this session
        self.danger_level = np.random.rand(self.numSites)
        self.threatPresenceSeq = [True if d_val > np.random.rand() else False for d_val in self.danger_level]

        # TODO: this step here is counterintuitive, because the robots should only assess danger before each site in empirical session
        # Should rewrite the code to make it more intuitive
        for human in self.human_list:
            human.assess_danger(self.danger_level)
        for robot in self.robot_list:
            robot.assess_danger(self.danger_level)
        
        # Randomly assign robot
        self.assign_robot()

        # Loop over each human
        for h,r in self.groups:
            h.initialize()
            for siteIndex in range(self.numSites):
                # Calculate the best action
                recommendation = r.calculate_best_action(h, self.numSites, siteIndex)
                # Reveal the truth and update trust
                # Human action is determined by the trust behavior model
                humanAction = h.generate_human_action(siteIndex, recommendation)
                # Update direct trust
                h.update_trust(self.threatPresenceSeq[siteIndex], recommendation)
                h.update_reward(self.threatPresenceSeq[siteIndex], humanAction)
            
            for other_h in self.human_list:
                if other_h.name != h.name:
                    # Update between-human trust
                    other_h.update_human_human_trust(h)
                    # Update indirect trust
                    other_h.update_indirect_trust(h, r)
        
        self.report(f)
        f.write("------------------------------------------------------------------------\n")
        
    
    def report(self, f):
        for h in self.human_list:
            h.report(f)



numTrials = 1
numSessions = 15
f = open("POMDP_TIP.txt", 'w')
for _ in range(numTrials):
    h1 = Human("x")
    r1 = Robot("A")
    human_list = [h1]
    robot_list = [r1]
    for human in human_list:
        human.initialize_trust(human_list, robot_list)
    for session_num in range(1, numSessions + 1):
        s = Session(session_num, [h1], [r1])
        s.simulate(f)
f.close()

