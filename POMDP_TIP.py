import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import gridspec
random.seed(11) # Setting the random seed


"""  Note: This python file is adapted from an original matlab program.   """
"""  There are multiple changes being made to adapt to the python style:  """
"""  1. Changes in the dummy variable values and indices to adapt to 0-indexing in python   """
"""  2. Typos fixed in multiple functions                                 """
N = 15 # Number of search sites in the mission
numSessions = 15
wf_human_robot_direct, ws_human_robot_direct = 10, 20 # Predetermined trust update parameters
wf_human_human, ws_human_human = 5, 20
wf_human_robot_indirect, ws_human_robot_indirect = 10, 20


# Parameters for the SAR mission
q = np.array([1,0.2]) # weight of health/time costs
h1, h2 = 1, 100 # Possible Health cost
c1, c2, c3, c4 = 300, 50, 250, 30 # Possible Time cost
ryy, ryn, rny, rnn = -np.dot(q, [h1, c1]), -np.dot(q, [h2, c2]), -np.dot(q, [0,c3]), -np.dot(q,[0,c4]) # Loss for different danger/action choice
robot_list = ["A", "B"]
human_list = ["x", "y"]

discount_factor = 0.9

d = np.random.rand(N, 1) # Danger level of each site
kappa1, kappa2 = 2, 50 
d_hat, d_tilde = np.random.beta(d*kappa2, (1-d)*kappa2), np.random.beta(d*kappa1, (1-d)*kappa1) # Human/Robot perception of danger level

# Initial alpha, beta values for initial trust
alpha_human_human_0, beta_human_human_0 = 100, 30 
alpha_human_robot_0, beta_human_robot_0 = 100, 50 

SCap = 800 # used for value/action matrix

#runSimulation = True
runSimulation = False
runMatrixVersion = True
# runMatrixVersion = False

# Parameter for Experiment 1
sensedTB = 0    # 0: reverse, 1: disuse
objective = 0   # 0: task, 1: Trustseeking

# Parameter for Experiment 2
actualTB = 0    # 0: reverse, 1: disuse
simNum = 10000

"""    Helper Functions   """

def UpdateTrust(alpha, beta, threatPresence, recommendation, ws, wf):
    alpha_updated, beta_updated = 0, 0
    if threatPresence - recommendation == 0:
        alpha_updated = alpha + ws
        beta_updated = beta
    else:
        alpha_updated = alpha
        beta_updated = beta + wf
    return alpha_updated, beta_updated


def GenerateHumanAction(alpha, beta, d_tilde_siteIndex, actualTB, recommendation):
    # generate what the soldier will do after given the robot's recommendation
    trust = alpha / (alpha + beta)
    if trust > np.random.rand():
        humanAction = recommendation
    elif actualTB == 0:
        humanAction = 1 - recommendation
    elif actualTB == 1:
        humanAction = d_tilde_siteIndex > np.random.rand()
    else:
        print("unknown human behavior")
    return humanAction


def UpdateReward(reward, threatPresence, humanAction):
    # update task reward according to the soldier's action and actual danger existence
    if threatPresence:
        reward_updated = (reward + ryy) if humanAction else (reward + ryn)
    else:
        reward_updated = (reward + rny) if humanAction else (reward + rnn)
    return reward_updated       



def ImmediateReward(pFollow, pNotFollow, siteIndex, d_hat, d_tilde, sensedTB, objective):
    # use the robot sensed danger level for the current site
    # use reported prior for the future sites
    if objective == 0:
        trustReward = 0
    else:
        trustReward = 80 / (1 + np.exp(0.5 * siteIndex))

    if sensedTB == 0:             # reverse
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

    elif sensedTB == 1:           # disuse    
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

def UpdateHumanHumanTrust(human_human_alpha_beta, performance, ws_human_human, wf_human_human):
    alpha, beta = human_human_alpha_beta
    return (alpha + ws_human_human * performance, beta + wf_human_human * (1-performance))

def UpdateIndirectTrust(indirect_alpha_beta, ws_human_robot_indirect, wf_human_robot_indirect, human_human_alpha_beta, direct_alpha_beta):
    alpha_indirect, beta_indirect = indirect_alpha_beta
    indirect_trust = alpha_indirect / (alpha_indirect + beta_indirect)

    alpha_human_human, beta_human_human = human_human_alpha_beta
    human_human_trust = alpha_human_human / (alpha_human_human + beta_human_human)

    alpha_direct, beta_direct = direct_alpha_beta
    direct_trust = alpha_direct / (alpha_direct + beta_direct)

    return (alpha_indirect + ws_human_robot_indirect * human_human_trust * np.maximum(0, direct_trust - indirect_trust), 
            beta_indirect + wf_human_robot_indirect * human_human_trust * np.maximum(0, indirect_trust - direct_trust))

def CalculateBestAction(siteIndex, alpha, beta, ws, wf, d_hat, d_tilde, sensedTB, objective):
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
            alpha_matrix[j, subSiteIndex] = alpha + (subSiteIndex - j) * ws
            beta_matrix[j, subSiteIndex] = beta + j * wf
        pFollow = np.divide(alpha_matrix[:subSiteIndex + 1, subSiteIndex],
        (alpha_matrix[:subSiteIndex + 1, subSiteIndex] + beta_matrix[:subSiteIndex + 1, subSiteIndex]))
        pNotFollow = 1 - pFollow
        # Calculate immediate reward
        VY, VN = ImmediateReward(pFollow, pNotFollow, subSiteIndex, d_hat, d_tilde, sensedTB, objective)
        # Update current value function and action function
        d_k = d_hat[subSiteIndex] if subSiteIndex == 0 else d_tilde[subSiteIndex]
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



def SimulatingActualSearch(actualTB, sensedTB, objective):
    trust_dict = {}
    
    f = open("testHistory.txt", 'w')
    for human in human_list:
        trust_dict[human] = {}
        for other_human in human_list:
            if human != other_human: trust_dict[human][other_human] = (alpha_human_human_0, beta_human_human_0)
        for robot in robot_list:
            trust_dict[human][robot] = (alpha_human_robot_0, beta_human_robot_0)

    # Loop over different sessions
    for session in range(numSessions):
        f.write(f"Starting Session {session + 1}\n")
        # Initialize the danger levels and danger presence for this session
        d = np.random.rand(N, 1)
        threatPresenceSeq = [True if d_val > np.random.rand() else False for d_val in d]
        danger_dict = {}
        reward = 0
        for human in human_list:
            danger_dict[human] = np.random.beta(d*kappa2, (1-d)*kappa2)
        for robot in robot_list:
            danger_dict[robot] = np.random.beta(d*kappa1, (1-d)*kappa1)
        threatPresenceSeq = [True if d_val > np.random.rand() else False for d_val in d]
        # Randomly assign robot
        assignment = {}
        if np.random.rand() >= 0.5:
            assignment["x"] = "A"
            assignment["y"] = "B"
        else:
            assignment["x"] = "B"
            assignment["y"] = "A"

        # Loop over each human
        for i in range(len(human_list)):
            human = human_list[i]
            direct_robot = assignment[human]
            alpha, beta = trust_dict[human][direct_robot]
            d_hat, d_tilde = danger_dict[human], danger_dict[direct_robot]
            f.write(f"Human {human} works with Robot {direct_robot}\n")
            score = 0
            robot_score = 0
            for siteIndex in range(N):
                # Calculate the best action
                recommendation = CalculateBestAction(siteIndex, alpha, beta, ws_human_robot_direct, wf_human_robot_direct,
                                                      d_hat[siteIndex:], d_tilde[siteIndex:], sensedTB, objective)
                # Reveal the truth and update trust
                # Human action is determined by the trust behavior model
                humanAction = GenerateHumanAction(alpha, beta, d_tilde[siteIndex], actualTB, recommendation)
                if recommendation == threatPresenceSeq[siteIndex]: robot_score += 1
                if humanAction == threatPresenceSeq[siteIndex]: score += 1
                alpha, beta = UpdateTrust(alpha, beta, threatPresenceSeq[siteIndex], recommendation, ws_human_robot_direct, wf_human_robot_direct)
                reward = UpdateReward(reward, threatPresenceSeq[siteIndex], humanAction)
            
            f.write(f"Reward : {reward}, score: {score}, robot_score: {robot_score}, trust from {human} to {direct_robot} is {alpha / (alpha + beta)}\n")
            # Update direct trust after the session has finished
            trust_dict[human][direct_robot] = (alpha, beta)

            for other_human in human_list:
                if other_human != human:
                    # Update between-human trust
                    trust_dict[other_human][human] = UpdateHumanHumanTrust(trust_dict[other_human][human], score/N, ws_human_human, wf_human_human)

                    # Update indirect trust
                    trust_dict[other_human][direct_robot] = UpdateIndirectTrust(trust_dict[other_human][direct_robot],
                                                                                ws_human_robot_indirect, wf_human_robot_indirect,
                                                                                trust_dict[other_human][human], trust_dict[human][direct_robot])
        for human in human_list:
            for target, value in trust_dict[human].items():
                f.write(f"Trust from {human} to {target} is {value[0]/(value[0] + value[1]):.2f}, where alpha, beta = ({value[0]:.2f}, {value[1]:.2f})\n")

        f.write(f"------------------------------------------------------------------------------\n")
    f.close()
        

SimulatingActualSearch(0,0,0)
