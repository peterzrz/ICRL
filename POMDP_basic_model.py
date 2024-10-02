import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import gridspec
random.seed(11) # Setting the random seed


"""  Note: This python file is adapted from an original matlab program.   """
"""  There are multiple changes being made to adapt to the python style:  """
"""  1. Changes in the dummy variable values and indices to adapt to 0-indexing in python   """
"""  2. Typos fixed in multiple functions                                 """


#################################################################################
############################  Key Parameters   ##################################
#################################################################################

N = 15 # Number of search sites in the mission
wf, ws = 10, 20 # Predetermined trust update parameters

# Parameters for the SAR mission
q = np.array([1,0.2]) # weight of health/time costs
h1, h2 = 1, 100 # Possible Health cost
c1, c2, c3, c4 = 300, 50, 250, 30 # Possible Time cost
ryy, ryn, rny, rnn = -np.dot(q, [h1, c1]), -np.dot(q, [h2, c2]), -np.dot(q, [0,c3]), -np.dot(q,[0,c4]) # Loss for different danger/action choice

discount_factor = 0.9

d = np.random.rand(N) # Danger level of each site
kappa1, kappa2 = 2, 50 
d_hat, d_tilde = np.random.beta(d*kappa2, (1-d)*kappa2), np.random.beta(d*kappa1, (1-d)*kappa1) # Human/Robot perception of danger level

alpha_0, beta_0 = 100, 50 # Initial alpha, beta values for initial trust

SCap = 800 # used for value/action matrix

runSimulation = True
#runSimulation = False
#runMatrixVersion = True
runMatrixVersion = False

# Parameter for Experiment 1
sensedTB = 0    # 0: reverse, 1: disuse
objective = 0   # 0: task, 1: Trustseeking

# Parameter for Experiment 2
actualTB = 0    # 0: reverse, 1: disuse
simNum = 10

#################################################################################
############################  Helper Functions  #################################
#################################################################################

def UpdateValueFunctionMatrix(V, VY, VN, SCap, ws, wf, discount_factor, siteIndex, d_hat, d_tilde):
    # Instead of finding different possible alpha/beta values after each site in simulation mode, 
    # this function maps out all possible alpha/beta values in a continuous range and finds best action accordingly
    d_k = d_hat[siteIndex] if siteIndex == 0 else d_tilde[siteIndex]
    if siteIndex < N - 1:
        tmp_matrix = np.zeros((wf + SCap, ws + SCap))
        tmp_matrix[:SCap, :SCap] = V[:,:,siteIndex + 1]
        V_next_temp = discount_factor * tmp_matrix
        # case Y:
        # action Y danger Y: prob dk      ,next state (alpha + ws, beta)
        # action Y danger N: prob 1 - dk  ,next state (alpha, beta + wf)
        VY = VY + \
        d_k       *   V_next_temp[:SCap,ws:SCap + ws] + \
        (1 - d_k) *   V_next_temp[wf:SCap + wf, :SCap]
        # case N:
        # action N danger Y: prob dk,     ,next state (alpha, beta + wf)
        # action N danger N: prob 1-dt    ,next state (alpha + ws, beta)
        VN = VN + \
        d_k        *   V_next_temp[wf:SCap + wf, :SCap] + \
        (1 - d_k)  *   V_next_temp[:SCap, ws:SCap + ws]
   
    ACurrentSite = [[True if VY[i][j] >= VN[i][j] else False for j in range(VY.shape[1])] for i in range(VY.shape[0])]
    VCurrentSite = np.maximum(VY, VN)
    return ACurrentSite, VCurrentSite



def UpdateTrust(alpha, beta, threatPresence, recommendation, ws, wf):
    # Update the actual change in trust according to the given robot recommendation and actual danger presence
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



def CalculateBestAction(siteIndex, alpha, beta, ws, wf, d_hat, d_tilde, sensedTB, objective):
    # Calculate the best action at the current site by solving a sub POMDP
    # Rename the index of the current site as 0, thus the index of the rest
    # site is n = N - siteIndex, and convert other variables into a subproblem from site siteIndex to N
    n = N - siteIndex
    # We need two matrix coding all the possible alpha and beta values during the process
    alpha_matrix = np.zeros((n+1, n+1))
    beta_matrix = np.zeros((n+1, n+1))
    a_matrix = np.zeros((n+1, n+1))
    v_matrix = np.zeros((n+1, n+1))
    # Update the value/action reversely: subSiteIndex from n-1 to 0
    # first we update the value function and action function reversely
    for k in range(n):
        subSiteIndex = n - k - 1 
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
        a_matrix[:subSiteIndex + 1, subSiteIndex] = [True if VY[i] > VN[i] else False for i in range(VY.shape[0])]

    bestAction = a_matrix[0, 0]
    return bestAction



def SimulatingActualSearch(alpha_0, beta_0, ws, wf, actualTB, sensedTB, objective, d, d_hat, d_tilde):
    alpha, beta = alpha_0, beta_0
    # Assign threat presence in each site based on the danger level as a probability
    threatPresenceSeq = [True if d_val > np.random.rand() else False for d_val in d]
    # Some logistics to keep track of
    alpha_beta_history = np.zeros((N+1,2))
    alpha_beta_history[0] = [alpha, beta]
    result = np.zeros((N, 10))
    # Total rewards gained during the simulation
    # Note: in the SAR model, this reward is actually a negative loss that we want to avoid
    reward = 0
    for siteIndex in range(N):
        # Calculate the best action by solving a POMDP backwards
        recommendation = CalculateBestAction(siteIndex, alpha, beta, ws, wf, d_hat[siteIndex:], d_tilde[siteIndex:], sensedTB, objective)
        # Reveal the truth and update trust
        # Human action is determined by the trust behavior model
        humanAction = GenerateHumanAction(alpha, beta, d_tilde[siteIndex], actualTB, recommendation)
        # Reveal the actual danger in the site and update trust&reward accordingly
        alpha, beta = UpdateTrust(alpha, beta, threatPresenceSeq[siteIndex], recommendation, ws, wf)
        reward = UpdateReward(reward, threatPresenceSeq[siteIndex], humanAction)
        # Update the logistics
        alpha_beta_history[siteIndex + 1] = [alpha, beta]
        result[siteIndex, :] = [recommendation, threatPresenceSeq[siteIndex], humanAction, alpha, beta, alpha/(alpha+beta), 
                             d[siteIndex], d_hat[siteIndex], d_tilde[siteIndex], reward]
    return result



def plotValueActionMatrix(siteIndexSet, SCap, ws, wf, A, V, d_hat, figureNumber):

    # Create figure
    fig = plt.figure(figureNumber)
    fig.set_size_inches(13, 3)
    fig.patch.set_facecolor('white')

    # Setup tight layout using gridspec
    gs = gridspec.GridSpec(2, 8)
    gs.update(left=0.03, right=0.97, top=0.94, bottom=0.06, wspace=0.04, hspace=0.03)

    # Calculate plotRegion and totalPlotNumber
    plotRegion = min(SCap - N * max(wf, ws), 500)
    totalPlotNumber = len(siteIndexSet)

    # Determine Vmin and Vmax
    Vmin = np.min(V[:plotRegion, :plotRegion, :])
    Vmax = np.max(V[:plotRegion, :plotRegion, :])

    for plotIndex in range(totalPlotNumber):
        k = int(siteIndexSet[plotIndex])

        # First subplot for action matrix A
        ax = fig.add_subplot(gs[0, plotIndex])
        ax.imshow(A[:plotRegion, :plotRegion, k], aspect='auto',origin='lower')
        ax.set_title(f'site {k}, $\hat{{d}}_{{{k}}}$ = {d_hat[k][0]:.2f}', fontsize=10)
        # ax.set_xlabel('$$\\alpha_k$$')
        # ax.set_ylabel('$$\\beta_k$$')
        ax.set_visible(True)
        ax.set_yticks([0,100,200,300,400,500])
        ax.set_xticks([0,500])

        # Second subplot for value matrix V
        ax2 = fig.add_subplot(gs[1, plotIndex])
        Vmin = np.min(V[:plotRegion, :plotRegion, k])
        Vmax = np.max(V[:plotRegion, :plotRegion, k])
        im = ax2.imshow(V[:plotRegion, :plotRegion, k], vmin=Vmin, vmax=Vmax, aspect='auto', origin='lower')
        # ax2.set_xlabel('$$\\alpha_k$$')
        # ax2.set_ylabel('$$\\beta_k$$')
        ax2.set_visible(True)
        ax2.set_yticks([0,100,200,300,400,500])
        ax2.set_xticks([0,500])

    # Switch case for figure export file
    if figureNumber == 0:
        exportTitle = 'POMDP_V_A_case_11_rev_mission.pdf'
    elif figureNumber == 1:
        exportTitle = 'POMDP_V_A_case_12_rev_trust.pdf'
    elif figureNumber == 10:
        exportTitle = 'POMDP_V_A_case_21_dis_mission.pdf'
    elif figureNumber == 11:
        exportTitle = 'POMDP_V_A_case_22_dis_trust.pdf'
    else:
        raise ValueError('figureNumber wrong')

    # Export the figure
    plt.savefig(exportTitle)


"""   Simulating the searching process   """

if runSimulation:
    rewardMeanHistory, rewardSTDHistory = [], []
    trustMeanHistory, trustSTDHistory = [], []
    resultHistory = []
    kappa_values = [[2,50]]
    f = open("resultTextHistory_untrusted_robot_low_accurate_robot.txt", 'w')
    alpha_0 = 100
    beta_0 = 50

    for kappa in kappa_values:
        kappa1, kappa2 = kappa[0], kappa[1]
        for objective in range(1):
            for sensedTB in range(1):
                for actualTB in range(1):
                    trustSum = 0
                    trustHistory = []
                    rewardSum = 0
                    rewardHistory = []
                    for j in range(simNum):
                        d = np.random.rand(N)

                        d_hat, d_tilde = np.random.beta(d*kappa2, (1-d)*kappa2), np.random.beta(d*kappa1, (1-d)*kappa1)
                        """d_hat = [0.25997403, 0.5538136,  0.73785895, 0.2621142,  0.63898997, 0.17989139,
                                    0.30456616 ,0.08711756, 0.34683773, 0.73849205, 0.41265332, 0.00726155,
                                    0.57311441, 0.08622867, 0.39927154]
                        d_tilde = [0.32345232, 0.94276979, 0.97967099, 0.3993069,  0.70621215, 0.50522464,
                                     0.02841104, 0.06829462, 0.80570023, 0.84645376, 0.27868049, 0.05882758,
                                    0.9012276,  0.34510161, 0.66817395]"""  
                        # Run a simulation trial and return the results
                        result = SimulatingActualSearch(alpha_0, beta_0, ws, wf, actualTB, sensedTB, objective, d, d_hat, d_tilde)
                        trustSum += result[N - 1,5]
                        trustHistory.append(result[N - 1,5])
                        rewardSum += result[N - 1,9]
                        rewardHistory.append(result[N - 1,9])
                    rewardMean = rewardSum / simNum
                    rewardMeanHistory.append(rewardMean)
                    rewardSTD = np.std(np.array(rewardHistory))
                    rewardSTDHistory.append(rewardSTD)

                    trustMean = trustSum / simNum
                    trustMeanHistory.append(trustMean)
                    trustSTD = np.std(np.array(trustHistory))
                    trustSTDHistory.append(trustSTD)

                    dispText = (f"alpha_1: {alpha_0}, beta_1: {beta_0}, kappa1: {kappa1}, kappa2: {kappa2}, "
                    f"obj: {objective}, sensedTB: {sensedTB}, actualTB: {actualTB}, rewardMean: {rewardMean}, "
                    f"rewardSTD: {rewardSTD}, trustMean: {trustMean}, trustSTD: {trustSTD}\n")
                    f.write(dispText)
                    resultHistory.append([alpha_0,beta_0,kappa1,kappa2,objective,sensedTB,actualTB,rewardMean,rewardSTD,trustMean,trustSTD])
    #f.write(resultHistory)
    f.close()


"""   Value/Action Matrix   """

if runMatrixVersion:
    for sensedTB in range(2):
        for objective in range(2):
            # Calculating value function and action function as matrix
            # The state set {S1(i), S2(i)}
            alpha = np.linspace(1, SCap, SCap)
            beta = np.linspace(1, SCap, SCap)
            s1, s2 = np.meshgrid(alpha, beta)
            pFollow = np.divide(s1, (s1+s2))
            pNotFollow = np.divide(s2, (s1+s2))
            # value function and action function
            V = np.zeros((SCap, SCap, N))
            A = np.zeros((SCap, SCap, N))

            for k in range(N):
                siteIndex = N - 1 - k
                # Calculate immediate reward
                VY, VN = ImmediateReward(pFollow, pNotFollow, siteIndex, d_hat, d_tilde, sensedTB, objective)
                # Update current value function and action function
                A[:,:,siteIndex], V[:,:,siteIndex] = UpdateValueFunctionMatrix(V, VY, VN, SCap, ws, wf, discount_factor, siteIndex, d_hat, d_tilde)

            # plotting the result of the value function and action function
            siteIndexSet = np.ceil(np.linspace(0, N - 1, 8))
            figureNumber = sensedTB * 10 + objective
            plotValueActionMatrix(siteIndexSet, SCap, ws, wf, A, V, d_hat, figureNumber)

