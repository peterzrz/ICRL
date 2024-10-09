import numpy as np
import random
import pandas as pd
import os
import glob
import re
import torch


numSites = 10

def interhuman_model(participant_count, trust_params, w):
    result = []
    result.append(trust_params)
    for count in participant_count:
        performance = torch.tensor([count / numSites, 1 - count / numSites])
        trust_params = trust_params + torch.mul(performance, w)
        result.append(trust_params)
    return result
        
def loss_beta(modeled_trust, player_trust):
    loss = 0
    for i in range(len(modeled_trust)):
        b = torch.distributions.Beta(modeled_trust[i][0], modeled_trust[i][1])
        loss = loss - b.log_prob(torch.tensor(player_trust[i]))
    return loss

def fit_human_human_trust(player_count, player_trust, epsilon=1e-2):
    # y = model(x), l = loss(y, y'), loss.backward(), optimizer.step(), x is the trust variables, w are the alpha, beta, ws, wf parameters
    # Loop through everything in a batch
    trust_parameters = torch.tensor([50., 50.], requires_grad = True)
    w_0 = torch.tensor([50., 50.], requires_grad = True)
    numIter = 100
    loss_prev = 0
    while True:
        optimizer = torch.optim.SGD([trust_parameters, w_0], lr = 0.01)
        optimizer.zero_grad()
        modeled_trust = interhuman_model(player_count, trust_parameters, w_0)
        loss = loss_beta(modeled_trust, player_trust)
        if abs(loss - loss_prev) <= epsilon: break
        loss_prev = loss
        loss.backward()
        optimizer.step()
    return trust_parameters, w_0


numParticipants = 30
TIP_data_dict = {}
for file in glob.glob("TIP_data/*.csv"):
    person_idx = int(re.findall(r'\d+', file)[0])
    df = pd.read_csv(file)
    TIP_data_dict[person_idx] = {}
    for (columnName, columnData) in df.iteritems():
        TIP_data_dict[person_idx][columnName] = list(columnData.values)

# The sessions are performed in pairs in the order of person_idx
TIP_fitting_result = {}
predicted_err_sq, approx_err_sq = 0, 0
numFitting = 13
numSessions = 15
epsilon = 1e-3
for i in range(1, 30, 2):
    
    trust_params, w = fit_human_human_trust(TIP_data_dict[i+1]["player_count"][1:numFitting + 1], 
                                            TIP_data_dict[i]["trust_O"][:numFitting + 1], 
                                            epsilon=epsilon)
    trust_O = interhuman_model(TIP_data_dict[i+1]["player_count"][1:], trust_params, w)

    real_trust = np.array(TIP_data_dict[i]["trust_O"][numFitting + 1:])
    predicted_trust = np.array([float(trust_O[j][0]/(trust_O[j][1] + trust_O[j][0])) for j in range(numFitting + 1, numSessions + 1)])
    predicted_err_sq += np.sum((predicted_trust - real_trust) ** 2)
    approx_trust = np.array([TIP_data_dict[i]["trust_O"][numFitting] for _ in range(numSessions - numFitting)])
    approx_err_sq += np.sum((approx_trust - real_trust) ** 2)
    print(real_trust, predicted_trust, approx_trust)
    print(i, predicted_err_sq, approx_err_sq)

    trust_params, w = fit_human_human_trust(TIP_data_dict[i]["player_count"][1:numFitting + 1], 
                                            TIP_data_dict[i+1]["trust_O"][:numFitting + 1],
                                            epsilon=epsilon)
    trust_O = interhuman_model(TIP_data_dict[i]["player_count"][1:], trust_params, w)

    real_trust = np.array(TIP_data_dict[i+1]["trust_O"][numFitting + 1:])
    predicted_trust = np.array([float(trust_O[j][0]/(trust_O[j][1] + trust_O[j][0])) for j in range(numFitting+1, numSessions+1)])
    predicted_err_sq += np.sum((predicted_trust - real_trust) ** 2)
    approx_trust = np.array([TIP_data_dict[i+1]["trust_O"][numFitting] for _ in range(numSessions - numFitting)])
    approx_err_sq += np.sum((approx_trust - real_trust) ** 2)

    print(real_trust, predicted_trust, approx_trust)
    print(i + 1, predicted_err_sq, approx_err_sq)


