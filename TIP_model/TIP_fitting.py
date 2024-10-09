import numpy as np
import random
import pandas as pd
import os
import glob
import re
import torch


numSites = 10

alpha_human_human_0, beta_human_human_0 = 5., 5.
alpha_human_robot_0, beta_human_robot_0 =5., 5.
wf_human_robot_direct, ws_human_robot_direct = 10., 10. 
wf_human_human, ws_human_human = 10., 10.
wf_human_robot_indirect, ws_human_robot_indirect = 10., 10.

def get_trust(trust_params):
    return trust_params[0] / (trust_params[0] + trust_params[1])

def full_model(person1, person2, trust_hh, trust_hr, w_direct, w_hh, w_indirect, numFitting):
    result_A, result_B, result_O = [], [], []
    trust_hr_A, trust_hr_B = trust_hr, trust_hr
    result_A.append(trust_hr_A)
    result_B.append(trust_hr_B)
    result_O.append(trust_hh)
    for i in range(1, numFitting + 1):
        # perform one session
        if person1["assignment"][i] == 'A':
            trust_hr_A = trust_hr_A + torch.mul(torch.tensor([person1["robot_count"][i], numSites - person1["robot_count"][i]]), w_direct)
            trust_hh = trust_hh + torch.mul(torch.tensor([person2["player_count"][i] / numSites, 1 - person2["player_count"][i] / numSites]), w_hh)
            trust_gap = person2["trust_B"][i] - get_trust(trust_hr_B)
            indirect_update_alpha = trust_gap if trust_gap > 0 else 0
            indirect_update_beta = -trust_gap if -trust_gap > 0 else 0
            #trust_hr_B = trust_hr_B + torch.mul(torch.tensor([indirect_update_alpha, indirect_update_beta]), w_indirect)
            trust_hr_B = trust_hr_B + get_trust(trust_hh) * torch.mul(torch.tensor([indirect_update_alpha, indirect_update_beta]), w_indirect)
        else:
            trust_hr_B = trust_hr_B + torch.mul(torch.tensor([person1["robot_count"][i], numSites - person1["robot_count"][i]]), w_direct)
            trust_hh = trust_hh + torch.mul(torch.tensor([person2["player_count"][i] / numSites, 1 - person2["player_count"][i] / numSites]), w_hh)
            trust_gap = person2["trust_A"][i] - get_trust(trust_hr_A)
            indirect_update_alpha = trust_gap if trust_gap > 0 else 0
            indirect_update_beta = -trust_gap if -trust_gap > 0 else 0
            #trust_hr_A = trust_hr_A + torch.mul(torch.tensor([indirect_update_alpha, indirect_update_beta]), w_indirect)
            trust_hr_A = trust_hr_A + get_trust(trust_hh) * torch.mul(torch.tensor([indirect_update_alpha, indirect_update_beta]), w_indirect)
        
        result_A.append(trust_hr_A)
        result_B.append(trust_hr_B)
        result_O.append(trust_hh)
        
    return result_A, result_B, result_O
        
def loss_beta(predicted_A, predicted_B, predicted_O, player_A, player_B, player_O):
    loss = 0
    for i in range(len(predicted_A)):
        b1 = torch.distributions.Beta(predicted_A[i][0], predicted_A[i][1])
        loss = loss - b1.log_prob(torch.tensor(player_A[i]))

        b2 = torch.distributions.Beta(predicted_B[i][0], predicted_B[i][1])
        loss = loss - b2.log_prob(torch.tensor(player_B[i]))

        b3 = torch.distributions.Beta(predicted_O[i][0], predicted_O[i][1])
        loss = loss - b3.log_prob(torch.tensor(player_O[i]))
    return loss

def fit_TIP_trust(player_idx, player1, player2, numFitting=numSites, epsilon=1e-2):
    # y = model(x), l = loss(y, y'), loss.backward(), optimizer.step(), x is the trust variables, w are the alpha, beta, ws, wf parameters
    # Loop through everything in a batch
        
    trust_hh = torch.tensor([alpha_human_human_0, beta_human_human_0], requires_grad = True)
    trust_hr = torch.tensor([alpha_human_robot_0, beta_human_robot_0], requires_grad = True)
    w_direct = torch.tensor([ws_human_robot_direct, wf_human_robot_direct], requires_grad = True)
    w_hh = torch.tensor([ws_human_human, wf_human_human], requires_grad = True)
    w_indirect = torch.tensor([ws_human_robot_indirect, wf_human_robot_indirect], requires_grad = True)
    loss_prev = 0
    i = 0
    while True:
        optimizer = torch.optim.SGD([trust_hh, trust_hr, w_direct, w_hh, w_indirect], lr = 0.0001)
        optimizer.zero_grad()
        predicted_A, predicted_B, predicted_O = full_model(player1, player2, trust_hh, trust_hr, w_direct, w_hh, w_indirect, numFitting)
        i+=1
        loss = loss_beta(predicted_A, predicted_B, predicted_O, player1["trust_A"], player1["trust_B"], player1["trust_O"])
        if i % 100 == 0:
            print(trust_hh, trust_hr, w_direct, w_hh, w_indirect)
            print(loss)
        if abs(loss-loss_prev) < epsilon:
            break
        loss_prev = loss
        loss.backward()
        optimizer.step()
    f2 = open(f"TIP_parameters_numFitting={numFitting}.txt", "a+")
    f2.write(f"Player {player_idx}\n")
    f2.write(f"trust_hh: {trust_hh[0]}, {trust_hh[1]}\n")
    f2.write(f"trust_hr: {trust_hr[0]}, {trust_hr[1]}\n")
    f2.write(f"w_direct: {w_direct[0]}, {w_direct[1]}\n")
    f2.write(f"w_hh: {w_hh[0]}, {w_hh[1]}\n")
    f2.write(f"w_indirect: {w_indirect[0]}, {w_indirect[1]}\n")
    f2.close()
    return trust_hh, trust_hr, w_direct, w_hh, w_indirect

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
predicted_err_sq = 0
predicted_err_sq_A = 0
predicted_err_sq_B = 0

predicted_human_err_sq, approx_human_err_sq = 0, 0
numFitting = 15
numSessions = 15
epsilon = 1e-2
for i in range(1, 31, 2):
    f = open(f"TIP_error_numFitting={numFitting}.txt", "a+")
    trust_hh, trust_hr, w_direct, w_hh, w_indirect = fit_TIP_trust(i, TIP_data_dict[i], TIP_data_dict[i+1], numFitting=numFitting, epsilon=epsilon)
    trust_A, trust_B, trust_O = full_model(TIP_data_dict[i], TIP_data_dict[i+1], trust_hh, trust_hr, w_direct, w_hh, w_indirect, numFitting=numSessions)
    """
    real_trust_A = np.array(TIP_data_dict[i]["trust_A"][numFitting + 1:])
    real_trust_B = np.array(TIP_data_dict[i]["trust_B"][numFitting + 1:])
    real_trust_O = np.array(TIP_data_dict[i]["trust_O"][numFitting + 1:])

    predicted_A = np.array([float(trust_A[j][0]/(trust_A[j][1] + trust_A[j][0])) for j in range(numFitting+1, numSessions+1)])
    predicted_B = np.array([float(trust_B[j][0]/(trust_B[j][1] + trust_B[j][0])) for j in range(numFitting+1, numSessions+1)])
    predicted_O = np.array([float(trust_O[j][0]/(trust_O[j][1] + trust_O[j][0])) for j in range(numFitting+1, numSessions+1)])

    approx_A = np.array([TIP_data_dict[i]["trust_A"][numFitting] for _ in range(numSessions - numFitting)])
    approx_B = np.array([TIP_data_dict[i]["trust_B"][numFitting] for _ in range(numSessions - numFitting)])
    approx_O = np.array([TIP_data_dict[i]["trust_O"][numFitting] for _ in range(numSessions - numFitting)])
    
    f.write(f"Player {i}\n")
    predicted_err_sq += (np.sum((predicted_A - real_trust_A) ** 2) + np.sum((predicted_B - real_trust_B) ** 2))    
    f.write(f"Trust A: {real_trust_A}, {predicted_A}, {approx_A}\n")
    f.write(f"Trust B: {real_trust_B}, {predicted_B}, {approx_B}\n")
    f.write(f"Trust O: {real_trust_O}, {predicted_O}, {approx_O}\n")
    f.write(f"predicted_err_sq:, {predicted_err_sq}\n")
    predicted_human_err_sq += np.sum((predicted_O - real_trust_O) ** 2)
    approx_human_err_sq += np.sum((approx_O - real_trust_O) ** 2)
    f.write(f"predicted_human_err_sq: {predicted_human_err_sq}, approx_human_err_sq: {approx_human_err_sq}\n")
    """

    real_trust_A = np.array(TIP_data_dict[i]["trust_A"][:])
    real_trust_B = np.array(TIP_data_dict[i]["trust_B"][:])

    predicted_A = np.array([float(trust_A[j][0]/(trust_A[j][1] + trust_A[j][0])) for j in range(numSessions+1)])
    predicted_B = np.array([float(trust_B[j][0]/(trust_B[j][1] + trust_B[j][0])) for j in range( numSessions+1)])

    f.write(f"Player {i}\n")
    predicted_err_sq_A += (np.sum((predicted_A - real_trust_A) ** 2))
    predicted_err_sq_B += (np.sum((predicted_B - real_trust_B) ** 2))
    predicted_err_sq += (predicted_err_sq_A + predicted_err_sq_B)
    f.write(f"Trust A: {real_trust_A}, {predicted_A}\n")
    f.write(f"Trust B: {real_trust_B}, {predicted_B}\n")
    f.write(f"predicted_err_sq:, {predicted_err_sq}\n")
    f.write(f"predicted_err_sq_A:, {predicted_err_sq_A}, predicted_err_sq_B: {predicted_err_sq_B}\n")
    
    trust_hh, trust_hr, w_direct, w_hh, w_indirect = fit_TIP_trust(i+1, TIP_data_dict[i+1], TIP_data_dict[i], numFitting=numFitting, epsilon=epsilon)
    trust_A, trust_B, trust_O = full_model(TIP_data_dict[i+1], TIP_data_dict[i], trust_hh, trust_hr, w_direct, w_hh, w_indirect, numFitting=numSessions)

    """
    real_trust_A = np.array(TIP_data_dict[i+1]["trust_A"][numFitting + 1:])
    real_trust_B = np.array(TIP_data_dict[i+1]["trust_B"][numFitting + 1:])
    real_trust_O = np.array(TIP_data_dict[i+1]["trust_O"][numFitting + 1:])

    predicted_A = np.array([float(trust_A[j][0]/(trust_A[j][1] + trust_A[j][0])) for j in range(numFitting+1, numSessions+1)])
    predicted_B = np.array([float(trust_B[j][0]/(trust_B[j][1] + trust_B[j][0])) for j in range(numFitting+1, numSessions+1)])
    predicted_O = np.array([float(trust_O[j][0]/(trust_O[j][1] + trust_O[j][0])) for j in range(numFitting+1, numSessions+1)])

    approx_A = np.array([TIP_data_dict[i+1]["trust_A"][numFitting] for _ in range(numSessions - numFitting)])
    approx_B = np.array([TIP_data_dict[i+1]["trust_B"][numFitting] for _ in range(numSessions - numFitting)])
    approx_O = np.array([TIP_data_dict[i+1]["trust_O"][numFitting] for _ in range(numSessions - numFitting)])

    predicted_err_sq += (np.sum((predicted_A - real_trust_A) ** 2) + np.sum((predicted_B - real_trust_B) ** 2))

    f.write(f"Player {i+1}\n")
    f.write(f"Trust A: {real_trust_A}, {predicted_A}, {approx_A}\n")
    f.write(f"Trust B: {real_trust_B}, {predicted_B}, {approx_B}\n")
    f.write(f"Trust O: {real_trust_O}, {predicted_O}, {approx_O}\n")
    f.write(f"predicted_err_sq:, {predicted_err_sq}\n")
    predicted_human_err_sq += np.sum((predicted_O - real_trust_O) ** 2)
    approx_human_err_sq += np.sum((approx_O - real_trust_O) ** 2)
    f.write(f"predicted_human_err_sq: {predicted_human_err_sq}, approx_human_err_sq: {approx_human_err_sq}\n")
    """

    
    real_trust_A = np.array(TIP_data_dict[i]["trust_A"][:])
    real_trust_B = np.array(TIP_data_dict[i]["trust_B"][:])

    predicted_A = np.array([float(trust_A[j][0]/(trust_A[j][1] + trust_A[j][0])) for j in range(numSessions+1)])
    predicted_B = np.array([float(trust_B[j][0]/(trust_B[j][1] + trust_B[j][0])) for j in range( numSessions+1)])

    f.write(f"Player {i + 1}\n")
    predicted_err_sq_A += (np.sum((predicted_A - real_trust_A) ** 2))
    predicted_err_sq_B += (np.sum((predicted_B - real_trust_B) ** 2))
    predicted_err_sq += (predicted_err_sq_A + predicted_err_sq_B)
    f.write(f"Trust A: {real_trust_A}, {predicted_A}\n")
    f.write(f"Trust B: {real_trust_B}, {predicted_B}\n")
    f.write(f"predicted_err_sq:, {predicted_err_sq}\n")
    f.write(f"predicted_err_sq_A:, {predicted_err_sq_A}, predicted_err_sq_B: {predicted_err_sq_B}\n")

    f.close()

