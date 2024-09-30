clear all
close all

% model_parameters;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% loading model parameters %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('export_fig')
addpath('tight_subplot')
rng(12);
% rng(412);
global N ryy ryn rny rnn
N = 15;
d = rand(N,1); % pre-defined danger level
kappa1 = 3;
kappa2 = 50;
d_hat = betarnd(d*kappa2,(1-d)*kappa2);
d_tilde = betarnd(d*kappa1,(1-d)*kappa1);
wf = 20;
ws = 10;
SCap =800; % state capacity plotRegion = SCap -N*max(wf,ws);
discount_factor = 0.9;
q = [1,0.2]; % weight between health and time cost
h1 = 1;
h2 = 100;
c1 = 300;
c2 = 50;
c3 = 250;
c4 = 30;
ryy = q*[-h1,-c1]';
ryn = q*[-h2,-c2]';
rny = q*[0,-c3]';
rnn = q*[0,-c4]';

alpha_1 = 100;
beta_1 = 50;
% show the sampled danger level
if 1 == 2
    figure(10)
    axis(); hold on;
    plot(1:N,d_hat - d,'bo','linewidth',2)
    plot(1:N,d_tilde - d,'ro','linewidth',2)
    ylim([-1 1])
    legend('error of d hat','error of d tilde')
end
% runSimulation = true;
% runMatrixVersion = false;
runSimulation = false;
runMatrixVersion = true;

% select model parameters

% parameter for experiment 1
sensedTB = 1; % 1: reverse, 2: disuse
objective = 2;                % 1: task,    2: TrustSeeking

% parameter for experiment 2
actualTB = 1; %1: reverse, 2: disuse
simNum = 10000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% simulating the searching process %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if runSimulation
    rewardMeanHistory = [];
    rewardSTDHistory = [];
    trustMeanHistory = [];
    trustSTDHistory = [];
    resultHistoty = [];
    kappa = [2,50;50,2];
    fid = fopen('resultTextHistoty_untrusted_robot_low_accurate_robot.txt', 'a+');
    for alpha_1 = 50
        for beta_1 = 100
            for kappaSelection = 1:2
                %                 for kappa2 = 2
                for objective = 1:2    % 1: task,    2: TrustSeeking
                    for sensedTB = 1:2 % 1: reverse, 2: disuse
                        % parameter for experiment 2
                        for actualTB = 1:2 %1: reverse, 2: disuse
                            kappa1 = kappa(kappaSelection,1);
                            kappa2 = kappa(kappaSelection,2);
                            trustSum = 0;
                            trustHistory = zeros(simNum,1);
                            rewardSum = 0;
                            rewardHistory = zeros(simNum,1);
                            for j = 1:simNum
                                d = rand(N,1); % pre-defined danger level
                                d_hat = betarnd(d*kappa2,(1-d)*kappa2);
                                d_tilde = betarnd(d*kappa1,(1-d)*kappa1);
                                result = SimulatingActualSearch(alpha_1, beta_1, ws, wf, ...
                                    actualTB, sensedTB, objective, ...
                                    d, d_hat, d_tilde);
                                % resultT = array2table(result,'VariableNames',{'recomd','threat','humAction', ...
                                %         'alpha', 'beta','trust','d','d_hat','d_tilde', 'reward'})
                                trustSum = trustSum + result(N,6);
                                trustHistory(j,1) = result(N,6);
                                rewardSum = rewardSum + result(N,10);
                                rewardHistory(j,1) = result(N,10);
                            end
                            rewardMean  = rewardSum/simNum;
                            rewardSTD = std(rewardHistory);
                            trustMean  = trustSum/simNum;
                            trustSTD = std(trustHistory);
                            % figure()
                            % histogram(rewardHistory, 20)
                            rewardMeanHistory = [rewardMeanHistory;rewardMean];
                            rewardSTDHistory = [rewardSTDHistory;rewardSTD];
                            trustMeanHistory = [trustMeanHistory;trustMean];
                            trustSTDHistory = [trustSTDHistory;trustSTD];
                            dispText = ['alpha_1: ', num2str(alpha_1), ...
                                'beta_1: ', num2str(beta_1), ...
                                'kappa1: ', num2str(kappa1), ...
                                ', kappa2: ', num2str(kappa2), ...
                                ', obj: ', num2str(objective), ...
                                ', sensedTB: ', num2str(sensedTB), ...
                                ', actualTB: ', num2str(actualTB), ...
                                ', rewardMean: ', num2str(rewardMean),...
                                ', rewardSTD: ', num2str(rewardSTD)...
                                ', trustMean: ', num2str(trustMean),...
                                ', trustSTD: ', num2str(trustSTD) ];
                            disp(dispText);
                            fprintf(fid, dispText);
                            resultHistoty = [resultHistoty;...
                                [alpha_1,beta_1,kappa1,kappa2,objective,sensedTB,actualTB,rewardMean,rewardSTD,trustMean,trustSTD]];
                        end
                    end
                end
            end
        end
        %         end
    end
    csvwrite('resultTextHistoty_untrusted_robot_low_accurate_robot.csv',resultHistoty)
    fclose(fid);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% value/action matrix %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if runMatrixVersion
    for sensedTB = 1%:2 % 1: reverse, 2: disuse
        for objective = 2 % 1: task,    2: TrustSeeking
            % close all
            
            % Calculating value function and action function as matrix
            % the state set {S1(i),S2(i)}
            alpha = 1:SCap;
            beta = 1:SCap;
            [S1,S2] = meshgrid(alpha,beta);
            pFollow = S1./(S1+S2);
            pNotFollow = S2./(S1+S2);
            % value function and action function
            V = zeros([size(S1),N]);
            A = zeros([size(S1),N]);
            
            for k = 1: N % start from the last selection
                siteIndex = N+1-k;
                % calculate immediate reward
                [VY,VN] = ImmediateReward(pFollow, pNotFollow, siteIndex, ...
                    d_hat, d_tilde, sensedTB, objective);
                % update current value function and action function
                [A(:,:,siteIndex), V(:,:,siteIndex)] = UpdateValueFunctionMatrix(V, VY, VN, ...
                    SCap, ws, wf, discount_factor, siteIndex, d_hat, d_tilde);
            end
            
            % plotting the result of the value function and action function
            siteIndexSet = 1:ceil(N/8):N;
            figureNumber = sensedTB*10 + objective;
            plotValueActionMatrix(siteIndexSet, SCap, ...
                ws, wf, A, V, d_hat, figureNumber)
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% helper functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function result = SimulatingActualSearch(alpha_0, beta_0, ws, wf, ...
    actualTB, sensedTB, objective, ...
    d, d_hat, d_tilde)
% this function simulate one task and output the result
global N
alpha = alpha_0;
beta = beta_0;
alpha_beta_history = zeros(N+1,2);
alpha_beta_history(1,:) = [alpha, beta];
threatPresenceSeq = d > rand(N, 1);
reward = 0;
result = zeros(N,10);
for siteIndex = 1: N
    % calculate the best action
    recommendation =  CalculateBestActoin(siteIndex, alpha, beta, ws, wf,...
        d_hat, d_tilde, sensedTB, objective);
    % reveal the truth and udpate trust
    % human action is determined by the trust behavior model
    humanAction = GenerateHumanAction(alpha, beta, d_tilde(siteIndex),...
        actualTB, recommendation);
    [alpha, beta] = UpdateTrust(alpha,beta,threatPresenceSeq(siteIndex),recommendation, ws, wf);
    reward = UpdateReward(reward, threatPresenceSeq(siteIndex), humanAction);
    alpha_beta_history(siteIndex + 1,:) = [alpha, beta];
    result(siteIndex, :) = [recommendation, threatPresenceSeq(siteIndex), ...
        humanAction, alpha, beta, alpha/(alpha+beta), d(siteIndex), ...
        d_hat(siteIndex), d_tilde(siteIndex), reward];
end

end




function bestAction = CalculateBestActoin(siteIndex, alpha, beta, ws, wf,...
    d_hat, d_tilde, sensedTB, objective)
global N
% calculate the best action at the current site by solving a sub POMDP
% rename the index of the current site as 1, thus the index of the last
% site is n = N - siteIndex + 1. And convert other variables into a
% subproblem from site siteIndex to N
n = N - siteIndex + 1;
d_hat = d_hat(siteIndex:end,:);
d_tilde = d_tilde(siteIndex:end,:);
% we need two matrix coding all the possible alpha and beta values during
% the process.
Alpha = zeros(n+1);
Beta = zeros(n+1);
A = zeros(n+1);
V = zeros(n+1);
% update the value/action reversely: subSiteIndex from n to 1
% first we let the last column of the value function be zero
V(:, n + 1) = zeros(n + 1, 1);
% second we update the value function and action function reversely.
for k = 1: n
    subSiteIndex = n - k + 1;
    % update the column of the experienceMatrix
    for j = 1: subSiteIndex
        Alpha(j, subSiteIndex) = alpha + (subSiteIndex - j) * ws;
        Beta(j, subSiteIndex)  = beta + (j - 1) * wf;
    end
    pFollow = Alpha(1:subSiteIndex,subSiteIndex)./...
        (Alpha(1:subSiteIndex,subSiteIndex)+Beta(1:subSiteIndex,subSiteIndex));
    pNotFollow = ones(subSiteIndex,1) - pFollow;
    % calculate immediate reward
    [VY,VN] = ImmediateReward(...
        pFollow, pNotFollow, subSiteIndex,...
        d_hat, d_tilde, sensedTB, objective);
    % update current value function and action function
    if subSiteIndex == 1
        d_k = d_hat(subSiteIndex);
    else
        d_k = d_tilde(subSiteIndex);
    end
    % case Y:
    % action Y danger Y: prob dk      ,next state (alpha + ws, beta)
    % action Y danger N: prob 1 - dk  ,next state (alpha, beta + wf)
    VY = ...
        VY + ...
        d_k      *   V(1:subSiteIndex,   subSiteIndex+1) + ...
        (1-d_k)  *   V(2:subSiteIndex+1, subSiteIndex+1);
    % case N:
    % action N danger Y: prob dk,     ,next state (alpha, beta + wf)
    % action N danger N: prob 1-dt    ,next state (alpha + ws, beta)
    VN = ...
        VN + ...
        (1-d_k)  *   V(1:subSiteIndex,   subSiteIndex+1) + ...
        d_k      *   V(2:subSiteIndex+1, subSiteIndex+1);
    V(1:subSiteIndex,subSiteIndex) = max(VY,VN);
    A(1:subSiteIndex,subSiteIndex) = VY>VN;
end
bestAction = A(1,1);
end



function [VY,VN] = ImmediateReward(...
    pFollow, pNotFollow, siteIndex, ...
    d_hat, d_tilde, sensedTB, objective)
% use the robot sensed danger level for the current site
% use reported prior for the future sites

global ryy ryn rny rnn
if objective == 1
    trustReward = 0;
elseif objective == 2
    trustReward = 80/(1+exp(0.5*siteIndex));
end

if sensedTB == 1 % reverse
    if siteIndex == 1
        d_k = d_hat(siteIndex);
    else
        d_k = d_tilde(siteIndex);
    end
    VY = ...
        pNotFollow * d_k     * (ryn + trustReward) + ...
        pFollow    * d_k     * (ryy + trustReward) + ...
        pNotFollow * (1-d_k) *  rnn + ...
        pFollow    * (1-d_k) *  rny;
    VN = ...
        pFollow    * d_k     *  ryn + ...
        pNotFollow * d_k     *  ryy + ...
        pFollow    * (1-d_k) *  (rnn + trustReward) + ...
        pNotFollow * (1-d_k) *  (rny + trustReward);
elseif sensedTB == 2 % disuse
    
    d_tilde_k = d_tilde(siteIndex);
    % unless at the current (1st) house, the robot will use d_tilde_k
    % to estimate d_k
    if siteIndex == 1
        d_hat_k = d_hat(siteIndex);
    else
        d_hat_k = d_tilde_k;
    end
    
    VY = ...
        (ryn + trustReward) * pNotFollow*(1-d_tilde_k)*d_hat_k +...
        (ryy + trustReward) * (pFollow+pNotFollow*d_tilde_k)*d_hat_k + ...
        rnn                 * pNotFollow*(1-d_tilde_k)*(1-d_hat_k) + ...
        rny                 * (pFollow+pNotFollow*d_tilde_k)*(1-d_hat_k);
    
    VN = ...
        ryn                 * ( pFollow+pNotFollow*(1-d_tilde_k) )*d_hat_k +...
        ryy                 * pNotFollow*d_tilde_k*d_hat_k + ...
        (rnn + trustReward) * (pFollow+pNotFollow*(1-d_tilde_k) )*(1-d_hat_k) + ...
        (rny + trustReward) * pNotFollow*d_tilde_k*(1-d_hat_k);
else
    error('unseen behavior')
    
end

end


function [ACurrentSite, VCurrentSite] = UpdateValueFunctionMatrix(V, VY, VN, ...
    SCap, ws, wf, discount_factor, siteIndex, d_hat, d_tilde)
% update value and action function for the matrix version
global N
if siteIndex == 1
    d_k = d_hat(siteIndex);
else
    d_k = d_tilde(siteIndex);
end
if  siteIndex < N
    V_next_temp = ...
        discount_factor *...
        [V(:,:,siteIndex+1),zeros(SCap,ws); zeros(wf,SCap+ws)];
    % case Y:
    % action Y danger Y: prob dk      ,next state (alpha + ws, beta)
    % action Y danger N: prob 1 - dk  ,next state (alpha, beta + wf)
    VY = ...
        VY + ...
        d_k      *   V_next_temp(1:SCap,1 + ws:SCap + ws) + ...
        (1-d_k)  *   V_next_temp(1 + wf:SCap + wf, 1 : SCap);
    % case N:
    % action N danger Y: prob dk,     ,next state (alpha, beta + wf)
    % action N danger N: prob 1-dt    ,next state (alpha + ws, beta)
    VN = ...
        VN + ...
        d_k      *   V_next_temp(1 + wf:SCap + wf, 1 : SCap) + ...
        (1-d_k)  *   V_next_temp(1:SCap,1 + ws:SCap + ws);
end
ACurrentSite = VY >= VN;
VCurrentSite = max(VY,VN);
end

function plotValueActionMatrix(siteIndexSet, SCap, ...
    ws, wf, A, V, d_hat, figureNumber)
global N
% figure(figureNumber)
% set(gcf,'position',[600 100 900 400])
% set(gcf, 'Color', 'w')

figure(figureNumber)
set(gcf,'position',[100 100 1300 300])
[ha, pos] = tight_subplot(2,8,[.03 .04],[.06 .03],[.03 .03]);
set(gcf, 'Color', 'w')
plotRegion = min(SCap -N*max(wf,ws),500);
totalPlotNumber = length(siteIndexSet);



Vmin = min(min(min(V(1:plotRegion,1:plotRegion,:))));
Vmax = max(max(max(V(1:plotRegion,1:plotRegion,:))));

for plotIndex = 1:totalPlotNumber
    
    k = siteIndexSet(plotIndex);
    
    %     subplot(2,totalPlotNumber,plotIndex)
    axes(ha(plotIndex))
    imshow(A(1:plotRegion,1:plotRegion,k))
    tmp = A(1:plotRegion,1:plotRegion,k)
    %     xlabel('$$\alpha_k$$','interpreter','latex')
    %     ylabel('$$\beta_k$$','interpreter','latex')
    h = gca;
    h.Visible = 'On';
    set(gca,'YDir','normal')
    %     set(gca,'Xtick',0:100:1000)
    titleText = ['site ' ,num2str(k),', $$\hat{d}_{',num2str(k),'}$$ = ', num2str(d_hat(k),'%0.2f ')];
    title(titleText ,'interpreter','latex')
    
    %     subplot(2,totalPlotNumber,plotIndex+totalPlotNumber)
    axes(ha(plotIndex+totalPlotNumber))
    Vmin = min(min(V(1:plotRegion,1:plotRegion,k)));
    Vmax = max(max(V(1:plotRegion,1:plotRegion,k)));
    imshow(V(1:plotRegion,1:plotRegion,k),[Vmin, Vmax])
    %     xlabel('$$\alpha_k$$','interpreter','latex')
    %     ylabel('$$\beta_k$$','interpreter','latex')
    h = gca;
    h.Visible = 'On';
    set(gca,'YDir','normal')
    %     title(['k = ' num2str(k) ', $$\hat{d}_k$$ = ' num2str(d_hat(k))] ,'interpreter','latex')
    
end
switch  figureNumber
    case 11
        exportTitle = 'POMDP_V_A_case_11_rev_mission.pdf';
    case 12
        exportTitle = 'POMDP_V_A_case_12_rev_trust.pdf';
    case 21
        exportTitle = 'POMDP_V_A_case_21_dis_mission.pdf';
    case 22
        exportTitle = 'POMDP_V_A_case_22_dis_trust.pdf';
    otherwise
        error('figureNumber wrong')
end
export_fig(exportTitle)

end

function [alpha_updated, beta_updated] = ...
    UpdateTrust(alpha,beta,threatPresence,recommendation, ws, wf)
if threatPresence - recommendation == 0
    alpha_updated = alpha + ws;
    beta_updated = beta;
else
    alpha_updated = alpha;
    beta_updated = beta + wf;
end

end

function     humanAction = GenerateHumanAction(alpha, beta, d_tilde_siteIndex,...
    actualTB, recommendation)
% generate what the solider will fo after given the robot's
% recommendation.
trust = alpha/(alpha + beta);
if trust>rand(1)
    humanAction = recommendation;
elseif actualTB == 1
    humanAction = 1 - recommendation;
elseif actualTB == 2
    humanAction = d_tilde_siteIndex>rand(1);
else
    error('unknown human behavior')
end
end

function reward_updated = UpdateReward(reward, threatPresence, humanAction)
% update task reward according to the soldier's action and actual
% danger existence
global ryy ryn rny rnn
if threatPresence == 1
    if humanAction == 1
        reward_updated = reward + ryy;
    else
        reward_updated = reward + ryn;
    end
else
    if humanAction == 1
        reward_updated = reward + rny;
    else
        reward_updated = reward + rnn;
    end
end
end