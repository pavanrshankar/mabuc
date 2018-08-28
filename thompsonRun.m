%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Thompson Sampling Bandit Player.
% Naive TS bandit player that ignores intuition at each round.
%
% (c) 2014 Pedro A. Ortega <pedro.ortega@gmail.com>
%     2015 Modified by Andrew Forney
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%---------DONE---------

function [Action, Reward, Prob, Conds] = thompsonRun(theta, T, allFactors, pObs)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialize.
%Priors are assumed as we are not aware of underlying payout rate for actions
s = [1, 1];
f = [1, 1];

Action = zeros(1, T);
Reward = zeros(1, T);
Conds  = zeros(1, 4); % B, D binary
Prob   = zeros(1, T); %Prob of success

%% Execute one run.
for t=1:T
    roundFactors = allFactors(:,t);
    B = roundFactors(1);
    D = roundFactors(2);
    Z = roundFactors(3);
    covariateIndex = B + D * 2 + 1;%00,01,10,11
    Conds(covariateIndex) = Conds(covariateIndex) + 1;
    
    % Choose action parameters
    theta_hat = [betarnd(s(1), f(1)), betarnd(s(2), f(2))];
    
    %action parameter and value
    [maxVal, action] = max(theta_hat);
    
    %current Theta is payout rate
    currentTheta = theta(action, covariateIndex);

    % Pull lever. We'll find the index of theta through some clever maths
    % with the value of B and D (see covariate index)
    reward = rand <= currentTheta;

    % Update.
    s(action) = s(action) + reward;
    f(action) = f(action) + 1 - reward;

    % Record.
    Action(t) = (action == 1);
    Reward(t) = reward;
    
    %Compares actions from row1 and 2 of theta table
    [bestVal, bestAction] = max([theta(1, covariateIndex), theta(2, covariateIndex)]);
    Prob(t) = action == bestAction;
end

