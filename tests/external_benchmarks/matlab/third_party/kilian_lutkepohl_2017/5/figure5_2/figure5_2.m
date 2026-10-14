% Figure5_2.m 
%
% This code adapted from code for an empirical illustration 
% in Koop and Korobilis (2009). It generates one panel of
% Figure 5.2 of 
% Kilian and Luetkepohl (2017), Structural VAR Analysis, Cambridge University Press 
%
% Minnesota prior
%
% The specification of the prior hyperparmeters are in the file
% prior_hyper_Minnesota.m. 
%
% ALPHA is the K x M matrix of VAR coefficients,
% alpha is the KM x 1 column vector of vectorized VAR coefficients, i.e.
% alpha = vec(ALPHA), and SIGMA is the M x M VAR covariance matrix.

clear all;
clc;
randn('seed',2); 
rand('seed',2); 

%------------------------------LOAD DATA-----------------------------------
% Load Quarterly US data on inflation, unemployment and interest rate, 
% 1953:Q1 - 2006:Q3
load Yraw.dat;
% 'Yraw' is a matrix with T rows by M columns,
% where T is the number of time series observations (usually months or
% quarters), while M is the number of VAR dependent macro variables.

%----------------------------PRELIMINARIES---------------------------------
% Define specification of the VAR model
constant = 1;        % 1: if you desire intercepts, 0: otherwise 
p = 4;               % Number of lags on dependent variables
%h = 1;               % Number of forecast periods
impulses = 1;        % 1: compute impulse responses, 0: no impulse responses
ihor = 25;           % Horizon to compute impulse responses

% Gibbs-related preliminaries
nsave = 10000;         % Final number of draws to save
nburn = 0;         % Draws to discard (burn-in)
% For models using analytical results, there are no convergence issues (You
% are not adviced to change the next 3 lines)
% if prior ==1 || prior == 2 || prior == 3
%    nburn = 0*nburn;
% end
ntot = nsave + nburn;  % Total number of draws
it_print = 2000;       % Print on the screen every "it_print"-th iteration
%--------------------------DATA HANDLING-----------------------------------
% Get initial dimensions of dependent variable
[Traw M] = size(Yraw)
Y1 = Yraw;
Y2 = Yraw;
        
% Generate lagged Y matrix. This will be part of the X matrix
Ylag = mlag2(Y2,p); % Y is [T x M]. ylag is [T x (Mp)]

% Now define matrix X which has all the R.H.S. variables (constant, lags of
% the dependent variable and exogenous regressors/dummies).
if constant
    X1 = [ones(Traw-p,1) Ylag(p+1:Traw,:)];
else
    X1 = Ylag(p+1:Traw,:);  %#ok<UNRCH>
end

% Get size of final matrix X
[Traw3 K] = size(X1);

% Create the block diagonal matrix Z
Z1 = kron(eye(M),X1);

% Form Y matrix accordingly
% Delete first "LAGS" rows to match the dimensions of X matrix
Y1 = Y1(p+1:Traw,:); % This is the final Y matrix used for the VAR

% Traw was the dimesnion of the initial data. T is the number of actual 
% time series observations of Y and X
T = Traw - p;
Y = Y1;
X = X1;
Z = Z1;

%========= IMPULSE RESPONSES SET-UP:
% Create matrices to store impulse responses
  % imp_infl = zeros(nsave,M,ihor); % impulse responses to a shock in inflation
  % imp_une = zeros(nsave,M,ihor);  % impulse responses to a shock in unemployment
    imp_int = zeros(nsave,M,ihor);  % impulse responses to a shock in interest rate
    bigj = zeros(M,M*p);
    bigj(1:M,1:M) = eye(M);

%-----------------------------PRELIMINARIES--------------------------------
% First get ML estimators
A_OLS = inv(X'*X)*(X'*Y); % This is the matrix of regression coefficients
a_OLS = A_OLS(:);         % This is the vector of parameters, i.e. it holds
                          % that a_OLS = vec(A_OLS)
SSE = (Y - X*A_OLS)'*(Y - X*A_OLS);   % Sum of squared errors
SIGMA_OLS = SSE./(T-K+1);

% Initialize Bayesian posterior parameters using OLS values
alpha = a_OLS;     % This is the single draw from the posterior of alpha
ALPHA = A_OLS;     % This is the single draw from the posterior of ALPHA
SSE_Gibbs = SSE;   % This is the SSE based on each draw of ALPHA
SIGMA = SIGMA_OLS; % This is the single draw from the posterior of SIGMA

% Storage space for posterior draws
alpha_draws = zeros(nsave,K*M);   % save draws of alpha
ALPHA_draws = zeros(nsave,K,M);   % save draws of alpha
SIGMA_draws = zeros(nsave,M,M);   % save draws of ALPHA

% load file which sets hyperparameters for chosen prior
prior_hyper_Minnesota;
    
%========================== Start Sampling ================================
%==========================================================================
tic;
disp('Number of iterations');
for irep = 1:ntot  %Start the Gibbs "loop"
    if mod(irep,it_print) == 0 % print iterations
        disp(irep);
        toc;
    end
    
    %--------- Draw ALPHA and SIGMA with Minnesota Prior
        for i = 1:M
            V_post = inv( inv(V_prior((i-1)*K+1:i*K,(i-1)*K+1:i*K)) + inv(SIGMA(i,i))*X'*X );
            a_post = V_post*(inv(V_prior((i-1)*K+1:i*K,(i-1)*K+1:i*K))*a_prior((i-1)*K+1:i*K,1) + inv(SIGMA(i,i))*X'*Y(:,i));
            alpha((i-1)*K+1:i*K,1) = a_post + chol(V_post)'*randn(K,1); % Draw alpha
        end
        ALPHA = reshape(alpha,K,M); % Create draw in terms of ALPHA
        
        % SIGMA in this case is a known matrix, whose form is decided in
        % the prior (see prior_hyper.m)
    % =============Estimation ends here
    
    % ****************************Impulse Responses***************************
    if irep > nburn               
               %=========IMPULSE RESPONSES:
        if impulses==1
            % ------------Identification code I:
            Bv = zeros(M,M,p);
            for i_1=1:p
                Bv(:,:,i_1) = ALPHA(1+((i_1-1)*M + 1):i_1*M+1,:)';               
            end
                        
            % st dev matrix for structural VAR
            shock = chol(SIGMA);
            %d = diag(diag(shock));
            %shock = inv(d)*shock;
            
            [responses]=impulse(Bv,shock,ihor);
            
            % Restrict to policy shocks
             responses1 = squeeze(responses(:,1,:));
             responses2 = squeeze(responses(:,2,:));
             responses3 = squeeze(responses(:,3,:));
            %responses3 = squeeze(responses(:,14,:));
            
             imp_infl(irep-nburn,:,:) = responses1;
             imp_une(irep-nburn,:,:) = responses2;
            imp_int(irep-nburn,:,:) = responses3;
                    
        end
               
        %----- Save draws of the parameters
        alpha_draws(irep-nburn,:) = alpha;
        ALPHA_draws(irep-nburn,:,:) = ALPHA;
        SIGMA_draws(irep-nburn,:,:) = SIGMA;
        
    end % end saving results
       
end %end the main Gibbs for loop
%====================== End Sampling Posteriors ===========================

%Posterior mean of parameters:
ALPHA_mean = squeeze(mean(ALPHA_draws,1)); %posterior mean of ALPHA
SIGMA_mean = squeeze(mean(SIGMA_draws,1)); %posterior mean of SIGMA

%Posterior standard deviations of parameters:
ALPHA_std = squeeze(std(ALPHA_draws,1)); %posterior std of ALPHA
SIGMA_std = squeeze(std(SIGMA_draws,1)); %posterior std of SIGMA

    % Set quantiles from the posterior density of the impulse responses
    qus = [.1, .5, .9];
    imp_resp_infl = squeeze(quantile(imp_infl,qus));       
    imp_resp_une = squeeze(quantile(imp_une,qus));
    imp_resp_int = squeeze(quantile(imp_int,qus));
 
 %======= Plot impulse responses
    figure
    set(0,'DefaultAxesColorOrder',[0 0 0],...
        'DefaultAxesLineStyleOrder','--|-|--')
    plot(squeeze(imp_resp_int(:,1,:))')
    hold;
    plot(zeros(ihor),'-')
    %title('...')
    xlim([1 ihor])
    set(gca,'XTick',0:4:ihor)

% Clear screen and print elapsed time
clc;
toc;

