% Define hyperparameters for Minnesota prior


    % Prior mean on VAR regression coefficients
    % Random walk prior for VAR(p) with intercept: A_prior = [zeros(1,M); eye(M,M); zeros((p-1)*M,M)]
    % White noise prior for VAR(p) with intercept: A_prior = [zeros(1,M); zeros(M,M); zeros((p-1)*M,M)]
    %
    A_prior = [zeros(1,M); eye(M,M); zeros((p-1)*M,M)]; 
    a_prior = A_prior(:);               %<---- prior mean of alpha (parameter vector)
    
    % Minnesota Variance on VAR regression coefficients
    % First define the hyperparameters 'a_bar_i'
    % Aleksei: NB!!! The squares are takein in the code below, don't need
    % to square  here
    a_bar_1 = 1.0;          % lambda
    a_bar_2 = 1.0;          % theta
    a_bar_3 = 10^6;         % prior variance of intercept
    
    % Now get residual variances of univariate p-lag autoregressions. Here
    % we just run the AR(p) model on each equation, ignoring the constant
    % and exogenous variables (if they have been specified for the original
    % VAR model)
    sigma_sq = zeros(M,1); % vector to store residual variances
    Ylags=ones(Traw-p,1);
    for i = 1:M
        % Create lags of dependent variables
        Ylag_i = mlag2(Yraw(:,i),p);
        Ylag_i = Ylag_i(p+1:Traw,:);
        Ylags = [Ylags Ylag_i]; 
    end
    
     for i = 1:M
        % Dependent variable in i-th equation
        Y_i = Yraw(p+1:Traw,i);
        % OLS estimates of i-th equation
        alpha_i = inv(Ylags'*Ylags)*(Ylags'*Y_i);
        sigma_sq(i,1) = (1./(Traw-p+1))*(Y_i - Ylags*alpha_i)'*(Y_i - Ylags*alpha_i);
    end
    % Now define prior hyperparameters.
    % Create an array of dimensions K x M, which will contain the K diagonal
    % elements of the covariance matrix, in each of the M equations.
    
    V_i = zeros(M,M);
    
    
    % Aleksei code start
    if constant==1
        V_pr = [a_bar_3 * ones(M, 1)];
    else
        V_pr = [];
    end
    for l = 1:p
        for i = 1:M% for each i-th equation
             for j = 1:M   % for each j-th RHS variable
                 if i == j
                        V_i(i,j) = a_bar_1^2 /(l^2);
                 else
                        V_i(i,j) = (a_bar_1^2)*(a_bar_2^2)*sigma_sq(i,1) / ((l^2)*sigma_sq(j,1));
                 end  
             end
        end
        V_pr = [V_pr V_i];
    end
    V_pr = V_pr';
    V_prior = diag(V_pr(:));
    % Aleksei code end
    
    % Now V is a diagonal matrix with diagonal elements the V_i
%     V_prior = diag(V_i(:));  % this is the prior variance of the vector alpha
    
    %NOTE: No prior for SIGMA. SIGMA is simply a diagonal matrix with each
    %diagonal element equal to sigma_sq(i). See Kadiyala and Karlsson (1997)
    SIGMA = diag(sigma_sq);
    
