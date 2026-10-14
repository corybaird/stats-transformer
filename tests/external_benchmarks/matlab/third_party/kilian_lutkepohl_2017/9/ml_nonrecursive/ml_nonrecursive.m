% ML_NONRECURSIVE.M 
%
% Kilian and Luetkepohl (2017), Structural VAR Analysis, Cambridge University Press.
% This file generates ML results of Section 9.4

clear

global A SIGMAu p

data; [t,q]=size(y); p=4; 

% Gaussian conditional MLE (closed-form estimate of reduced-form model)
[A,SIGMA,U,V]=olsvarc(y,p);		
SIGMAu=U*U'/(t-p);	

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version 1 of method of moments
% Set seed (to ensure replicability of fsolve results)
randn('seed',1) 

% Set some options for fsolve
warning off
options=optimset('TolX',1e-4,'TolFun',1e-4,'MaxFunEvals',1e+50000,'MaxIter',1000);

% Identification based on nonrecursive short-run restrictions
OUTPUT=fsolve('version1',[eye(q) 1.5*ones(q,1)],options); 
B0=OUTPUT(1:q,1:q) 
SIGMAw=diag(OUTPUT(1:q,q+1)) 

% For completeness:
disp('Structural impact multiplier matrix')
disp(inv(B0)*SIGMAw^(0.5))


