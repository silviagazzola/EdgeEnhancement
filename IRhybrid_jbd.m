function [X, info] = IRhybrid_jbd(A, b, varargin)

% IRhybrid_jbd Hybrid version of a joint bidiagonalization (JBD) algorithm, 
% handling Tikhonov regularization in general form
%
% options  = IRhybrid_jdb('defaults')
% [X,info] = IRhybrid_jbd(A,b)
% [X,info] = IRhybrid_jdb(A,b,K)
% [X,info] = IRhybrid_jbd(A,b,options)
% [X,info] = IRhybrid_jbd(A,b,K,options)
%
% IRhybrid_jbd is a hybrid iterative regularization method used for 
% solving large-scale, ill-posed inverse problems of the form:
%               b = A*x + noise .
% The method is based on a JBD algorithm that can 
% naturally be used with Tikhonov regularization in general form. 
% The approximation susbpace for the solution computed by JBD is comparable 
% to the GSVD solution subspace, and it is expanded at each iteration.
% Each iteration of the JBD algorithm computes projections by running some 
% LSQR iterations. 
%
% With 'defaults' as input returns the default options.  Otherwise outputs
% the iterates specified in K, using max(K) as MaxIter, and using all other
% default options.  With options as input: uses the user-specified options
% and all the other default options.
%
% Inputs:
%  A : either (a) a full or sparse matrix
%             (b) a matrix object that performs the matrix*vector operation
%             (c) user-defined function handle
%  b : right-hand side vector
%  K : (optional) integer vector that specifies which iterates are returned
%      in X; the maximum number of iterations is assumed to be max(K)
%      [ positive integer | vector of positive components ]
%  options : structure with the following fields (optional)
%      x_true     - true solution; allows us to returns error norms with
%                   respect to x_true at each iteration
%                   [ array | {'none'} ]
%      RegParam   - a value or a method to find the regularization
%                   parameter for the projected problems: 
%                   [  non-neg. scalar | 'discrepit' | 'Lcurve' ]
%      NoiseLevel - norm of noise in rhs divided by norm of rhs (must be
%                   assigned if RegParam is 'discrep')
%                   [ {'none'} | nonnegative scalar ]
%      eta        - safety factor for the discrepancy principle
%                   [ {1.1} | scalar greater than (and close to) 1 ]
%      MaxIter    - maximum number of iterations
%                   [ {'none'} | positive integer ]
%      RegMatrix  - regularization matrix for Tikhonov regularization (in general form)
%                   [ {'identity'} | 'Laplacian1D' | 'Laplacian2D' |
%                   square nonsingular matrix | function handle | 
%                   'Gradient1D' | 'Gradient2D' ] 
%      IterBar    - shows the progress of the iterations
%                   [ {'on'} | 'off' ]
%      NoStop     - specifies whether the iterations should proceed
%                   after a stopping criterion is satisfied
%                   [ 'on' | {'off'} ]
%        AllX     - specifies whether the approximate solution should be
%                   computed at each iteration
%                   [ 'on' | {'off'} ]
%                   (if 'on' an additional LSQR iteration cycle must be run
%                   at each iteration)
%      InProjIt   - maximum number of LSQR iterations to compute the
%                   projection at each JBD iteration
%                   [ {'none'} | positive integer ]
%      jbdTol     - tolerance for the computation of the projections via
%                   LSQR (at each JBD iteration)
%                   [ {10^-4} | non-negative scalar ]
%      jbdTolx    - tolerance for the computation of the projections via
%                   LSQR (to compute the approximate solution)
%                   [ {10^-4} | non-negative scalar ]
%  RegParamVect0  - vector of potential regularization parameter values,
%                   given in increasing order (used if RegParam is 'Lcurve')
%                   [ {fliplr(logspace(2,-3,30))} | vector with nonnegative entries ]
%
% Note: the options structure can be created using the function IRset. 
%
% Outputs:
%   X : computed solutions, stored column-wise (at the iterations listed in K)
%   info: structure with the following fields:
%      its      - number of the last computed iteration
%      Xnrm     - solution norms at each iteration
%      Rnrm     - relative residual norms at each iteration
%      Enrm     - relative error norms at each iteration (requires x_true)
%      RegP     - sequence of the regularization parameters
%    XnrmLcurve - 2D array storing the norm of the approximate solution 
%                 for each sampled value of the regularization parameter 
%                 and at each iteration
%                 (is RegParam is 'Lcurve')
%    RnrmLcurve - 2D array storing the norm of residual 
%                 for each sampled value of the regularization parameter 
%                 and at each iteration
%                 (is RegParam is 'Lcurve')
%      idxVect  - vector of indeces of the regularization parameters selected  
%                 by the L-curve criterion at each iteration (among the
%                 ones in RegParamVect0)
%                 (is RegParam is 'Lcurve')
%
% See also: IRcgls, IRhybrid_lsqr

% Reference:
% S. Gazzola, M. E. Kilmer, J. G. Nagy, O. Semerici, E. L. Miller
% An Inner-Outer Iterative Method for Edge Preservation in Image Restoration and Reconstruction
% arXiv, math.NA 1912.13103, 2019

% Silvia Gazzola, University of Bath
% Misha E. Kilmer, Tufts University
% Eric L. Miller, Tufts University
% James G. Nagy, Emory University
% June 2020.

% Initialization
defaultopt = struct('MaxIter', 100, 'InProjIt', 500, ...
    'x_true', 'none', 'AllX', 'off', 'NoStop','on', 'IterBar', 'on',...
    'RegParam','Lcurve','RegMatrix', 'Gradient2D',...
    'jbdTol', 1e-4, 'jbdTolx', 1e-4, ...
    'NoiseLevel', 'none', 'eta', 1.1, 'RegParamVect0', fliplr(logspace(2,-3,30)));

if nargin == 0
    error('Not enough input arguments')
elseif nargin == 1 
    % If input is 'defaults,' return the default options in X
    if nargout <= 1 && isequal(A,'defaults')
        X = defaultopt;
        return;
    else
        error('Not enough input arguments')
    end
end

defaultopt.restart = 'off';
defaultopt.verbosity = 'on';
% tolerance for the weights
defaultopt.weight0 = 'none';
defaultopt.tolX    = 1e-10;
defaultopt.qnorm   = 1;

% Check for acceptable number of optional input arguments
switch length(varargin)
    case 0 
        K = []; options = [];
    case 1
        if isa(varargin{1}, 'double')
            K = varargin{1}; options = [];
        else
            % no matter the order of appearance
            K = []; options = varargin{1};
        end
    case 2
        if isa(varargin{1}, 'double')
            K = varargin{1}; options = varargin{2};
        else
            % again, no matter the order of appearance
            K = varargin{2}; options = varargin{1};
        end
        if isfield(options, 'MaxIter') && ~isempty(options.MaxIter) && (~isempty(K) && options.MaxIter ~= max(K))
            warning('The value of MaxIter is discarded; the maximum value in K is taken as MaxIter')
        end 
    otherwise
        error('Too many input parameters')
end

if isempty(options)
    options = defaultopt;
end

options = IRset(defaultopt, options);

MaxIter       = IRget(options, 'MaxIter',      [], 'fast');
InProjIt      = IRget(options, 'InProjIt',     [], 'fast');
RegParam      = IRget(options, 'RegParam',     [], 'fast');
x_true        = IRget(options, 'x_true',       [], 'fast');
AllX          = IRget(options, 'AllX',         [], 'fast');
NoStop        = IRget(options, 'NoStop',       [], 'fast');
IterBar       = IRget(options, 'IterBar',      [], 'fast');
L             = IRget(options, 'RegMatrix',    [], 'fast');
jbdTol        = IRget(options, 'jbdTol',       [], 'fast');
jbdTolx       = IRget(options, 'jbdTolx',      [], 'fast');
NoiseLevel    = IRget(options, 'NoiseLevel',   [], 'fast');
eta           = IRget(options, 'eta',          [], 'fast');
RegParamVect0 = IRget(options, 'RegParamVect0',[], 'fast');
restart       = IRget(options, 'restart',      [], 'fast');

% verbose = strcmp(verbose, 'on');

% adaptWGCV = strcmp(RegParam, {'wgcv'}) && strcmp(omega, {'adapt'});

% setting K
if isempty(K)
    K = MaxIter;
end
% sorting the iterations (in case they are shuffled in input)
K = K(:); K = sort(K,'ascend'); K = unique(K);
if ~((isreal(K) && (all(K > 0)) && all(K == floor(K))))
    error('K must be a vector of positive real integers')
end
if K(end) ~= MaxIter
    MaxIter = K(end);    
end
% note that there is no control on K, as it does not go through IRset

if (strcmp(RegParam,'discrep') || strcmp(RegParam,'discrepit')) && ischar(NoiseLevel)
    error('The noise level (NoiseLevel) must be assigned')
end

nlambda = length(RegParamVect0);

d = Atransp_times_vec(A, b);
n = length(d);
m = length(b);

if ismatrix(L)
    [mL,nL] = size(L);
    if nL ~= n
        error('The number of columns of the regularization matrix should be the same as the length of x')
    end
else
    error('Currently accepting only regularization operators stored as matrices')
end

inputproj.n = n;
inputproj.m = m;
inputproj.mL = mL;

r = b(:);

% means no true solution
notrue = strcmp(x_true,'none');
AllX = strcmpi(AllX, 'on');
% means we do not want to stop when the stopping criterion is satisfied
NoStop = strcmp(NoStop,'on');
if ~NoStop
    warning('A stopping criterion for the joint bidiagonalization iterations is not implemented at the moment. The maximum number of iterations will be performed.')
end

if strcmp(RegParam,'optimal') && notrue
    error('The exact solution must be assigned (to compute the optimal regularization parameter)')
end

if strcmp(RegParam,'off')
    RegParam = 0;
end
if isscalar(RegParam)
    if isempty(NoiseLevel) || strcmp(NoiseLevel,'none')
        NoiseLevel = 0;
    else
        NoiseLevel = eta*NoiseLevel;
    end
end

beta = norm(r(:));
nrmb = norm(b(:));

len=m+mL; 
if strcmpi(RegParam,'Lcurve')
    P=spalloc(MaxIter,MaxIter,MaxIter); 
    XnrmLcurve = zeros(nlambda,max(K));
    RnrmLcurve = zeros(nlambda,max(K));
    idxVect = zeros(max(K));
end

% Declare matrices.
X            = zeros(n,length(K));
Xnrm         = zeros(max(K),1);
Rnrm         = zeros(max(K),1);
RegParamVect = zeros(max(K),1);
%Ri           = zeros(max(K),max(K)+1);
if notrue
    errornorms = false;
else
    errornorms = true;
    if AllX
        Enrm       = zeros(max(K),1);
    end
    nrmtrue = norm(x_true(:));
end

% Main Code Begins Here
phi = beta*ones(1,nlambda); 
Util_matr = zeros(m+mL, MaxIter+1);
Uhat_matr = zeros(m+mL, MaxIter);
U_matr = zeros(m, MaxIter+1);
Vtil_matr = zeros(m+mL, MaxIter);
Vhat_matr = zeros(m+mL, MaxIter);
B_matr = zeros(MaxIter+1, MaxIter);
Bbar_matr = zeros(MaxIter, MaxIter);

util = [b;zeros(mL,1)]/beta;
Util_matr(:,1) = util;
U_matr(:,1) = util(1:m);

[ptil, ~, res1] = lsqr_proj_hybrid(A, L, util, jbdTol, InProjIt, inputproj);  % ptil ~= QQ' util
if length(res1) == InProjIt
    disp('Warning: the number of inner iterations was reached.')
    disp('The relative residual upon return was: ')
    res1(length(res1))
end

alpha=norm(ptil);
vtil=ptil/alpha; 


Bi(1,1)=alpha; 
B_matr(1,1) = alpha;
uhat=zeros(len,1); betahat=1;

noIterBar = strcmp(IterBar,{'off'});
if ~noIterBar
  h_wait = waitbar(0, 'Running iterations, please wait ...');
end
% j = 0;
for k=1:MaxIter
    if ~noIterBar
        waitbar(k/MaxIter, h_wait)
    end
    Vtil_matr(:,k) = vtil; % lines 2 and 5
    %% computes the equivalent of the Q2 piece
    vhat=(-1)^(k+1)*vtil;
    Vhat_matr(:,k) = vhat;
    rhat=[zeros(m,1);vhat(m+1:len)]- betahat*uhat;
    alphahat=norm(rhat); uhat=rhat/alphahat;  
    Uhat_matr(:,k) = uhat;
    Bbar_matr(k,k) = alphahat; 
    if k>1, Bbar_matr(k-1,k) = betahat; end % Bbar_matr(j,j-1) not assigned if j=1
                                              
    %% save alphahat, betahat for later use
    alphahat_old=alphahat; betahat_old=betahat;
    
    %% computes the equivalent of the Q1 piece
    rtil=[vtil(1:m);zeros(mL,1)] - alpha*util;
    beta=norm(rtil); util=rtil/beta;   %Bi(j+1,j)=beta; 
    Util_matr(:,k+1) = util;
    U_matr(:,k+1) = util(1:m);
    B_matr(k,k) = alpha;
    B_matr(k+1,k) = beta;
      
    % save alpha, beta for later use
    alpha_old=alpha; beta_old=beta; vtil_old=vtil;
    
    
    %keep doing the update for the Q1 piece
    [prj,~,res1]=lsqr_proj_hybrid(A,L,util,jbdTol,InProjIt,inputproj); %prj ~= Q Q' util
    if length(res1)==InProjIt
        disp('Warning: during external iteration, the number of inner iterations in call to projection was reached.')
    end
    ptil=prj-beta*vtil;
    
    alpha=norm(ptil);  Bi(k+1,k+1)=alpha;
    vtil=ptil/alpha;
      
    % computes the rest of the Q2 piece
    betahat=(alpha*beta)/alphahat; Ri(k,k)=alphahat; Ri(k,k+1)=betahat;
    P(k,k)=(-1)^(k+1);
    Rihat=Ri(1:k,1:k)*P(1:k,1:k);
    
    % Applying the DP here
    if strcmpi(RegParam,'discrepit') 
        B_matr_temp = B_matr(1:k+1,1:k);
        rhsproj = [nrmb; zeros(k,1)];
        % check if the discrepancy principle can be satisfied
        ynoreg = B_matr_temp\rhsproj;
        discr = norm(B_matr_temp*ynoreg - rhsproj);
        if discr<=eta*nrmb*NoiseLevel
            RegParamk = fzero(@(l)discrfcn(l, B_matr_temp, Rihat, rhsproj, nrmb, eta*NoiseLevel), [0, 1e10]);
        else
            RegParamk = 1e-8;
        end
        RegParamVect(k) = RegParamk; 
        yklam = [B_matr_temp; RegParamk*Rihat]\[rhsproj; zeros(k,1)];
        Rnrm(k) = norm(B_matr_temp*yklam - rhsproj)/nrmb;
        Vylam = Vtil_matr(:,1:k)*yklam;
        if AllX %% then the solutions for chosen lambda at each iteration, are requested.
            [~,xk]=lsqr_proj_hybrid(A,L,Vylam,jbdTolx,InProjIt,inputproj);       
            X(:,k)=xk;
            if errornorms
                Enrm(k)=norm(xk - x_true)/nrmtrue;
            end
        end
        temp = Vylam;
    % Inner loop (on k) over the lambda values
    % This computes the QR factorization of [Bj;lam(k)*Rj*Pj]; only one rotation
    % is needed for each loop on j
    elseif strcmpi(RegParam,'Lcurve') 
%         alphaJ = zeros(nlambda,1);
%         betaJ = zeros(nlambda,1);
%         alphabarJ = zeros(nlambda,1);
%         betabarJ = zeros(nlambda,1);
%         theta = zeros(nlambda,1);
%         epsilon = zeros(nlambda,1);
%         ctheta = zeros(nlambda,1);
%         stheta = zeros(nlambda,1);
%         csuper = zeros(nlambda,1);
%         ssuper = zeros(nlambda,1);
%         cd1 = zeros(nlambda,1);
%         sd1 = zeros(nlambda,1);
%         csub = zeros(nlambda,1);
%         sssub = zeros(nlambda,1);
%         phiold = zeros(nlambda,1);
        for l=1:nlambda
            alphaJ(l)=alpha_old; betaJ(l)=beta_old; 
            alphabarJ(l)=RegParamVect0(l)*((-1)^(k+1))*alphahat_old; 
               %%potential sign change takes care of post-mult of Ri by P
            if k>1
                betabarJ(l)=RegParamVect0(l)*((-1)^(k+1))*betahat_old; 
               %%potential sign change takes care of post-mult of Ri by P
            end     
      %Now apply the rotation from the last iteration to
      %the newest entries in the bidiagonal matrices and rhs vector
            if k >=3 
                theta(l)=ssuper(l)*betabarJ(l);
                betabarJ(l)=csuper(l)*betabarJ(l);
            end
            if k >=2
                bb=betabarJ(l);
                betabarJ(l)=-sd1(l)*alphaJ(l)+cd1(l)*bb;
                alphaJ(l)=cd1(l)*alphaJ(l)+sd1(l)*bb;
        
                epsilon(l)=alphaJ(l)*sssub(l);
                alphaJ(l)=alphaJ(l)*csub(l);
        
                phi(l)=-sssub(l)*phiold(l); myres=abs(phi(l));
            end
    
        % Now compute the rotations based on the current
        % columns of the matrices
            if k >=3
                d = sqrt(theta(l)^2 + betabarJ(l)^2);
                ctheta(l)=betabarJ(l)/d; stheta(l)=-theta(l)/d;
                betabarJ(l)=d;
            end
            if k >=2
                d = sqrt(betabarJ(l)^2 + alphabarJ(l)^2);
                csuper(l)=alphabarJ(l)/d; ssuper(l)=-betabarJ(l)/d;
                alphabarJ(l)=d;
            end
            if k >= 1
                d=sqrt(alphabarJ(l)^2 + betaJ(l)^2);
                cd1(l)=betaJ(l)/d; sd1(l)=alphabarJ(l)/d;
                betaJ(l)=d;        
                d=sqrt(alphaJ(l)^2+betaJ(l)^2);
                csub(l)=alphaJ(l)/d; sssub(l)=betaJ(l)/d;
                alphaJ(l)=d;
                phiold(l)=phi(l); phi(l)=csub(l)*phi(l); myres=abs(-sssub(l)*phiold(l));
                %% myres contains the residual norm of the system
                %% [A;lam(k)*B]*xk - [b;zeros(s,1)]
                %% which is equivalent to the residual norm of the system
                %% [B1;lam(k)*B2]*xk - beta*e_1 
                %% when j=1, phi(k) contains the first entry in f.  
                %% when j>1, phi(k) becomes the jth entry in f *before* this last set of rots is
                %% applied (at the beginning of loop), so this right. 
            end
        
            %%  update the recursions (in the paper) if we want to find the
            %%  solution estimate for each j for each lambda value.  If only doing
            %%  parameter estimation and don't need solutions, the next several lines
            %%  can be omitted.
            if k==1
                D(:,l)=(1/alphaJ(l))*vtil_old;
                S(:,l)=phi(l)*D(:,l);    
            else
                D(:,l)=(1/alphaJ(l))*(vtil_old-epsilon(l)*D(:,l));
                S(:,l)=S(:,l)+phi(l)*D(:,l);
                % note:  the j,j element of the upper triangular factor is now
                %   (alphaJ(k)) and that the new j-1,j element is epsilon(k).  
            end
        
            %%  based on the comments in the above if-then-loop, we are trying to set up the
            %%  info to implement Malena's recurrence for the norms. 
            if k==1 
                Dbar(:,l)=(1/alphaJ(l))*Rihat(k,k); 
                Sbar(:,l)=phi(l)*Dbar(:,l); 
            else
 
                dold=[Dbar(:,l);0]; 
                bsize=length(dold);
                if k>2
                    Bbarold=[zeros(bsize-2,1);Rihat(k-1,k);Rihat(k,k)];
                else
                    Bbarold=[Rihat(k-1,k);Rihat(k,k)];
                end
            
                newDbar(:,l)=(1/alphaJ(l))*(-epsilon(l)*dold+Bbarold);
                newSbar(:,l)=[Sbar(:,l);0]+phi(l)*newDbar(:,l);
            end 
       
        
            % NOTE: the following norms are not relative
            if k==1
                XnrmLcurve(l,k) = norm(Sbar);
                RnrmLcurve(l,k)=sqrt(myres^2-RegParamVect0(l)^2*XnrmLcurve(l,k)^2);
            else
                XnrmLcurve(l,k)=norm(newSbar(:,l));
                RnrmLcurve(l,k)=sqrt(myres^2-RegParamVect0(l)^2*XnrmLcurve(l,k)^2);
            end
     
        end %%end loop over lambdas (i.e. loop on l)
        %% reset Dbar; Sbar for next iteration j
        if k>1
            clear Dbar Sbar
            Dbar=newDbar; 
            Sbar=newSbar;
            clear newDbar newSbar
        end
        %% now, compute the corner of the L-curve for iteration j 
        if length(RegParamVect0)>5
            if k>1
                idx=pch_corner(XnrmLcurve(:,k),RnrmLcurve(:,k));
            else
                idx = 1;
            end
            RegParamVect(k) = RegParamVect0(idx);
            Xnrm(k) = XnrmLcurve(idx,k); 
            Rnrm(k) = RnrmLcurve(idx,k)/nrmb; %%%
            idxVect(k) = idx;
        end

        if AllX %% then the solutions for chosen lambda at each iteration, are requested.
            [~,xk]=lsqr_proj_hybrid(A,L,S(:,idx),jbdTolx,InProjIt, inputproj);       
            X(:,k)=xk;
            if errornorms
                Enrm(k)=norm(xk - x_true)/nrmtrue;
            end
        end
        temp = S(:,idx);
    end
end %% end loop on number of iterations (i.e. on k)

if ~AllX
    [~,xk]=lsqr_proj_hybrid(A,L,temp,jbdTolx,InProjIt, inputproj);      
    X = xk;
    if errornorms
        Enrm=norm(xk - x_true)/nrmtrue;
    end
end

if ~noIterBar, close(h_wait), end
if nargout==2
  info.its = k;
  if errornorms
    info.Enrm = Enrm;
  end
  info.Xnrm = Xnrm;
  info.Rnrm = Rnrm;
  info.RegP = RegParamVect;
  if strcmpi(RegParam,'Lcurve') 
      info.XnrmLcurve = XnrmLcurve;
      info.RnrmLcurve = RnrmLcurve;
      info.idxVect = idxVect;
  end
end

function [prj,x,Rnrm,NE_Rnrm,U,V]=lsqr_proj_hybrid(A,L,b,ctol,k,inputproj)
%% prj is the approximate orthogonal projection of the approx solution onto range of the matrix.
%% x is the approximate solution.   LSQR based.


mA = inputproj.m;
mL = inputproj.mL;
n  = inputproj.n;
                  % so B is n1xn2 and B' is n2 x n1. 
                  % we are assuming A is m1 x m2.
m = mA + mL;      % m is the number of rows in the stacked matrix [A;B]. n is the number of columns.
                  
if nargout>4
    flag=1;
else
    flag=0;
end

Rnrm    = zeros(k,1);
NE_Rnrm = zeros(k,1);

v = zeros(n,1); x = v; nrmb = norm(b); beta = nrmb;
if (beta==0), error('Right-hand side must be nonzero'), end

u=b/beta;

tmp1 = Atransp_times_vec(A, u(1:mA));
tmp2 = Atransp_times_vec(L, u(mA+1:mA+mL)); 

r = tmp1 + tmp2; 
  
alpha=norm(r);  % cctol=alpha*beta*ctol;  %cctol = norm(A'*b)*ctol

nrmAtb = alpha*beta; % norm(tmp1)*nrmb;
v=r/alpha;

phibar=beta; rhobar=alpha; w=v;
% initialize for projection estimate
prj=zeros(m,1); mk=zeros(m,1); theta=0;
if flag, U(:,1)=u; V(:,1)=v; end

for i=2:k
    %keep alpha_(i-1) and beta_(i-1) and u(i-1)
    % remember, alpha_1 is the 1,1 entry in Bk, but beta_1 is not in Bk
    
    alphaold=alpha; uold=u;
    %% M = the stacked matrix [A;B].
    %% find M*v-alpha*u, then normalize to get u_i
    %  u=M*v-alpha*u; beta=norm(u); u=u/beta;
    tmp(1:mA,1) = A_times_vec(A,v);
    tmp(mA+1:mA+mL,1) = A_times_vec(L,v);
     
    u=tmp-alpha*u; beta=norm(u); u=u/beta;
    
    %% M = the stacked matrix [A;B], so M'=[A',B'];
    %% find M'*u-beta*v, then normalize to get v_i
    %% v=M'*u-beta*v; alpha=norm(v); v=v/alpha;  
    
    clear tmp
    
    tmp1 = Atransp_times_vec(A, u(1:mA)); 
    tmp2 = Atransp_times_vec(L, u(mA+1:mA+mL)); 
    tmp  = tmp1 + tmp2; 
    v=tmp - beta*v; alpha=norm(v); v=v/alpha;
    
    if flag, U(:,i)=u; V(:,i)=v; end
    
    %% find and apply Givens rotation. Givens is applied to
    %% [ rho_bar 0; beta alpha]. It's a *real* rotation, so this only
    %% works for real matrices A right now. The rotation has the form
    %% [c s; s -c] 
    
    rho=pythag(rhobar,beta); %R(i-1,i-1)
    cr=rhobar/rho; sr=beta/rho;  thetaold=theta;
    theta=sr*alpha; rhobar=-cr*alpha;  %theta=R(i-1,i), current r(i,i)
    phi=cr*phibar; phibar=sr*phibar;
    
    
    % Update the solution.  This actually gives x_(i-1)
    
    x = x + (phi/rho)*w; w = v - (theta/rho)*w;
 
    Rnrm(i-1)=abs(phibar)/nrmb;          %%norm(A*x-b)/norm(b);
    NE_Rnrm(i-1)=abs(alpha*cr*phibar)/nrmAtb; %%norm(A'*(A*x-b))/norm(A'*b); 
    
   
    % Update the projection. This actually gives prj_(i-1)
    nk=alphaold*uold+beta*u;
    mk=(1/rho)*(nk-thetaold*mk);
    prj=prj+phi*mk;
    
    %detect stagnation; a hack, but necessary
    if i > 10
        if abs(Rnrm(i-1)-Rnrm(i-2))/Rnrm(i-2) < ctol % 1e-4
            Rnrm = Rnrm(1:i-1);
            NE_Rnrm = NE_Rnrm(1:i-1);
            break
        end
    end
    %if res2(i-1)<cctol, break; end
end

function x = pythag(y,z) 
%PYTHAG Computes sqrt( y^2 + z^2 ). 
% 
% x = pythag(y,z) 
% 
% Returns sqrt(y^2 + z^2) but is careful to scale to avoid overflow. 
 
% Christian H. Bischof, Argonne National Laboratory, 03/31/89. 
 
rmax = max(abs([y;z])); 
if (rmax==0) 
  x = 0; 
else 
  x = rmax*sqrt((y/rmax)^2 + (z/rmax)^2); 
end 

function [index, info] = pch_corner( res, nhx, show)
  
%disp('in corner')
if (nargin < 3) || isempty(show)
	show = 0;		% default is no graphs
end

info = 0;

fin = isfinite(res+nhx);	% a nan of inf may cause trouble
nzr = res.*nhx~=0;		% a zero may cause trouble
kept = find(fin&nzr);
if isempty(kept)
	error( 'too many inf/Nan/zeros found in data')
end
if size(kept,1) < size(res,1)
	info = info + 1;
	disp( 'Corner:badData inf/Nan/zeros found in data')
end
res = res(kept);
nhx = nhx(kept);

if any(res(1:end-1)<res(2:end)) || any(nhx(1:end-1)>nhx(2:end))
	info = info + 10;
	%warning( 'Corner:lackOfMonotonicity', 'lack of monotonicity')
end

% initialization
nP = length(res);		% number of points
P = log10([res nhx]);		% these are the points of the L-curve
V = P(2:nP,:)-P(1:nP-1,:);	% these are the vectors defined by them
v = sqrt(sum(V.^2,2)); % compute the lenght of the vectors

W = V./[v v];                   % normalize vectors

% Choose the p longest vectors
clist = [];			% list of candidates
[~,I] = sort(v);
I = flipud(I);

p = min(5, nP-1);
convex = 0;

while p < (nP-1)*2
	elmts = sort(I(1:min(p, nP-1)));
	candidate = rt( W(elmts,:), elmts);
	if candidate>0
		convex = convex + 1;
	end
	if candidate && ~any(clist==candidate)
		clist = [clist;candidate];
	end
	candidate = LCurveCorner(P, W(elmts,:), elmts);
	if ~any(clist==candidate)
		clist = [clist; candidate];
	end
	p = p*2;
end

if sum(clist==1) == 0
	clist = [1;clist];
end
clist = sort(clist);

if convex==0
	index = [];
	info = info + 100;
	warning( 'Corner:lackOfConvexity', 'lack of convexity')
	return
end

% select the vectors (joining candidates) for which the norm blows up
vz = find( diff(P(clist,2)) >= abs(diff(P(clist,1))) );
if length(vz)>1
	if(vz(1) == 1),  vz = vz(2:end);  end
elseif length(vz)==1
	if(vz(1) == 1),  vz = [];  end
end
if isempty(vz)		% take the last candidate if the vectors are OK
	index = clist(end);
else			% otherwise the best candidate marks the transition 
	vects = [P(clist(2:end),1)-P(clist(1:end-1),1) ...
			P(clist(2:end),2)-P(clist(1:end-1),2)];
        lengths = sqrt(sum(vects.^2,2));
        vects = vects./[lengths lengths];
	delta = vects(1:end-1,1).*vects(2:end,2) ...
			- vects(2:end,1).*vects(1:end-1,2);  % wedge products
	vv = find(delta(vz-1)<=0);
	if isempty(vv)
		index = clist(vz(end));
	else
		index = clist(vz(vv(1)));
	end
end

index = kept(index);

if show
	figure(show);
	plot(P(:,1),P(:,2),'-o',P(clist,1),P(clist,2),'r*', ...
				P(index,1),P(index,2),'g*')
	axis equal
	title( sprintf( 'L-curve - index=%g', index))
	xlabel( '||Ax-b||_2')
	ylabel('||Hx||_2');
end


function index = rt( W, kv)
delta = W(1:end-1,1).*W(2:end,2) - W(2:end,1).*W(1:end-1,2);  % wedge products
[mm, kk] = min(delta);
if mm < 0		% is it really a corner?
	index = kv(kk) + 1;
else			% if there's no corner it returns 0
	index = 0;
end


function corner = LCurveCorner(P, vects, elmts)

angles = acos(-vects(:,1));

[~, In] = sort(angles);

count = 1;
ln = length(In);
mn = In(1);
mx = In(ln);
while(mn>=mx)
	mx = max([mx In(ln-count)]);
	count = count + 1;
	mn = min([mn In(count)]);
end
if count > 1
	I = 0; J = 0;
	for i=1:count
		for j=ln:-1:ln-count+1
			if(In(i) < In(j))
				I = In(i); J = In(j); break
			end
		end
		if I>0, break; end
	end
else
	I = In(1); J = In(ln);
end

% Extrapolated point of intersection
x3 = P(elmts(J)+1,1)+(P(elmts(I),2)-P(elmts(J)+1,2))/(P(elmts(J)+1,2) ...
			-P(elmts(J),2))*(P(elmts(J)+1,1)-P(elmts(J),1));
origin = [x3 P(elmts(I),2)];

% Find distances from the original L-curve to the intersection of the
% most horizontal and the most vertical vector. The intersection is given by
% pointToHit.
dists = (origin(1)-P(:,1)).^2+(origin(2)-P(:,2)).^2;
[~,corner] = min(dists);


