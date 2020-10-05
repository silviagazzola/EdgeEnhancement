% This script executes the first tomographic example among the ones
% illustrated in:
% S. Gazzola, M. E. Kilmer, J. G. Nagy, O. Semerici, E. L. Miller
% An Inner-Outer Iterative Method for Edge Preservation in Image Restoration and Reconstruction
% arXiv, math.NA 1912.13103, 2019

% To generate the test problem, functions form AIR Tools II ( jakobsj/AIRToolsII ) are needed.

% Silvia Gazzola, University of Bath, June 2020

clear, clc, close all
%% set the test problem
angles = 0:2:130;
n =128;
phantomImage = 'grains';
ProbOptions = PRtomo('defaults');
ProbOptions = PRset(ProbOptions, 'angles', angles, 'phantomImage', phantomImage);
[A, b_true, x_true, ProbInfo] = PRtomo(n, ProbOptions);

NoiseLevel = 1e-3;
b = PRnoise(b_true, NoiseLevel);

%% set the parameters for the solver
q = 2; % exponent for the weights
MaxIts = 20; % maximum number of outer iterations
tau = 1; % threshold for the stopping criterion
outerits = 60; % this is number of outer iterations for tikh_prj_hybrid.  Don't have a good stopping criteria beyond outerits - something 
               % like what is used in "hybR" perhaps.   Right now, there's
               % a hack to detect if the selected parameter is consistent
               % over the last few previous steps, and if so, then it's
               % probably converged.
innerits =500; % At each step of tikh_prj_hybrid, we have to compute an (appx) orthogonal projection, which
               % we do in LSQR-like fashion.  Inner its refers to the
               % number of inner iterations to compute the projection. This is often reached before the tolerance kicks. 


%% Generate the regularization operator.
N=n^2;
I = speye(n);
c = [-1,1];
nd = n-1;
L = sparse(nd,n);
L = L + sparse(1:nd,(1:nd),c(1)*ones(1,nd),nd,n);
L = L + sparse(1:nd,(1:nd)+1,c(2)*ones(1,nd),nd,n);

Dx = kron(I,L); Dy=kron(L,I);
sz = size(Dx,1);
L = [Dx;Dy];
L = sparse(L);
D = L;
L = 10*L;    % this factor helps to make [A;L] better conditioned - which helps significantly in speeding the 
             % convergence for the innermost iteration which computes the
             % orthogonal projection


% Choose some lambda values, to try if lambda is set according to the L-curve criterion
nlam = 30;
lam = (logspace(2,-3,nlam));
lam = fliplr(lam);

% Setting font size, line width, and marker sizes for plots:
FS = 18;
LW = 2;
MS = 6;

%% new weights; selecting the regularization parameter via the L-curve               
% Setting the solver options
AllX = 'off';
optn.MaxIter = outerits;
optn.InProjIt = innerits;
optn.RegParam = 'Lcurve';
optn.RegParamVect0  = lam;
optn.x_true = x_true;
optn.AllX = AllX;
% Allocating storage for the outputs
if strcmp(AllX, 'on'), RelErrorNorms_Lcurve = zeros(outerits, MaxIts); end
RelErrorFinal_Lcurve = zeros(MaxIts, 1);
bigRho_Lcurve = zeros(nlam, MaxIts);
bigEta_Lcurve = zeros(nlam, MaxIts);
Lambda_Lcurve = zeros(MaxIts, 1);
X_Lcurve = zeros(N, MaxIts);
Lgx_Lcurve = zeros(MaxIts, 1);
WeightsNew_Lcurve = zeros(2*sz, MaxIts);
Weights_Lcurve = zeros(2*sz, MaxIts);
figure(1);
% inizialization
stopped = 0;
Wk = speye(2*sz);
for i = 1:MaxIts
    B=Wk*L; %current regularization operator, fed to routine below
    optn.RegMatrix = B;
    [X, info] = IRhybrid_jbd(A, b, optn);
    sp = sprintf('Iteration: %d  RelError: %0.5g  Lambda selected: %0.5g',i,info.Enrm(end),info.RegP(end));
    disp(sp)
    if strcmp(AllX, 'on'), RelErrorNorms_Lcurve(:,i) = info.Enrm; end
    RelErrorFinal_Lcurve(i) = info.Enrm(end); 
    bigRho_Lcurve(:,i)=info.RnrmLcurve(:,end); 
    bigEta_Lcurve(:,i) = info.XnrmLcurve(:,end);
    Lambda_Lcurve(i) = info.RegP(end); % store the parameter that gave the 'optimal' solution this round.
    X_Lcurve(:,i) = X;   % store the solution.
    
    %%% definition (and updates) of the weights
    tmp = (Wk*L)*X;
    tmp = abs(tmp)/norm(tmp,Inf);
    
    if i > 1
        Lgx0=Lgx; % value at the previous iteration
    end
    Lgx = norm(D*X);
    Lgx_Lcurve(i) = Lgx;
    
    w = (1 - tmp.^q); 
    newW = spdiags(w,0,2*sz,2*sz);  % this is the diagonal matrix that will update the current reg operator.
    Wk = newW*Wk; % Compute the current weighting matrix for the next round.  Note that these are diagonal, so the product commutes.
    WeightsNew_Lcurve(:,i) = w; 
    Weights_Lcurve(:,i) = diag(Wk); % store the weighting matrix that was used in the call above.
    
    if i > 1
        if Lgx < tau*Lgx0
            stopped = 1;
            break
        end
    end
    
    XX=(reshape(X,n,n)); 
    mytitle=sprintf('L-curve, Iteration %d; $\\lambda_{*,%d}=$ %0.3g',i,i,Lambda_Lcurve(i));
    imagesc(XX);
    axis('image'); 
    axis('off');
    title(mytitle, 'FontSize', FS,'Interpreter','latex')
    
end

% tidying up the outputs
if stopped
    j_Lcurve = i-1;
else
    j_Lcurve = MaxIts;
end
% 
if strcmp(AllX, 'on'), RelErrorNorms_Lcurve = RelErrorNorms_Lcurve(:,1:j_Lcurve); end
RelErrorFinal_Lcurve = RelErrorFinal_Lcurve(1:j_Lcurve);
bigRho_Lcurve = bigRho_Lcurve(:,1:j_Lcurve);
bigEta_Lcurve = bigEta_Lcurve(:,1:j_Lcurve);
Lambda_Lcurve = Lambda_Lcurve(1:j_Lcurve); 
X_Lcurve = X_Lcurve(:,1:j_Lcurve);
WeightsNew_Lcurve = WeightsNew_Lcurve(:, 1:j_Lcurve);
Weights_Lcurve = Weights_Lcurve(:, 1:j_Lcurve);
 
%% new weights; selecting the regularization parameter via the discrepancy principle
figure(2);
Wk = speye(2*sz);
stopped = 0;
% Allocating storage for the outputs
RelErrorNorms_DP = zeros(outerits, MaxIts);
bigRho_DP = zeros(outerits, MaxIts);
Lambda_DP = zeros(MaxIts, 1);
X_DP = zeros(N, MaxIts);
Lgx_DP = zeros(MaxIts, 1);
RelErrorFinal_DP = zeros(MaxIts, 1);
WeightsNew_DP = zeros(2*sz, MaxIts);
Weights_DP = zeros(2*sz, MaxIts);
% Setting the solver options
optn.MaxIter = outerits;
optn.InProjIt = innerits;
optn.RegParam = 'discrepit';
optn.x_true = x_true;
optn.AllX = 'off';
optn.NoiseLevel = NoiseLevel;
optn.eta = 1.05;
for i = 1:MaxIts
    B=Wk*L; % current regularization operator, fed to routine below
    optn.RegMatrix = B;
    [X, info] = IRhybrid_jbd(A, b, optn);
    RelErrorNorms_DP(:,i) = info.Enrm;
    sp = sprintf('Iteration: %d  RelError: %0.5g  Lambda selected: %0.5g',i,info.Enrm,info.RegP(end));
    disp(sp)
    %%% definition (and updates) of the weights
    tmp = (Wk*L)*X;
    tmp = abs(tmp)/norm(tmp,'inf');
    
    if i > 1
        Lgx0=Lgx; % value at the previous iteration
    end
    Lgx = norm(D*X);
    Lgx_DP(i) = Lgx;
   
    w = (1 - tmp.^q); 
    newW = spdiags(w,0,2*sz,2*sz);  % this is the diagonal matrix that will update the current reg operator.
    Wk = newW*Wk; % Compute the current weighting matrix for the next round.  Note that these are diagonal, so the product commutes.
    
    Lambda_DP(i) = info.RegP(end); % store the parameter that gave the 'optimal' solution this round.
    RelErrorFinal_DP(i) = info.Enrm(end); 
    bigRho_DP(:,i)=info.Rnrm; 
    X_DP(:,i) = X; % store the corresponding relative error and solution.
    
    WeightsNew_DP(:,i) = w; 
    Weights_DP(:,i) = diag(Wk); %store the weighting matrix that was used in the call above.
    
    if i > 1
        if Lgx < tau*Lgx0
            stopped = 1;
            break
        end
    end
    
    XX=(reshape(X,n,n)); 
    mytitle=sprintf('Discrepancy P., Iteration %d; $\\lambda_{*,%d}=$ %0.3g',i,i,Lambda_DP(i));
    imagesc(XX);
    axis('image'); 
    axis('off');
    title(mytitle, 'FontSize', FS,'Interpreter','latex')
end

% tidying up the outputs
if stopped
    j_DP = i-1;
else
    j_DP = MaxIts;
end

if strcmp(AllX, 'on'), RelErrorNorms_DP = RelErrorNorms_DP(:,1:j_DP); end
RelErrorFinal_DP = RelErrorFinal_DP(1:j_DP);
bigRho_DP = bigRho_DP(:,1:j_DP);
Lambda_DP = Lambda_DP(1:j_DP); 
X_DP = X_DP(:,1:j_DP);
WeightsNew_DP = WeightsNew_DP(:, 1:j_DP);
Weights_DP = Weights_DP(:, 1:j_DP);


figure(3), clf
mytitle=sprintf('L-curve, Initial reconstruction; $\\lambda_{*,1}=$ %0.3g', Lambda_Lcurve(1));
imagesc(reshape(X_Lcurve(:,1),n,n)); 
axis('image'); 
axis('off'); 
colorbar; 
title(mytitle,'FontSize',FS,'Interpreter','latex'); 

figure(4), clf
mytitle=sprintf('L-curve, Last reconstruction; $\\lambda_{*,%d}=$ %0.3g',j_Lcurve,Lambda_Lcurve(j_Lcurve));
imagesc(reshape(X_Lcurve(:,j_Lcurve),n,n)); 
% colormap(gray)
axis('image'); 
axis('off'); 
colorbar; 
title(mytitle, 'FontSize', FS,'Interpreter','latex')

j = 1;
figure(5), clf
weightstemp = Weights_DP(:,j);
weightstemp = weightstemp(1:sz);
imagesc(reshape(log10(weightstemp),n-1,n)), axis image, axis off
c= colorbar; 
c.FontSize = 16;
mytitle=sprintf('`Vertical'' weights at iteration %d',j+1);
axis('image'); 
axis('off'); 
c= colorbar; 
c.FontSize = 16; 
title(mytitle, 'FontSize', FS,'Interpreter','latex')
figure(6), clf
weightstemp = Weights_DP(:,j);
weightstemp = weightstemp(sz+1:end);
imagesc(reshape(log10(weightstemp),n,n-1)), axis image, axis off
c= colorbar; 
c.FontSize = 16;
mytitle=sprintf('`Horizontal'' weights at iteration %d',j+1);
axis('image'); 
axis('off'); 
c= colorbar; 
c.FontSize = 16; 
title(mytitle, 'FontSize', FS,'Interpreter','latex')

j = 1;
figure(5), clf
weightstemp = Weights_DP(:,j);
weightstemp = weightstemp(1:sz);
imagesc(reshape(log10(weightstemp),n-1,n)), axis image, axis off
c= colorbar; 
c.FontSize = 16;
mytitle=sprintf('DP, `Vertical'' weights at iteration %d',j+1);
axis('image'); 
axis('off'); 
c= colorbar; 
c.FontSize = 16; 
title(mytitle, 'FontSize', FS,'Interpreter','latex')
figure(6), clf
weightstemp = Weights_DP(:,j);
weightstemp = weightstemp(sz+1:end);
imagesc(reshape(log10(weightstemp),n,n-1)), axis image, axis off
c= colorbar; 
c.FontSize = 16;
mytitle=sprintf('DP, `Horizontal'' weights at iteration %d',j+1);
axis('image'); 
axis('off'); 
c= colorbar; 
c.FontSize = 16; 
title(mytitle, 'FontSize', FS,'Interpreter','latex')

j = j_DP;
figure(7), clf
weightstemp = Weights_DP(:,j);
weightstemp = weightstemp(1:sz);
imagesc(reshape(log10(weightstemp),n-1,n)), axis image, axis off
c= colorbar; 
c.FontSize = 16;
mytitle=sprintf('`Vertical'' weights at iteration %d',j+1);
axis('image'); 
axis('off'); 
c= colorbar; 
c.FontSize = 16; 
title(mytitle, 'FontSize', FS,'Interpreter','latex')
figure(8), clf
weightstemp = Weights_DP(:,j);
weightstemp = weightstemp(sz+1:end);
imagesc(reshape(log10(weightstemp),n,n-1)), axis image, axis off
c= colorbar; 
c.FontSize = 16;
mytitle=sprintf('`Horizontal'' weights at iteration %d',j+1);
axis('image'); 
axis('off'); 
c= colorbar; 
c.FontSize = 16; 
title(mytitle, 'FontSize', FS,'Interpreter','latex')

figure(9), clf
axes('FontSize', FS), hold on
semilogy(RelErrorFinal_Lcurve(1:j_Lcurve),'b-o','LineWidth',LW,'MarkerSize',MS)
semilogy(RelErrorFinal_DP(1:j_DP),'m-s','LineWidth',LW,'MarkerSize',MS)
xlabel('Outer iteration, $\ell$', 'Interpreter', 'latex')
ylabel('$\|x - x^{(*,\ell)}\|_2/\|x\|_2$','Interpreter','latex')
legend({'${\mathcal L}$-curve', 'discrepancy principle'}, 'Interpreter', 'latex')
title('Relative errors','FontSize',FS,'Interpreter','latex')
% 
figure(10), clf
axes('FontSize', FS), hold on
semilogy(Lambda_Lcurve(1:j_Lcurve),'b-o','LineWidth',LW,'MarkerSize',MS)
semilogy(Lambda_DP(1:j_DP),'m-s','LineWidth',LW,'MarkerSize',MS)
xlabel('Outer iteration, $\ell$', 'Interpreter', 'latex')
ylabel('$\lambda_{*,\ell}$','Interpreter','latex')
legend({'${\mathcal L}$-curve', 'discrepancy principle'}, 'Interpreter', 'latex')
title('Regularization parameters','FontSize',FS,'Interpreter','latex')
