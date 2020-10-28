%% Automatic identification of calendar fade models
% Paul Gasper, NREL, Energy Conversion and Storage Systems Center
% Paul.Gasper@nrel.gov
% September 2020

% This code is provided to replicate some of the results from the manuscript
% "Challenging Practices of Algebraic Battery Life Models through
% Statistical Validation and Model Identification via Machine-Learning",
% published in ECSarXiv and submitted to ECS.
% The experimental data, which tracks the capacity fade of 3 Ah Sony-Murata
% LFP/Gr lithium-ion batteries during calendar aging as a function of ambient
% temperature and state of charge, as well as the baseline model, referred to 
% by "t^0.5 (ArrTfl_mod)", are from Schimpe et. al., 2018, JES, 
% http://dx.doi.org/10.1149/2.1181714jes.
% In this work, calendar degradation models are identified using bi-level
% optimization and symoblic regression. Many models were investigated, as
% documented in the manuscript. This script and the various functions
% replicate the identification procedure of two well-performing models with
% power law and sigmoidal structures, and compare them with the baseline
% model published previously by Schimpe et. al.

clear; clc; close all;
addpath('Functions')

% To just load results for plotting comparisons of models (last section),
% uncomment the following line:
% load fitted_models

% There are sometimes ill-conditioned Jacobians during bootstrap resampling.
% These results are ignored by looking at the 90% confidence intervals.
warning('Off')

%% Loading and formatting data
% Load the scraped data:
load data_Schimpe_2018 data
% Scraped data contains the temperature, SOC, time in hours, and
% the relative capacity loss. For personal preference, modeling will be
% conducted using time in days and the relative capacity (instead of lost
% capacity). The data table also has a column denoting the cell number, so
% that each individual cell data series can be separated, and the RPT
% number for each measurement, so that the whole data set can be sliced
% according to the number of meausrements conducted on each cell.

% Add U_a to the data table:
% Calculation of the anode-to-reference potential, from Safari and
% Delacourt,  2011, JES, https://doi.org/10.1149/1.3567007. The x_a
% equation was parameterized by Schimpe et. al. using graphite/lithium
% coin cell meausrements.
% x_a: stoichiometric fraction of lithium intercalated into graphite as a
% function of state-of-charge (SOC).
x_a = @(soc) 8.5e-3 + soc.*(7.8e-1 - 8.5e-3);
% U_a: graphite to lithium open cirucit potential as a function of x_a
U_a = @(x_a) 0.6379 + 0.5416.*exp(-305.5309.*x_a) + 0.044.*tanh(-1.*(x_a-0.1958)./0.1088) - 0.1978.*tanh((x_a-1.0571)./0.0854) - 0.6875.*tanh((x_a+0.0117)./0.0529) - 0.0175.*tanh((x_a-0.5692)./0.0875);
% Add a column to the table:
data.U_a = U_a(x_a(data.soc));

% Create training and validation sets:
cellNums = unique(data.cellNum);
% Consistent rng seed for reproducible test/train sets:
rng(6688) % This seed trains w/o 35 degC data, several SOCs at 45 deg C and 15 deg C.
data_partition = cvpartition(length(cellNums),'HoldOut',0.3);
trainset = cellNums(training(data_partition));
data_train = data(any(data.cellNum==trainset',2),:);
validationset = cellNums(test(data_partition));
data_validation = data(any(data.cellNum==validationset',2),:);

% Create data to simulate 20 years aging:
sim_cellNums = [23; 24; 25];
temps = [10, 25, 40];
soc = 0.5;
Ua = U_a(x_a(soc));
t = [0:30:365*20]';
data_sim = table();
for i = 1:length(sim_cellNums)
    temp = table();
    temp.cellNum = sim_cellNums(i).*ones(length(t),1);
    temp.t = t;
    temp.t_years = t./365;
    temp.TdegC = temps(i).*ones(length(t),1);
    temp.TdegK = (temps(i)+273.15).*ones(length(t),1);
    temp.soc = soc.*ones(length(t),1);
    temp.U_a = Ua.*ones(length(t),1);
    data_sim = [data_sim; temp];
end
clearvars -except data data_train data_validation data_sim

%% Replication of baseline model from Schimpe et. al., 2018
disp("Replicating published model from Schimpe et. al., 2018, JES.")
% Some constants:
TdegK_ref = 298.15; % 25 deg C
Ua_ref = 0.123; % V, Ua @ 50% SOC
Rug = 8.314; % Universal gas constant
F = 96485; % Faraday's constant

% Fade rate sub-model (ArrTfl_mod):
% Input vars: TdegK, U_a
% Input params: kcal_ref, Ea, alpha, k0
k_cal = @(p,x) p(1).*exp(-p(2)./Rug.*((1./x(:,1))-(1/TdegK_ref))).*(exp(p(3).*F/Rug.*((Ua_ref-x(:,2))/TdegK_ref))+p(4));
p_k_cal = [0.0003694,20592,0.384,0.142];

% Capacity loss model:
% Input vars: t_hrs, TdegK, U_a
% Input params: kcal_ref, Ea, alpha, k0 (no new parameters)
schimpe_model = @(p,x) 1 - k_cal(p,x(:,2:3)).*x(:,1).^(0.5);
p_schimpe_model = p_k_cal;

% Evaluate predictions for the entire data set:
x = [data.t_hrs, data.TdegK, data.U_a];
qdis_pred = schimpe_model(p_schimpe_model,x);
% Residuals
R = data.qdis - qdis_pred;
% Mean absolute error:
MAE = sum(abs(R))/length(R);
fprintf("As published Schimpe model MAE on training set: %0.3g%%\n", MAE*100);

% Plot predictions:
cellNums = unique(data.cellNum)';
colors = lines(length(cellNums));
figure; 
ax1 = subplot(3,2,1:4); hold on; box on; grid on; % capacity vs time
ax2 = subplot(3,2,5); hold on; box on; grid on; % residuals vs time
ax3 = subplot(3,2,6); box on; grid on; % residuals histogram
for cellNum = cellNums
    mask = data.cellNum == cellNum;
    p1 = plot(ax1, data.t(mask), data.qdis(mask), 'ok', 'MarkerFaceColor', colors(cellNum,:), 'MarkerSize', 6);
    p2 = plot(ax1, data.t(mask), qdis_pred(mask), '-', 'Color', colors(cellNum,:), 'LineWidth', 1);
    plot(ax2, data.t(mask), R(mask), '-', 'Color', colors(cellNum,:), 'LineWidth', 1);
end
histogram(ax3, R, 'BinWidth', 0.0025, 'Orientation', 'horizontal')
% Format residuals plots:
yline(ax2, 0, '--k', 'LineWidth', 2);
RLim = max(abs(ax2.YLim)); ax2.YLim = [-RLim, RLim];
yline(ax3, 0, '--k', 'LineWidth', 2); ax3.YLim = [-RLim, RLim];
% Plot decorations:
xlabel(ax1, 'Time (Days)'); ylabel(ax1, 'Relative discharge capacity');
legend(ax1, [p1,p2], {'Data','t^{0.5} (ArrTfl_{mod})'}, 'Location', 'southwest')
title(ax1,sprintf("Baseline model, q_{dis} = 1 - Arr*Tafel_{mod}*t^{0.5}, MAE=%0.3g%%", MAE*100))
xlabel(ax2, 'Time (Days)'); ylabel(ax2, 'Residual error');
xlabel(ax3, 'Counts'); ylabel(ax3, 'Residual error'); grid on;
% The baseline model shows a clear preference for over-predicting the
% capacity fade, clearly seen in the residuals plot and histogram.

% Reoptimize the model from Schimpe et. al. on just the training set (different
% training data than used by Schimpe et. al.)
% Assemble data:
x = [data_train.t, data_train.TdegK, data_train.U_a];
y = data_train.qdis;
cellNums = data_train.cellNum;
% Optimize with cross-validation and bootstrap resampling:
fitOpt.CV = 'LeaveOut'; fitOpt.bootstrap = 'On'; fitOpt.bootstrapIterations = 1000;
schimpe_train = optimize(x, y, cellNums, schimpe_model, p_schimpe_model, fitOpt);
disp("Reoptimizing Schimpe model with CV and bootstrapping...")
fprintf("Reoptimized Schimpe model MAE on training set: %0.3g%%\n", schimpe_train.MAE*100);

clearvars -except data data_train data_validation data_sim...
    TdegK_ref Ua_ref Rug F schimpe_model schimpe_train

%% Reoptimization of baseline model
% The baseline model is reoptimized on the training data set for comparison
% to the automatically identified models. This is a useful demonstration of
% the workflow for manually identifying a model, as well. During
% reoptimization, the cross-validation error and confidence intervals model
% predictions can also be made.
disp(" ")
disp("Optimizing simplified sqrt(t) model.")
% Define a local equation for determining the time-invariant capacity fade rates, beta_1.
% Input vars: t
% Input params: beta_1
sqrt_model = @(p,x) 1 - p(1).*x(:,1).^(0.5);
p0_sqrt_model = 0.001;
% Assemble data:
x = data_train.t;
y = data_train.qdis;
cellNums = data_train.cellNum;
% Optimize:
sqrt_local_fit = optimize_local(x, y, cellNums, sqrt_model, p0_sqrt_model, []);
fprintf("Sqrt(t) local model MAE on training set: %0.3g%%\n", sqrt_local_fit.MAE*100);
% Plot the local fit result:
plotOpt.layout = 'single axis';
plotOpt.labels = {'t^{0.5} fit'};
plotOpt.colors = {'k'};
plotOpt.xlabel = 'Time (days)';
plotOpt.ylabel = 'Relative discharge capacity';
plotOpt.title = sprintf("Local fits of q_{dis} = 1 - \\beta_1*t^{0.5}, MAE=%0.3g%%", sqrt_local_fit.MAE*100);
plot_capacity_fits(x(:,1), sqrt_local_fit, data_train, plotOpt)
% Local model has clear systematic deviation. Most data series overpredict
% capacity fade early and underpredict later.

% Model beta_1 with the ArrTfl_mod sub-model:
% The model from Schimpe can be simplified by removing constants. This
% makes the units of parameters slightly less interpretable, but makes the
% structure of the model readily comparable to automatically identified
% models.
% Input vars: TdegK, U_a
% Input params: gamma_0, gamma_1, gamma_2, gamma_3
ArrTfl_mod_model = @(p,x) p(1).*exp(p(2).*(1./x(:,1))).*(exp(p(3).*x(:,2))+p(4));
p0_ArrTfl_mod = [0.001, 0, 0, 0];
% Assemble data:
% Only one beta_1 value per data series, grab the value of invariant data
% variables for each data series to train with:
submodel_data = assemble_invariant_data(data_train);
x = [submodel_data.TdegK, submodel_data.U_a];
y = sqrt_local_fit.p(:,1);
cellNums = submodel_data.cellNum;
% Optimize with cross-validation:
fitOpt.CV = 'LeaveOut';
ArrTfl_mod_train = optimize(x, y, cellNums, ArrTfl_mod_model, p0_ArrTfl_mod, fitOpt);
% Plot sub-model fit result:
figure; hold on; box on; grid on;
plot(submodel_data.TdegC, ArrTfl_mod_train.y, 'ok', 'LineWidth', 1.5)
plot(submodel_data.TdegC, ArrTfl_mod_train.y_fit, 'xr', 'LineWidth', 1.5)
xlabel('Temperature (\circC)'); ylabel('\beta_1 (days^{-0.5})');
legend('Locally fit values', 'Sub-model prediction', 'Location', 'northwest')
title(sprintf("Modified Arrhenius Tafel sub-model, MAPE=%0.3g%%", ArrTfl_mod_train.MAPE*100))

% Use the beta_1 sub-model to construct a global model
% Input vars: t, TdegK, U_a
% Input params: beta_1(gamma_0, gamma_1, gamma_2, gamma_3)
sqrt_ArrTflmod_model = @(p,x) 1 - ArrTfl_mod_model(p, x(:,2:3)).*x(:,1).^(0.5);
p0_sqrt_ArrTflmod_model = ArrTfl_mod_train.p;
% Assemble data:
x = [data_train.t, data_train.TdegK, data_train.U_a];
y = data_train.qdis;
cellNums = data_train.cellNum;
% Optimize with cross-validation and bootstrap resampling:
fitOpt.CV = 'LeaveOut'; fitOpt.bootstrap = 'On'; fitOpt.bootstrapIterations = 1000;
disp("Optimizing sqrt(t) global model with CV and bootstrapping...")
sqrt_ArrTflmod_train = optimize(x, y, cellNums, sqrt_ArrTflmod_model, p0_sqrt_ArrTflmod_model, fitOpt);
fprintf("Sqrt(t) global model MAE on training set: %0.3g%%\n", sqrt_ArrTflmod_train.MAE*100);
% Parameter values only change slightly from the initial guess when refit
% to the whole data set.
% Plot global fit results:
plotOpt.layout = 'individual axes';
plotOpt.labels = {'Fit'};
plotOpt.colors = {'k'};
plotOpt.xlabel = 'Time (days)';
plotOpt.ylabel = 'Relative discharge capacity';
plotOpt.title = sprintf("q_{dis} = 1 - ArrTfl_{mod}*t^{0.5}, MAE=%0.3g%%", sqrt_ArrTflmod_train.MAE*100);
plotOpt.confidenceInterval = [5 95];
plot_capacity_fits(x(:,1), sqrt_ArrTflmod_train, data_train, plotOpt)
% Systematic error from local model propogates to global model. Overall
% MAE is 0.26% for global model compared to 0.15% for the local model,
% due to inaccuracies of the beta_1 sub-model.

% Bootstrapping gives useful information such as the distributions of parameter values.
figure; plotmatrix(sqrt_ArrTflmod_train.p_boot)
% Some parameters have extreme values, due to ill-conditioned Jacobians on
% some bootstrap iterations. 90% conf. intervals should look more normal.
p_boot_90CI = [];
for i = 1:size(sqrt_ArrTflmod_train.p_boot,2)
    p_boot = sqrt_ArrTflmod_train.p_boot(:,i);
    CI = prctile(p_boot, [5 95]);
    mask = p_boot >= CI(1) & p_boot <= CI(2);
    p_boot_90CI = [p_boot_90CI, p_boot(mask)];
end
figure; plotmatrix(p_boot_90CI);
% Parameter values look uncorrelated. This implies parameters are each 
% modeling distinct features of the data, which is good.

% Evaluate the data on the validation set:
x = [data_validation.t, data_validation.TdegK, data_validation.U_a];
y = data_validation.qdis;
sqrt_ArrTflmod_validation = evaluate(x, y, sqrt_ArrTflmod_model, sqrt_ArrTflmod_train);
fprintf("Sqrt(t) global model MAE on validation set: %0.3g%%\n", sqrt_ArrTflmod_validation.MAE*100);
plotOpt.title = sprintf("q_{dis} = 1 - ArrTfl_{mod}*t^{0.5}, Validation MAE=%0.3g%%", sqrt_ArrTflmod_validation.MAE*100);
plot_capacity_fits(x(:,1), sqrt_ArrTflmod_validation, data_validation, plotOpt)

% Simulate 20 years aging:
x = [data_sim.t, data_sim.TdegK, data_sim.U_a];
sqrt_ArrTflmod_sim = simulate(x, sqrt_ArrTflmod_model, sqrt_ArrTflmod_train);
% Plot the simulation result:
x = data_sim.t_years;
plotOpt.xlabel = 'Time (years)';
plotOpt.title = '20 year simulation, q_{dis} = 1 - ArrTfl_{mod}*t^{0.5}';
plot_capacity_sim(x, sqrt_ArrTflmod_sim, data_sim, plotOpt);

clearvars -except data data_train data_validation data_sim...
    TdegK_ref Ua_ref Rug F schimpe_model schimpe_train...
    sqrt_local_fit sqrt_ArrTflmod_model sqrt_ArrTflmod_train sqrt_ArrTflmod_validation sqrt_ArrTflmod_sim

%% Check validity of model simplification
% Plot a comparison of the Schimpe model and simplified model predictions:
plotOpt.layout = 'individual axes';
plotOpt.labels = {'ArrTfl_{mod} Fit','Schimpe fit'};
plotOpt.colors = {'k','g'};
plotOpt.xlabel = 'Time (days)';
plotOpt.ylabel = 'Relative discharge capacity';
plotOpt.title = sprintf("t^{0.5} (ArrTfl_{mod}) MAE=%0.3g%%, Schimpe t^{0.5} model MAE=%0.3g%%", sqrt_ArrTflmod_train.MAE*100, schimpe_train.MAE*100);
plotOpt.confidenceInterval = [5 95];
plot_capacity_fits(data_train.t, [sqrt_ArrTflmod_train,schimpe_train], data_train, plotOpt)
% The models perfectly overlap, as expected, because they are mathematically identical after the simplification.

clearvars -except data data_train data_validation data_sim ...
    sqrt_local_fit sqrt_ArrTflmod_model sqrt_ArrTflmod_train sqrt_ArrTflmod_validation sqrt_ArrTflmod_sim

%% Automatically identify the power law model
% The power law model is a more generalized approach to fitting the
% degradation data than the square-root model, allowing both the y-axis
% intercept and the power exponent of time to be free parameters.
% Here, the intercept and power exponent of time are optimized globally
% (one value for the entire data set), and the degradation rate is
% optimized locally (one value for each data series). Models with all
% possible combinations of local/global parameters were compared in the
% manuscript (excepting a model with only global parameters, which is
% obviously bad). Global parameters are denoted by alpha, local parameters
% by beta.
disp(" ")
disp("Identifying and optimizing power law model.")
% Define an equation for bi-level optimization:
% Input vars: t
% Input params: alpha_0, beta_1, alpha_2
power_model_bilevel = @(p_gbl,p_lcl,x) p_gbl(1) - p_lcl(1).*x(:,1).^p_gbl(2);
p0_gbl = [1,0.5];
p0_lcl = 0.001;
% Assemble the data:
x = data_train.t;
y = data_train.qdis;
cellNums = data_train.cellNum;
% Optimize:
power_model_bilevel_training = optimize_bilevel(x, y, cellNums, power_model_bilevel, p0_gbl, p0_lcl);
fprintf("Power law local model MAE on training set: %0.3g%%\n", power_model_bilevel_training.MAE*100);
% Plot the local fit results:
plotOpt.layout = 'single axis';
plotOpt.labels = {'Fit'};
plotOpt.colors = {'b'};
plotOpt.xlabel = 'Time (days)';
plotOpt.ylabel = 'Relative discharge capacity';
plotOpt.title = sprintf("q_{dis} = \\alpha_0 - \\beta_1*t^{(\\alpha_2)}, Local MAE=%0.3g%%", power_model_bilevel_training.MAE*100);
plotOpt.confidenceInterval = [];
plot_capacity_fits(x(:,1), power_model_bilevel_training, data_train, plotOpt)
% Comparing these residuals to the t^0.5 model, it is clear that there is
% still some systematic error (residuals all cross 0 around the middle of
% the time range, underpredicting on one side and overpredicting on the
% other). However, the residuals are now symmetric about the x-axis, and
% the error is slightly lower overall, so this is an improvement.

% Automatically identify beta_1 sub-model:
disp("Identifying local parameter sub-model using LASSO...")
% Assemble data:
% Only one beta_1 value per data series, grab the value of invariant data
% variables for each data series to train with:
submodel_data = assemble_invariant_data(data_train);
x = [submodel_data.TdegK, submodel_data.soc, submodel_data.U_a];
y = power_model_bilevel_training.p_lcl(:,1);
cellNums = submodel_data.cellNum;
% Generate possible descriptors for linear and multiplicative models:
x0 = {submodel_data.TdegK, [submodel_data.soc, submodel_data.U_a]};
x0_vars = {{'TdegK'}, {'soc', 'U_a'}};
[xLin, xLin_vars] = generate_features_linear(x0, x0_vars);
[xMult, xMult_vars] = generate_features_multiplicative(x0, x0_vars);
% Identify linear sub-model descriptors
disp("Sub-models for beta_1")
CV = 4; % 4 fold cross-validation for LASSO optimization.
plotOpt.CVplot = 'On'; plotOpt.lambdaplot = 'Off'; % The lambdaplot (see lasso help) is often uninterpretable.
linearFitInfo = identify_descriptors_linear(xLin, xLin_vars, y, CV, plotOpt);
linear_model = construct_func_handle(linearFitInfo.eq_1SE, [x0_vars{:}]);
multFitInfo = identify_descriptors_multiplicative(xMult, xMult_vars, y, CV, plotOpt);
mult_model = construct_func_handle(multFitInfo.eq_1SE, [x0_vars{:}]);
% Reoptimize the models:
fitOpt.CV = 'LeaveOut'; fitOpt.bootstrap = 'Off';
linear_model_train = optimize(x, y, cellNums, linear_model, linearFitInfo.p_1SE, fitOpt);
mult_model_train = optimize(x, y, cellNums, mult_model, multFitInfo.p_1SE, fitOpt);
fprintf("Linear model R2adj: %0.3g, Multiplicative model R2adj: %0.3g\n", linear_model_train.R2adj, mult_model_train.R2adj);
% Plot a comparison of the results:
figure; hold on; box on; grid on;
plot(submodel_data.TdegC, y, 'ok', 'MarkerSize', 6, 'LineWidth', 1.5)
plot(submodel_data.TdegC, linear_model_train.y_fit, 'xr', 'MarkerSize', 6, 'LineWidth', 1.5);
plot(submodel_data.TdegC, mult_model_train.y_fit, '+b', 'MarkerSize', 6, 'LineWidth', 1.5);
xlabel('Temperature (\circC)'); ylabel(sprintf("\\beta_1 (days^{-%0.2g})", power_model_bilevel_training.p_gbl(2)));
legend('Locally fit values', 'Linear sub-model', 'Multiplicative sub-model', 'Location', 'northwest')
title(sprintf("Linear sub-model: R^2_{adj}=%0.2g, MAPE=%0.3g%% \nMultiplicative sub-model: R^2_{adj}=%0.2g, MAPE=%0.3g%%",...
    linear_model_train.R2adj, linear_model_train.MAPE*100, mult_model_train.R2adj, mult_model_train.MAPE*100))
% The multiplicative model is clearly much better.

% Build a global model using the multiplicative model:
% Input vars: t, TdegK, soc, U_a
% Input params: alpha_0, alpha_2, beta_1(gamma_0, gamma_1, gamma_2, gamma_3, gamma_4)
power_model = @(p,x) p(1) - mult_model(p(3:7),x(:,[2:4])).*(x(:,1).^p(2));
p0 = [power_model_bilevel_training.p_gbl, multFitInfo.p_1SE];
% Assemble the data:
x = [data_train.t, data_train.TdegK, data_train.soc, data_train.U_a];
y = data_train.qdis;
cellNums = data_train.cellNum;
% Optimize with cross-validation and bootstrap resampling:
fitOpt.CV = 'LeaveOut'; fitOpt.bootstrap = 'On'; fitOpt.bootstrapIterations = 1000;
disp("Optimizing power law global model with CV and bootstrapping...")
power_model_train = optimize(x, y, cellNums, power_model, p0, fitOpt);
fprintf("Power law global model MAE on training set: %0.3g%%\n", power_model_train.MAE*100);
% Plot global fit results:
plotOpt.layout = 'individual axes';
plotOpt.labels = {'Fit'};
plotOpt.colors = {'b'};
plotOpt.xlabel = 'Time (days)';
plotOpt.ylabel = 'Relative discharge capacity';
plotOpt.title = sprintf("q_{dis} = \\alpha_0 - \\beta_1(T,SOC,U_a)*t^{(\\alpha_2)}, Global MAE=%0.3g%%", power_model_train.MAE*100);
plotOpt.confidenceInterval = [5 95];
plot_capacity_fits(x(:,1), power_model_train, data_train, plotOpt)

% Test model on validation data:
x = [data_validation.t, data_validation.TdegK, data_validation.soc, data_validation.U_a];
y = data_validation.qdis;
power_model_validation = evaluate(x, y, power_model, power_model_train);
fprintf("Power law global model MAE on validation set: %0.3g%%\n", power_model_validation.MAE*100);
plotOpt.title = sprintf("q_{dis} = \\alpha_0 - \\beta_1(T,SOC,U_a)*t^{(\\alpha_2)}, Global Validation MAE=%0.3g%%", power_model_validation.MAE*100);
plot_capacity_fits(x(:,1), power_model_validation, data_validation, plotOpt)

% Simulate 20 years aging:
x = [data_sim.t, data_sim.TdegK, data_sim.soc, data_sim.U_a];
power_model_sim = simulate(x, power_model, power_model_train);
% Plot the simulation result:
x = data_sim.t_years;
plotOpt.xlabel = 'Time (years)';
plotOpt.title = '20 year simulation, q_{dis} = \\alpha_0 - \\beta_1(T,SOC,U_a)*t^{(\\alpha_2)}';
plot_capacity_sim(x, power_model_sim, data_sim, plotOpt);

clearvars -except data data_train data_validation data_sim ...
    sqrt_local_fit sqrt_ArrTflmod_model sqrt_ArrTflmod_train sqrt_ArrTflmod_validation sqrt_ArrTflmod_sim ...
    power_model_bilevel_training power_model power_model_train power_model_validation power_model_sim

%% Automatically identify sigmoidal model
% The sigmoidal model has one additional parameter compared to the power
% law model, allowing the opportunity to more closely model cell
% degradation trends across the data set, asusming accurate sub-models can
% be identified. The combination of local and global parameters used here
% was the best performing of all possible combinations, as discussed in the
% manuscript. This is not a general statement, but true for this data set.
% The beta_1 parameter of the sigmoidal model represents the limit of the
% model at infinite times. This is physically realistic for degradation
% during calendar aging, as the chemical and electrochemical reactions
% responsible for capacity fade during calendar aging will consume
% reactants within the cell, slowing over time.
disp(" ")
disp("Identifying and optimizing sigmoidal model.")
% Define an equation for bi-level optimization:
% Input vars: t
% Input params: alpha_0, beta_1, alpha_2, beta_3
sigmoidal_model_bilevel = @(p_gbl,p_lcl,x) p_gbl(1) - 2.*p_lcl(1).*(0.5-(1./(1+exp((p_gbl(2).*x(:,1)).^p_lcl(2)))));
p0_gbl = [1, 1e-2];
p0_lcl = [0.1, 0.5];
% Assemble the data:
x = data_train.t;
y = data_train.qdis;
cellNums = data_train.cellNum;
% Optimize:
sigmoidal_model_bilevel_training = optimize_bilevel(x, y, cellNums, sigmoidal_model_bilevel, p0_gbl, p0_lcl);
fprintf("Sigmoidal local model MAE on training set: %0.3g%%\n", sigmoidal_model_bilevel_training.MAE*100);
% Plot the local fit results:
plotOpt.layout = 'single axis';
plotOpt.labels = {'Fit'};
plotOpt.colors = {'r'};
plotOpt.xlabel = 'Time (days)';
plotOpt.ylabel = 'Relative discharge capacity';
plotOpt.title = sprintf("q_{dis} = \\alpha_0 - 2*\\beta_1*(0.5-(1/(1+exp((\\alpha_2*t)^{(\\beta_3)})))), Local MAE=%0.3g%%", sigmoidal_model_bilevel_training.MAE*100);
plotOpt.confidenceInterval = [];
plot_capacity_fits(x(:,1), sigmoidal_model_bilevel_training, data_train, plotOpt)
% MAE for the sigmoidal local model is much lower than the other models,
% which makes sense, since there are many more parameters. Residual errors
% are dominated by a few noisy data points, all other data points are fit
% almost exactly.

% Automatically identify beta_1 sub-model:
disp("Identifying local parameter sub-models using LASSO...")
disp("Sub-model for beta_1")
% Assemble data:
% Only one beta_1 value per data series, grab the value of invariant data
% variables for each data series to train with:
submodel_data = assemble_invariant_data(data_train);
x = [submodel_data.TdegK, submodel_data.soc, submodel_data.U_a];
y = sigmoidal_model_bilevel_training.p_lcl(:,1);
cellNums = submodel_data.cellNum;
% Generate possible descriptors for linear and multiplicative models:
x0 = {submodel_data.TdegK, [submodel_data.soc, submodel_data.U_a]};
x0_vars = {{'TdegK'}, {'soc', 'U_a'}};
[xLin, xLin_vars] = generate_features_linear(x0, x0_vars);
[xMult, xMult_vars] = generate_features_multiplicative(x0, x0_vars);
% Identify linear sub-model descriptors
CV = 4; % 4 fold cross-validation for LASSO optimization.
plotOpt.CVplot = 'On'; plotOpt.lambdaplot = 'Off'; % The lambdaplot (see lasso help) is often uninterpretable.
b1_linearFitInfo = identify_descriptors_linear(xLin, xLin_vars, y, CV, plotOpt);
b1_linear_model = construct_func_handle(b1_linearFitInfo.eq_1SE, [x0_vars{:}]);
b1_multFitInfo = identify_descriptors_multiplicative(xMult, xMult_vars, y, CV, plotOpt);
b1_mult_model = construct_func_handle(b1_multFitInfo.eq_1SE, [x0_vars{:}]);
% Reoptimize the models:
fitOpt.CV = 'LeaveOut'; fitOpt.bootstrap = 'Off';
b1_linear_model_train = optimize(x, y, cellNums, b1_linear_model, b1_linearFitInfo.p_1SE, fitOpt);
b1_mult_model_train = optimize(x, y, cellNums, b1_mult_model, b1_multFitInfo.p_1SE, fitOpt);
fprintf("Linear model R2adj: %0.3g, Multiplicative model R2adj: %0.3g\n", b1_linear_model_train.R2adj, b1_mult_model_train.R2adj);
% Plot a comparison of the results:
figure; hold on; box on; grid on;
plot(submodel_data.TdegC, y, 'ok', 'MarkerSize', 6, 'LineWidth', 1.5)
plot(submodel_data.TdegC, b1_linear_model_train.y_fit, 'xr', 'MarkerSize', 6, 'LineWidth', 1.5);
plot(submodel_data.TdegC, b1_mult_model_train.y_fit, '+b', 'MarkerSize', 6, 'LineWidth', 1.5);
xlabel('Temperature (\circC)'); ylabel("\beta_1");
legend('Locally fit values', 'Linear sub-model', 'Multiplicative sub-model', 'Location', 'northwest')
title(sprintf("Linear sub-model: R^2_{adj}=%0.2g, MAPE=%0.3g%% \nMultiplicative sub-model: R^2_{adj}=%0.2g, MAPE=%0.3g%%",...
    b1_linear_model_train.R2adj, b1_linear_model_train.MAPE*100, b1_mult_model_train.R2adj, b1_mult_model_train.MAPE*100))
% Multiplicative model is best.

% Automatically identify beta_3 sub-model:
disp("Sub-model for beta_3")
% Assemble data:
% Only one beta_1 value per data series, grab the value of invariant data
% variables for each data series to train with:
x = [submodel_data.TdegK, submodel_data.soc, submodel_data.U_a];
y = sigmoidal_model_bilevel_training.p_lcl(:,2);
cellNums = submodel_data.cellNum;
% Generate possible descriptors for linear and multiplicative models:
x0 = {submodel_data.TdegK, [submodel_data.soc, submodel_data.U_a]};
x0_vars = {{'TdegK'}, {'soc', 'U_a'}};
[xLin, xLin_vars] = generate_features_linear(x0, x0_vars);
[xMult, xMult_vars] = generate_features_multiplicative(x0, x0_vars);
% Identify linear sub-model descriptors
CV = 4; % 4 fold cross-validation for LASSO optimization.
plotOpt.CVplot = 'On'; plotOpt.lambdaplot = 'Off'; % The lambdaplot (see lasso help) is often uninterpretable.
b3_linearFitInfo = identify_descriptors_linear(xLin, xLin_vars, y, CV, plotOpt);
b3_linear_model = construct_func_handle(b3_linearFitInfo.eq_1SE, [x0_vars{:}]);
b3_multFitInfo = identify_descriptors_multiplicative(xMult, xMult_vars, y, CV, plotOpt);
b3_mult_model = construct_func_handle(b3_multFitInfo.eq_1SE, [x0_vars{:}]);
% Reoptimize the models:
fitOpt.CV = 'LeaveOut'; fitOpt.bootstrap = 'Off';
b3_linear_model_train = optimize(x, y, cellNums, b3_linear_model, b3_linearFitInfo.p_1SE, fitOpt);
b3_mult_model_train = optimize(x, y, cellNums, b3_mult_model, b3_multFitInfo.p_1SE, fitOpt);
fprintf("Linear model R2adj: %0.3g, Multiplicative model R2adj: %0.3g\n", b3_linear_model_train.R2adj, b3_mult_model_train.R2adj);
% Plot a comparison of the results:
figure; hold on; box on; grid on;
plot(submodel_data.TdegC, y, 'ok', 'MarkerSize', 6, 'LineWidth', 1.5)
plot(submodel_data.TdegC, b3_linear_model_train.y_fit, 'xr', 'MarkerSize', 6, 'LineWidth', 1.5);
plot(submodel_data.TdegC, b3_mult_model_train.y_fit, '+b', 'MarkerSize', 6, 'LineWidth', 1.5);
xlabel('Temperature (\circC)'); ylabel("\beta_3)");
legend('Locally fit values', 'Linear sub-model', 'Multiplicative sub-model', 'Location', 'northwest')
title(sprintf("Linear sub-model: R^2_{adj}=%0.2g, MAPE=%0.3g%% \nMultiplicative sub-model: R^2_{adj}=%0.2g, MAPE=%0.3g%%",...
    b3_linear_model_train.R2adj, b3_linear_model_train.MAPE*100, b3_mult_model_train.R2adj, b3_mult_model_train.MAPE*100))
% Linear model is better, but personal preference is for the multiplicative
% model, which matches the power exponent of cells with higher fade rates
% better.

% Build a global model using the multiplicative model:
% Input vars: t, TdegK, soc, U_a
% Input params: alpha_0, alpha_2, beta_1(gamma_0, gamma_1, ...), beta_3(...)
sigmoidal_model = @(p,x) p(1) - 2.*b1_mult_model(p(3:8),x(:,2:4)).*(0.5-(1./(1+exp((p(2).*x(:,1)).^b3_mult_model(p(9:end),x(:,2:4))))));
p0 = [sigmoidal_model_bilevel_training.p_gbl, b1_multFitInfo.p_1SE, b3_multFitInfo.p_1SE];
% Assemble the data:
x = [data_train.t, data_train.TdegK, data_train.soc, data_train.U_a];
y = data_train.qdis;
cellNums = data_train.cellNum;
% Optimize with cross-validation and bootstrap resampling:
fitOpt.CV = 'LeaveOut'; fitOpt.bootstrap = 'On'; fitOpt.bootstrapIterations = 1000;
disp("Optimizing sigmoidal global model with CV and bootstrapping...")
disp("Note: bootstrapping of sigmoidal model may take several minutes.")
sigmoidal_model_train = optimize(x, y, cellNums, sigmoidal_model, p0, fitOpt);
fprintf("Sigmoidal global model MAE on training set: %0.3g%%\n", sigmoidal_model_train.MAE*100);
% Plot global fit results:
plotOpt.layout = 'individual axes';
plotOpt.labels = {'Fit'};
plotOpt.colors = {'r'};
plotOpt.xlabel = 'Time (days)';
plotOpt.ylabel = 'Relative discharge capacity';
plotOpt.title = sprintf("q_{dis} = \\alpha_0 - 2*\\beta_1*(0.5-(1/(1+exp((\\alpha_2*t)^{(\\beta_3)})))), Global MAE=%0.3g%%", sigmoidal_model_train.MAE*100);
plotOpt.confidenceInterval = [5 95];
plot_capacity_fits(x(:,1), sigmoidal_model_train, data_train, plotOpt)

% Test model on validation data:
x = [data_validation.t, data_validation.TdegK, data_validation.soc, data_validation.U_a];
y = data_validation.qdis;
sigmoidal_model_validation = evaluate(x, y, sigmoidal_model, sigmoidal_model_train);
fprintf("Sigmoidal global model MAE on validation set: %0.3g%%\n", sigmoidal_model_validation.MAE*100);
plotOpt.title = sprintf("q_{dis} = \\alpha_0 - 2*\\beta_1*(0.5-(1/(1+exp((\\alpha_2*t)^{(\\beta_3)})))), Global Validation MAE=%0.3g%%", sigmoidal_model_validation.MAE*100);
plot_capacity_fits(x(:,1), sigmoidal_model_validation, data_validation, plotOpt)

% Simulate 20 years aging:
x = [data_sim.t, data_sim.TdegK, data_sim.soc, data_sim.U_a];
sigmoidal_model_sim = simulate(x, sigmoidal_model, sigmoidal_model_train);
% Plot the simulation result:
x = data_sim.t_years;
plotOpt.xlabel = 'Time (years)';
plotOpt.title = '20 year simulation, q_{dis} = \alpha_0 - 2*\beta_1*(0.5-(1/(1+exp((\alpha_2*t)^{(\beta_3)}))))';
plot_capacity_sim(x, sigmoidal_model_sim, data_sim, plotOpt);

clearvars -except data data_train data_validation data_sim ...
    sqrt_local_fit sqrt_ArrTflmod_model sqrt_ArrTflmod_train sqrt_ArrTflmod_validation sqrt_ArrTflmod_sim ...
    power_model_bilevel_training power_model power_model_train power_model_validation power_model_sim ...
    sigmoidal_model_bilevel_training sigmoidal_model sigmoidal_model_train sigmoidal_model_validation sigmoidal_model_sim

%% Compare various models
% Compare global model capacity predictions:
plotOpt.labels = {'t^{0.5} (ArrTfl_{mod})', 'Power Law', 'Sigmoidal'};
plotOpt.colors = {'k', 'b', 'r'};
plotOpt.xlabel = 'Time (days)';
plotOpt.ylabel = 'Relative discharge capacity';

% Training data:
plotOpt.title = "Comparison of global models - Training data";
fitResults = [sqrt_ArrTflmod_train, power_model_train, sigmoidal_model_train];
% Individual plots with confidence intervals:
plotOpt.layout = 'individual axes'; plotOpt.confidenceInterval = [5 95];
plot_capacity_fits(data_train.t, fitResults, data_train, plotOpt)
% Single plot w/o confidence intervals (nice for see global residuals):
plotOpt.layout = 'single axis'; plotOpt.confidenceInterval = [];
plot_capacity_fits(data_train.t, fitResults, data_train, plotOpt)

% Validation data:
plotOpt.title = "Comparison of global models - Validation data";
fitResults = [sqrt_ArrTflmod_validation, power_model_validation, sigmoidal_model_validation];
% Individual plots with confidence intervals:
plotOpt.layout = 'individual axes'; plotOpt.confidenceInterval = [5 95];
plot_capacity_fits(data_validation.t, fitResults, data_validation, plotOpt)
% Single plot w/o confidence intervals (nice for see global residuals):
plotOpt.layout = 'single axis'; plotOpt.confidenceInterval = [];
plot_capacity_fits(data_validation.t, fitResults, data_validation, plotOpt)

% 20 year simulation:
plotOpt.title = "Comparison of global models - 20 year simulation";
fitResults = [sqrt_ArrTflmod_sim, power_model_sim, sigmoidal_model_sim];
% Individual plots with confidence intervals:
plotOpt.layout = 'individual axes'; plotOpt.confidenceInterval = [5 95];
plot_capacity_sim(data_sim.t, fitResults, data_sim, plotOpt)
% Single plot w/o confidence intervals (nice for see global residuals):
plotOpt.layout = 'single axis'; plotOpt.confidenceInterval = [];
plot_capacity_sim(data_sim.t, fitResults, data_sim, plotOpt)

% Comparison of fit metrics from training:
fitResults = [sqrt_ArrTflmod_train, power_model_train, sigmoidal_model_train];
labels = categorical(plotOpt.labels);
MSE = [fitResults.MSE];
MSE_CV = [fitResults.MSE_CV];
R2adj = [fitResults.R2adj];
% MSE plot:
figure; hold on; box on; grid on;
plot(labels, MSE, 'ok', 'MarkerSize', 6)
plot(labels, MSE_CV, 'ok', 'MarkerFaceColor', 'k', 'MarkerSize', 6)
ylabel('Mean squared error')
legend('MSE', 'MSE_{CV}', 'Location', 'best')
% R2adj plot:
figure; box on; grid on;
plot(labels, R2adj, 'dr', 'MarkerFaceColor', 'r', 'MarkerSize', 6)
ylabel('Adj. coeff. of determination')

% Clean up:
clearvars -except data data_train data_validation data_sim ...
    sqrt_local_fit sqrt_ArrTflmod_model sqrt_ArrTflmod_train sqrt_ArrTflmod_validation sqrt_ArrTflmod_sim ...
    power_model_bilevel_training power_model power_model_train power_model_validation power_model_sim ...
    sigmoidal_model_bilevel_training sigmoidal_model sigmoidal_model_train sigmoidal_model_validation sigmoidal_model_sim
% Save all workspace variable to file:
save("fitted_models.mat")