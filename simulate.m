function simResult = simulate(x, modelEq, model_training_fitResult)
% Simulate the model over the data
p = model_training_fitResult.p;
y_fit = modelEq(p,x);

% If there's bootstrapped parameter values, evaluate the model with each
% param values from each iteration.
if ~isempty(model_training_fitResult.p_boot)
    p_boot = model_training_fitResult.p_boot;
    y_fit_boot = [];
    R_boot = [];
    for boot_iter = 1:size(model_training_fitResult.p_boot,1)
        y_fit_boot_iter = modelEq(p_boot(boot_iter,:),x);
        y_fit_boot = [y_fit_boot, y_fit_boot_iter];
        % Residuals is the difference from the best fit model, just used to
        % get confidence intervals.
        R_boot_iter = y_fit - y_fit_boot_iter;
        R_boot = [R_boot, R_boot_iter];
    end
else
    p_boot = [];
    y_fit_boot = [];
    R_boot = [];
end

% Assemble fitResult struct:
simResult.x = x;
simResult.y_fit = y_fit;
simResult.p = p;
simResult.y_fit_boot = y_fit_boot;
simResult.R_boot = R_boot;
simResult.p_boot = p_boot;