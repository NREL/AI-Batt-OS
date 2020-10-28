function evalResult = evaluate(x, y, modelEq, model_training_fitResult)
% Evaluate the model over the data
p = model_training_fitResult.p;
y_fit = modelEq(p,x);
% Calculate unweighted residual errors:
R = y - y_fit;
% If there's bootstrapped parameter values, evaluate the model with each
% param values from each iteration.
if ~isempty(model_training_fitResult.p_boot)
    p_boot = model_training_fitResult.p_boot;
    y_fit_boot = [];
    R_boot = [];
    for boot_iter = 1:size(model_training_fitResult.p_boot,1)
        y_fit_boot_iter = modelEq(p_boot(boot_iter,:),x);
        R_boot_iter = y - y_fit_boot_iter;
        y_fit_boot = [y_fit_boot, y_fit_boot_iter];
        R_boot = [R_boot, R_boot_iter];
    end
else
    p_boot = [];
    y_fit_boot = [];
    R_boot = [];
end

% Fit metrics:
% Mean signed difference:
MSD = sum(R)/length(R);
% Mean absolute error:
MAE = sum(abs(R))/length(y);
% Mean absolute percent error:
percentError = R./y;
percentError = percentError(~isinf(percentError) & ~isnan(percentError)); % x/y can equal NaN or Inf if y=0
MAPE = sum(abs(percentError))/length(y);
% Coefficient of determination:
R2 = 1 - sum(R.^2)./sum((y - mean(y)).^2);
% Adjusted coefficient of determination:
DOF = length(y) - length(p);
R2adj = 1 - (1 - R2)*(length(y) - 1)/DOF;
% Mean squared error:
MSE = sum(R.^2)/DOF;
RMSE = sqrt(MSE);

% Assemble fitResult struct:
evalResult.x = x;
evalResult.y = y;
evalResult.y_fit = y_fit;
evalResult.R = R;
evalResult.p = p;
evalResult.MAE = MAE;
evalResult.MAPE = MAPE;
evalResult.R2 = R2;
evalResult.R2adj = R2adj;
evalResult.MSE = MSE;
evalResult.RMSE = RMSE;
evalResult.MSD = MSD;
evalResult.y_fit_boot = y_fit_boot;
evalResult.R_boot = R_boot;
evalResult.p_boot = p_boot;
end
