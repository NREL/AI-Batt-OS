function fitResult = optimize_bilevel(x, y, cellNums, modelEq, p0_gbl, p0_lcl)
% Scope p_lcl globally so we can grab it from inside the global
% optimization
global p_lcl y_fit_gbl
p_lcl = ones(length(unique(cellNums)),1)*p0_lcl;
% Optimize the equation:
[p_gbl, R] = nlinfit(x, y, @global_eq, p0_gbl);
y_fit = y_fit_gbl;

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
DOF = length(y) - (length(p_gbl)+numel(p_lcl));
R2adj = 1 - (1 - R2)*(length(y) - 1)/DOF;
% Mean squared error:
MSE = sum(R.^2)/DOF;
RMSE = sqrt(MSE);

% Assemble fitResult struct:
fitResult.x = x;
fitResult.y = y;
fitResult.y_fit = y_fit;
fitResult.R = R;
fitResult.p_gbl = p_gbl;
fitResult.p_lcl = p_lcl;
fitResult.MAE = MAE;
fitResult.MAPE = MAPE;
fitResult.R2 = R2;
fitResult.R2adj = R2adj;
fitResult.MSE = MSE;
fitResult.RMSE = RMSE;
fitResult.MSD = MSD;
fitResult.y_fit_boot = [];
fitResult.R_boot = [];
fitResult.p_boot = [];

    function y_fit = global_eq(p_gbl, x) 
        y_fit = zeros(size(x,1),1);
        iter = 1;
        for cellNum = unique(cellNums)'
            % Iteratively update p0_lcl to the previous result, hopefully
            % guesses from the previous global optimization loop will be
            % better than the initial guesses, saving some time.
            p0_lcl_1 = p_lcl(iter,:);
            mask = cellNums == cellNum;
            x_lcl = x(mask,:);
            y_lcl = y(mask);
            local_eq = @(p_lcl, x_lcl) modelEq(p_gbl, p_lcl, x_lcl);
            p_lcl(iter,:) = nlinfit(x_lcl, y_lcl, local_eq, p0_lcl_1);
            y_fit(mask) = local_eq(p_lcl(iter,:), x_lcl);
            iter = iter + 1;
        end
        y_fit_gbl = y_fit;
    end
end
