function fitResult = optimize_local(x, y, cellNums, modelEq, p0, fitOpt)
% Optimize the modelEq locally for each cell data series, as denoted by
% the cellNum of each row.
p = []; R = []; y_fit = [];
for cellNum = unique(cellNums)'
    % Assemble local cell data:
    mask = cellNums == cellNum;
    x_lcl = x(mask,:);
    y_lcl = y(mask);
    % Optimize the equation to the local data series:
    [p_lcl, R_lcl] = nlinfit(x_lcl, y_lcl, modelEq, p0);
    p = [p; p_lcl];
    R = [R; R_lcl];
    y_fit_lcl = modelEq(p_lcl,x_lcl);
    y_fit = [y_fit; y_fit_lcl];
    % Calculate cross-validation error:
    if isfield(fitOpt,'CV')
        R_CV = runCrossValidation(x_lcl, y_lcl, modelEq, p0, fitOpt);
    else
        R_CV = [];
    end
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
DOF = length(y) - length(p_lcl)*length(unique(cellNums));
R2adj = 1 - (1 - R2)*(length(y) - 1)/DOF;
% Mean squared error:
MSE = sum(R.^2)/DOF;
RMSE = sqrt(MSE);
% Cross-validation MSE:
if ~isempty(R_CV)
    MSE_CV = sum(R_CV.^2)/DOF;
else
    MSE_CV = [];
end

% Assemble fitResult struct:
fitResult.x = x;
fitResult.y = y;
fitResult.y_fit = y_fit;
fitResult.R = R;
fitResult.p = p;
fitResult.MAE = MAE;
fitResult.MAPE = MAPE;
fitResult.R2 = R2;
fitResult.R2adj = R2adj;
fitResult.MSE = MSE;
fitResult.RMSE = RMSE;
fitResult.MSE_CV = MSE_CV;
fitResult.MSD = MSD;
end

function R_CV = runCrossValidation(x, y, modelEq, p0, fitOpt)
if strcmp(fitOpt.CV, 'LeaveOut')
    cv = cvpartition(length(y), 'LeaveOut');
elseif strcmp(fitOpt.CV, 'KFold')
    if ~isfield(fitOpt, 'CV_Folds')
        error("Must include number of folds for K-fold CV in fitOpt.CV_Folds.")
    end
    cv = cvpartition(length(y), 'KFold', fitOpt.CV_Folds);
else
    error("Must specify type of cross-validation folds, see cvparition object help")
end
R_CV = [];
for cv_fold = 1:cv.NumTestSets
    % CV training data:
    train_mask = training(cv, cv_fold);
    x_train = x(train_mask,:);
    y_train = y(train_mask);
    % CV test data:
    test_mask = test(cv, cv_fold);
    x_test = x(test_mask,:);
    y_test = y(test_mask);
    % Optimize
    beta_CV = nlinfit(x_train, y_train, modelEq, p0);
    % Predict test data:
    y_pred = modelEq(beta_CV, x_test);
    R_CV = [R_CV; y_test - y_pred];
end
end
