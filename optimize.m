function fitResult = optimize(x, y, cellNums, modelEq, p0, fitOpt)
% Weight data series evenly:
weights = evenlyWeightDataSeries(length(y), cellNums);
% Optimize the modelEq over the entire data set:
p = nlinfit(x, y, modelEq, p0, 'Weights', weights);
y_fit = modelEq(p,x);
% Calculate unweighted residual errors:
R = y - y_fit;
% Calculate cross-validation error:
if isfield(fitOpt, 'CV')
    R_CV = runCrossValidation(x, y, cellNums, modelEq, p0, fitOpt);
else
    R_CV = [];
end
% Calculate uncertainty by bootstrap resampling:
if isfield(fitOpt, 'bootstrap') && strcmp(fitOpt.bootstrap, 'On')
    [y_fit_boot, R_boot, p_boot] = runBootstrapResampling(x, y, cellNums, modelEq, p, fitOpt);
else
    y_fit_boot = [];
    R_boot = [];
    p_boot = [];
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
fitResult.y_fit_boot = y_fit_boot;
fitResult.R_boot = R_boot;
fitResult.p_boot = p_boot;
end

function R_CV = runCrossValidation(x, y, cellNums, modelEq, p0, fitOpt)
% Calculate the cross-validation error of the model. Cross-validation
% splits up the data by data series, rather than data points.
unique_cellNums = unique(cellNums);
if strcmp(fitOpt.CV, 'LeaveOut')
    cv = cvpartition(length(unique_cellNums), 'LeaveOut');
elseif strcmp(fitOpt.CV, 'KFold')
    if ~isfield(fitOpt, 'CV_Folds')
        error("Must include number of folds for K-fold CV in fitOpt.CV_Folds.")
    end
    cv = cvpartition(length(unique_cellNums), 'KFold', fitOpt.CV_Folds);
else
    error("Must specify type of cross-validation folds, see cvparition object help")
end
R_CV = [];
for cv_fold = 1:cv.NumTestSets
    % CV training data:
    train_cells = unique_cellNums(training(cv, cv_fold));
    train_mask = any(cellNums == train_cells',2);
    x_train = x(train_mask,:);
    y_train = y(train_mask);
    % CV test data:
    test_cells = unique_cellNums(test(cv, cv_fold));
    test_mask = any(cellNums == test_cells',2);
    x_test = x(test_mask,:);
    y_test = y(test_mask);
    % Optimize
    weights = evenlyWeightDataSeries(length(y_train), cellNums(train_mask));
    p_CV = nlinfit(x_train, y_train, modelEq, p0, 'Weights', weights);
    % Predict test data:
    y_pred = modelEq(p_CV, x_test);
    R_CV = [R_CV; y_test - y_pred];
end
end

function [y_fit_boot, R_boot, p_boot] = runBootstrapResampling(x, y, cellNums, modelEq, p0, fitOpt)
% Train the model on bootstrap resampled data sets.
if ~isfield(fitOpt, 'bootstrapIterations')
    error("fitOpt struct must include field 'bootstrapIterations' with a numerical value.")
end
% Create a matrix of random data indicies to randomly sample
% with replacement the x and y data for bootstrapping. Dataseries are
% randomly resampled rather than data points.
unique_cellNums = unique(cellNums);
num_cells = length(unique_cellNums);
rng('default')
randomIndices = ceil(rand(num_cells, fitOpt.bootstrapIterations).*num_cells);
% Instantiate result vars:
y_fit_boot = [];
R_boot = [];
p_boot = [];
for i = 1:fitOpt.bootstrapIterations
    % Get the randomly resampled data and optimize:
    boot_cells = unique_cellNums(randomIndices(:,i));
    x_rand = []; y_rand = []; boot_data_series_num = []; data_series_idx = 1;
    for cellNum = boot_cells'
        mask = cellNums == cellNum;
        x_rand = [x_rand; x(mask,:)];
        y_rand = [y_rand; y(mask)];
        % Re-number the data series for weighting resampled data series
        % evenly (cellNums are not unique to each data series due to
        % bootstrap resampling).
        boot_data_series_num = [boot_data_series_num; ones(length(mask(mask)),1).*data_series_idx];
        data_series_idx = data_series_idx + 1;
    end
    % Optimize on the resampled data:
    weights = evenlyWeightDataSeries(length(y_rand), boot_data_series_num);
    p_boot_iter = nlinfit(x_rand, y_rand, modelEq, p0, 'Weights', weights);
    p_boot = [p_boot; p_boot_iter];
    % Calculate the prediction over all the data:
    y_fit_iter = modelEq(p_boot_iter, x);
    y_fit_boot = [y_fit_boot, y_fit_iter];
    R_boot_iter = y - y_fit_iter;
    R_boot = [R_boot, R_boot_iter];
end
end

function weights = evenlyWeightDataSeries(len_data, cellNums)
% Weight the data so that each cell's data is equally considered. Often,
% cells that degrade quickly have fewer data points than cells that degrade
% slowly, and thus receive less weight if optimized using evenly weighted
% data points rather than evenly weighted data series. Without weights,
% each data point has a weight of 1, so total weight is equal to len_data.
weights = ones(len_data,1);
num_cells = length(unique(cellNums));
for cellNum = unique(cellNums)'
    mask = cellNums == cellNum;
    len_data_series = length(mask(mask)); % mask(mask) outputs only the true array elements
    weights(mask) = weights(mask).*(len_data/(num_cells*len_data_series));
end
end
