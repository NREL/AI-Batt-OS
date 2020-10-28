function FitInfo = identify_descriptors_linear(x, xvars, y, CV, plotOpt)
rng('default')
lambda = [0 logspace(-6, 0, 200)];
[BB, FitInfo] = lasso(x, y, 'Lambda', lambda, 'CV', CV, 'PredictorNames', xvars);
if strcmp(plotOpt.CVplot, 'On')
    lassoPlot(BB, FitInfo, 'PlotType', 'CV'); legend('show')
    set(gca, 'YScale', 'log')
    title("Cross-Validated MSE of Lasso Fit for Linear Model")
end
if strcmp(plotOpt.lambdaplot,'On')
    lassoPlot(BB, FitInfo, 'PlotType', 'Lambda');
end
% Grab results:
BB_1SE = BB(:,FitInfo.Index1SE);
nonzero_params = BB_1SE ~= 0;
p_1SE = [FitInfo.Intercept(FitInfo.Index1SE), BB_1SE(nonzero_params)'];
Xvars_1SE = ['c_0', xvars(nonzero_params)];
% Create equation string and display the chosen descriptors:
disp("Linear equation:")
for i = 1:length(Xvars_1SE)
    if i == 1
        eq_1SE = 'p(1)';
        eq_str = "c_0";
    else
        eq_1SE = strcat(eq_1SE,'+p(',num2str(i),').*',Xvars_1SE{i});
        eq_str = strcat(eq_str, " + c_", num2str(i-1), "*", Xvars_1SE{i});
    end
end
disp(eq_str)
FitInfo.Xvars_1SE = Xvars_1SE;
FitInfo.eq_1SE = eq_1SE;
FitInfo.p_1SE = p_1SE;
end
