function plot_capacity_fits(x, fitResults, data_table, plotOpt)
% Plots the predictions from various models, stored in their corresponding
% fitResult struct, versus the given x variable. Various plot settings are
% stored in the plotOpt struct. Each data series is specified by a unique
% value of cellNums. This plotting assumes each model was trained on the
% same training data.
% plotOpt (struct):
%   labels (cell array of char vectors): label for each model for legend
%   colors (cell array): cell array of color specification for each model
%   xlabel (char vector): x axis label
%   ylabel (char vector): y axis label
%   title (char vector): title
%   layout (char vector): specifies layout
%       'single axis': all data series plotted on one axis
%       'individual plots': each data series is plotted on its own axis
%   confidenceInterval (array): sets the upper and lower percentiles for
%       plotting shaded confidence intervals
figure;
t = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');
% Set some flags for plot options
[flag_individualPlots, flag_CI] = setFlags(plotOpt);
% Create single set of axes for all data series
if ~flag_individualPlots
    [ax1, ax2] = createAxes;
end
% Iterate through the data series
cellNums = data_table.cellNum;
for cellNum = unique(cellNums,'stable')'
    if flag_individualPlots
        % create new axes for each data series
        [ax1, ax2] = createAxes;
    end
    mask = cellNums == cellNum;
    % gobjects var for line objects to make a clean legend:
    lines = gobjects(length(fitResults)+1,1);
    % Plot training data:
    lines(1) = plot(ax1, x(mask), fitResults(1).y(mask), 'ok', 'MarkerFaceColor', 'k', 'MarkerSize', 6, 'DisplayName', 'Data');
    % Plot predictions from each model:
    for i = 1:length(fitResults)
        % Grab vars:
        fitResult = fitResults(i);
        color = plotOpt.colors{i};
        label = plotOpt.labels{i};
        x_cell = x(mask);
        y_fit_cell = fitResult.y_fit(mask);
        R_cell = fitResult.R(mask);
        % Prediction:
        lines(i+1) = plot(ax1, x_cell, y_fit_cell, '-', 'Color', color, 'DisplayName', label, 'LineWidth', 1.5);
        if flag_CI && ~isempty(fitResult.y_fit_boot)
            [y_fit_CI, R_CI] = getConfidenceInterval(fitResult.y_fit_boot(mask,:), fitResult.R_boot(mask,:), plotOpt.confidenceInterval);
            % Set up shading vars:
            x_combined = [x_cell' fliplr(x_cell')];
            y_combined = [y_fit_CI(:,1)' fliplr(y_fit_CI(:,2)')];
            % Plot the shaded confidence interval:
            patch(ax1, x_combined, y_combined, color, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
        end
        % Residual error:
        plot(ax2, x(mask), R_cell, '-', 'Color', color, 'LineWidth', 1.5);
        if flag_CI && ~isempty(fitResult.y_fit_boot)
            y_combined = [R_CI(:,1)' fliplr(R_CI(:,2)')];
            patch(ax2, x_combined, y_combined, color, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
        end
    end
    % Plot decorations for individual plots
    if flag_individualPlots
        % Only put legend on first plot
        if cellNum == cellNums(1)
            
            legend(ax1, lines, 'Location', 'southwest')
        end
        % Individual plot title:
        cell_data = data_table(mask,:);
        title(ax1, sprintf("%0.0f \\circC, %0.0f%% SOC", cell_data.TdegC(1), cell_data.soc(1)*100));
        % Decorate and format the plots:
        [ax1, ax2] = decoratePlots(ax1, ax2, plotOpt);
    end
end
% Plot decorations, single axis
if ~flag_individualPlots
    legend(ax1, lines, 'Location', 'southwest')
    decoratePlots(ax1, ax2, plotOpt);
end
title(t, plotOpt.title);
end

function [ax1, ax2] = createAxes
ax1 = nexttile([3 2]); % capacity vs x
hold on; grid on; box on;
xlim([0 Inf]);
ax2 = nexttile([3 1]); % residuals vs x
hold on; grid on; box on;
xlim([0 Inf]);
end

function [flag_individualPlots, flag_CI] = setFlags(plotOpt)
if isfield(plotOpt, 'layout')
    if strcmp(plotOpt.layout, 'individual axes')
        flag_individualPlots = 1;
    else
        flag_individualPlots = 0;
    end
else
    % default single axis, do nothing
    flag_individualPlots = 0;
end
if isfield(plotOpt, 'confidenceInterval')
    if ~isempty(plotOpt.confidenceInterval)
        flag_CI = 1;
    else
        flag_CI = 0;
    end
else
    flag_CI = 0;
end
end

function [y_fit_CI, R_CI] = getConfidenceInterval(y_fit_boot, R_boot, confidenceInterval)
% Sum the residuals to get total error for each bootstrapped prediction:
sumR_boot = sum(R_boot, 1);
% Get the indicies at the specified percentiles:
sumR_CI = prctile(sumR_boot, confidenceInterval);
[~, idx_lb] = min(abs(sumR_boot-sumR_CI(1)));
[~, idx_ub] = min(abs(sumR_boot-sumR_CI(2)));
% Assemble outputs:
y_fit_CI = [y_fit_boot(:,idx_lb), y_fit_boot(:,idx_ub)];
R_CI = [R_boot(:,idx_lb), R_boot(:,idx_ub)];
end

function [ax1, ax2] = decoratePlots(ax1, ax2, plotOpt)
xlabel(ax1, plotOpt.xlabel); xlabel(ax2, plotOpt.xlabel);
ylabel(ax1, plotOpt.ylabel); ylabel(ax2, 'Residual Error');
% Residuals plot formatting
yline(ax2, 0, '--k', 'LineWidth', 1.5);
yLimits = max(abs(ylim(ax2)));
if yLimits > 1
    ylim(ax2, [-1 1]);
else
    ylim(ax2, [-yLimits yLimits])
end
end