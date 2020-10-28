function plot_capacity_sim(x, simResults, data_table, plotOpt)
% Plots the simulations from various models, stored in their corresponding
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
    ax1 = nexttile([3 2]); % capacity vs x
    hold on; grid on; box on;
    xlim([0 Inf]);
end
% Iterate through the data series
cellNums = data_table.cellNum;
for cellNum = unique(cellNums,'stable')'
    if flag_individualPlots
        ax1 = nexttile([3 2]); % capacity vs x
        hold on; grid on; box on;
        xlim([0 Inf]);
    end
    mask = cellNums == cellNum;
    % gobjects var for line objects to make a clean legend:
    lines = gobjects(length(simResults),1);
    % Plot predictions from each model:
    for i = 1:length(simResults)
        % Grab vars:
        simResult = simResults(i);
        color = plotOpt.colors{i};
        label = plotOpt.labels{i};
        x_cell = x(mask);
        y_fit_cell = simResult.y_fit(mask);
        % Prediction:
        lines(i) = plot(ax1, x_cell, y_fit_cell, '-', 'Color', color, 'DisplayName', label, 'LineWidth', 1.5);
        if flag_CI && ~isempty(simResult.y_fit_boot)
            [y_fit_CI, ~] = getConfidenceInterval(simResult.y_fit_boot(mask,:), simResult.R_boot(mask,:), plotOpt.confidenceInterval);
            % Set up shading vars:
            x_combined = [x_cell' fliplr(x_cell')];
            y_combined = [y_fit_CI(:,1)' fliplr(y_fit_CI(:,2)')];
            % Plot the shaded confidence interval:
            patch(ax1, x_combined, y_combined, color, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
        end
    end
    % Plot decorations for individual plots
    if flag_individualPlots
        % Only put legend on first plot
        if cellNum == cellNums(1)
            legend(ax1, lines, 'Location', 'southwest')
        end
        % Decorate and format the plots:
        cell_data = data_table(mask,:);
        title(ax1, sprintf("%0.0f \\circC, %0.0f%% SOC", cell_data.TdegC(1), cell_data.soc(1)*100));
        xlabel(ax1, plotOpt.xlabel);
        ylabel(ax1, plotOpt.ylabel);
    end
end
% Plot decorations, single axis
if ~flag_individualPlots
    legend(ax1, lines, 'Location', 'southwest')
    xlabel(ax1, plotOpt.xlabel);
    ylabel(ax1, plotOpt.ylabel);
end
title(t, plotOpt.title);
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