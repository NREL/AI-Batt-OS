function invariant_data = assemble_invariant_data(data)
% Processes the data table to extract all data values that are invariant
% for each cell data series.
invariant_data = table();
cellNums = data.cellNum;
dataVars = data.Properties.VariableNames;
% Iterate through each variable, and if each data series has only one
% unique value for a given variable, add the values from each data series
% to the table.
for i = 1:length(dataVars)
    var_data = table2array(data(:,i));
    invariant_var_data = [];
    for cellNum = unique(cellNums)'
        mask = cellNums == cellNum;
        cell_data = var_data(mask);
        invariant_var_data = [invariant_var_data; unique(cell_data)];
    end
    if length(invariant_var_data) == length(unique(cellNums))
        invariant_data.(dataVars{i}) = invariant_var_data;
    end
end
end
