function [Xout, Xoutvars] = generate_features_multiplicative(X, Xvars)
% Generate calendar storage features for exponential models
% Input: 
%   X (cell array): cell array of arrays with column vectors with input
%       variables. Each cell is a set of input variables that have the same
%       units, i.e., they should not be multipled together (for example, Ua is
%       calculated using a non-linear transformation of SOC, so they should not
%       be combined).
%   Xvars (cell array): cell array of cell arrays with char vectors. Each
%       char vector is a variable name, corresponding to the column in X
%       with the same indices.
% Output:
%   Xout (array): array of output features, each column is a feature
%   Xoutvars (cell array): cell array of descriptive strings for each feature

% Feature generating operators:
%   Syntax: A operators transform ONLY INPUT features, adding a new feature for
%   each input. B operators transform ALL features, adding a new feature
%   for each. C operators calculate interactions between GROUPS of
%   features, creating a new group. Each operator creates new features, and
%   the original input features are always carried forward after an
%   operation.

% 1st: input features are modified: sqrt, squared, and cubed. Only apply
%   to SOC and Ua features. Just square temperature.
%   Operators A1, A2, A3
for idx_group = 1:length(X)
    if idx_group == 1
        [A2_X,A2_Xvars] = operatorA2(X{idx_group},Xvars{idx_group});
        X{idx_group} = [X{idx_group},A2_X];
        Xvars{idx_group} = [Xvars{idx_group},A2_Xvars];
    else
        [A1_X,A1_Xvars] = operatorA1(X{idx_group},Xvars{idx_group});
        [A2_X,A2_Xvars] = operatorA2(X{idx_group},Xvars{idx_group});
        [A3_X,A3_Xvars] = operatorA3(X{idx_group},Xvars{idx_group});
        X{idx_group} = [X{idx_group},A1_X,A2_X,A3_X];
        Xvars{idx_group} = [Xvars{idx_group},A1_Xvars,A2_Xvars,A3_Xvars];
    end
    % 2nd: add inverse of all features
    %   Operator B1
    [B1_X,B1_Xvars] = operatorB1(X{idx_group},Xvars{idx_group});
    X{idx_group} = [X{idx_group},B1_X];
    Xvars{idx_group} = [Xvars{idx_group},B1_Xvars];
end

% 3rd: all input features of different units are multiplied by one another
[X{end+1},Xvars{end+1}] = operatorC1(X{1},X{2},Xvars{1},Xvars{2});


% unwrap the groups of X and Xvars and combine to get Xout and Xoutvars
X = [X{:}];
Xvars = [Xvars{:}];

% Remove any columns with infinities. This could be for a number of
% reasons. The most obvious is that soc can equal 0, and 1/0 = Inf. Also,
% there are many exponential functions with complicated inputs, which can
% run away sometimes.
Xout = X(:,all(isfinite(X),1));
Xoutvars = Xvars(all(isfinite(X),1));

    function [Xout, Xoutvars] = operatorA1(X, Xvars)
        % Take the sqrt of each input feature
        % Input:
        %   X: array of features to transform - each column is a feature
        %   Xvars: cell array of X feature var names
        % Output formatted similary. Total number of output features is N_X.
        Xout = X;
        Xoutvars = Xvars;
        for i = 1:size(X,2)
            Xout(:,i) = X(:,i).^0.5;
            Xoutvars{i} = strcat('(',Xvars{i},'.^0.5',')');
        end
    end

    function [Xout, Xoutvars] = operatorA2(X, Xvars)
        % Take the square of each input feature
        % Input:
        %   X: array of features to transform - each column is a feature
        %   Xvars: cell array of X feature var names
        % Output formatted similary. Total number of output features is N_X.
        Xout = X;
        Xoutvars = Xvars;
        for i = 1:size(X,2)
            Xout(:,i) = X(:,i).^2;
            Xoutvars{i} = strcat('(',Xvars{i},'.^2',')');
        end
    end

    function [Xout, Xoutvars] = operatorA3(X, Xvars)
        % Take the cube of each input feature
        % Input:
        %   X: array of features to transform - each column is a feature
        %   Xvars: cell array of X feature var names
        % Output formatted similary. Total number of output features is N_X.
        Xout = X;
        Xoutvars = Xvars;
        for i = 1:size(X,2)
            Xout(:,i) = X(:,i).^3;
            Xoutvars{i} = strcat('(',Xvars{i},'.^3',')');
        end
    end

    function [Xout, Xoutvars] = operatorB1(X, Xvars)
        % Get the inverse of each feature.
        % Input:
        %   X: array of features to transform - each column is a feature
        %   Xvars: cell array of X feature var names
        % Output formatted similary. Total number of output features is N_X.
        Xout = X;
        Xoutvars = Xvars;
        for i = 1:size(X,2)
            Xout(:,i) = 1./X(:,i);
            Xoutvars{i} = strcat('(1./',Xvars{i},')');
        end
    end

    function [Xout, Xoutvars] = operatorC1(X1, X2, X1vars, X2vars)
        % Multiply all features by each other (not including by themselves)
        % Input:
        %   X1: array of features with same units - each column is a feature
        %   X2: array of features with different units than X1
        %   X1vars: cell array of X1 feature var names
        %   X2vars: cell array of X2 feature var names
        % Output formatted similarly to input. Total number of output features is
        %   N_X2*N_X2.
        numdatapoints = size(X1,1);
        numfeaturevars = size(X1,2)*size(X2,2);
        Xout = zeros(numdatapoints,numfeaturevars);
        Xoutvars = cell(1,numfeaturevars);
        idxXout = 1;
        for i = 1:size(X1,2)
            for j = 1:size(X2,2)
                Xout(:,idxXout) = X1(:,i).*X2(:,j);
                Xoutvars{idxXout} = strcat(X1vars{i},'.*',X2vars{j});
                idxXout = idxXout + 1;
            end
        end
    end
end