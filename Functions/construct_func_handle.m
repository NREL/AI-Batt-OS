function eq_handle = construct_func_handle(eq_string, xvars)
% Replaces xvars with calls to an array, and returns a nice function handle
for i = 1:length(xvars)
    xvar = xvars{i};
    array_idx = strcat('x(:,', num2str(i), ')');
    eq_string = strrep(eq_string, xvar, array_idx);
end
eq_string = strcat('@(p,x)', eq_string);
eq_handle = str2func(eq_string);
end
