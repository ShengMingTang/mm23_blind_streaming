% https://www.1ju.org/matlab/matlab-differential
syms m;
syms k;
syms h;
syms P;

f = ((m * (k-1) + (1/P))^(m*P)) * (( 1/k + ((1-1/k) * (h - m*k)) / (m*(k-1)) ))^(m*P);
df_k = diff(f, k); % calculate derivative
z = solve(df_k, k); % solve for roots
% df_k = P*m^2*(m*(k - 1) + 1/P)^(P*m - 1)*(1/k - ((1/k - 1)*(h - k*m))/(m*(k - 1)))^(P*m) + P*m*(m*(k - 1) + 1/P)^(P*m)*(1/k - ((1/k - 1)*(h - k*m))/(m*(k - 1)))^(P*m - 1)*((1/k - 1)/(k - 1) - 1/k^2 + (h - k*m)/(k^2*m*(k - 1)) + ((1/k - 1)*(h - k*m))/(m*(k - 1)^2))
% z = 
% (m - 1/P)/m
% (h + m)/m
% (P*(h + m)*(P*m - 1))^(1/2)/(P*m)
% -(P*(h + m)*(P*m - 1))^(1/2)/(P*m)

% plot
zero = subs(subs(subs(z, m, 0.1), h, 0.5), P, 100); % replace symbols with values
y = subs(subs(subs(f, m, 0.1), h, 0.5), P, 100);
x = linspace(1.1, 0.5/0.1, 20000);
v = subs(y, k, x);
plot(x, v);
h = gobjects(size(zero)); 
for i = 1:numel(zero)
    h(i) = xline(double(zero(i))); % use double() to evaluate const expression
end


%##
syms m;
syms k;
syms h;
syms P;
f = (1/k + (1-1/k)/(m*k)*(h-m)) ^(m*P);
df_k = diff(f, k); % calculate derivative
z = solve(df_k, k); % solve for roots
% z = 
% (2*h - 2*m)/h
% (h - m)/h
