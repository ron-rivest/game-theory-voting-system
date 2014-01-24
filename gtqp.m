% gtqp.m
% Ronald L. Rivest and Emily Shen
% 3/29/2010

% This code usable under "MIT License":
%
% Copyright (c) 2010 Ronald L. Rivest and Emily Shen
% 
%  Permission is hereby granted, free of charge, to any person
%  obtaining a copy of this software and associated documentation
%  files (the "Software"), to deal in the Software without
%  restriction, including without limitation the rights to use,
%  copy, modify, merge, publish, distribute, sublicense, and/or sell
%  copies of the Software, and to permit persons to whom the
%  Software is furnished to do so, subject to the following
%  conditions:
% 
%  The above copyright notice and this permission notice shall be
%  included in all copies or substantial portions of the Software.
% 
%  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
%  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
%  OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
%  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
%  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
%  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
%  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
%  OTHER DEALINGS IN THE SOFTWARE.

function p = gtqp(M)
% Solve zero-sum two-person symmetric game M of payoffs (to row player)
% returned value p is the optimal mixed strategy
% that minimizes sum of squares of p_i (i.e. it is balanced)

% Basic algorithm is also given in on page 740 of
% TES Raghavan, "Zero-sum two-person games",
% in Handbook of Game Theory, volume 2, (1994), p. 735--759.
% (Or see http://en.wikipedia.org/wiki/Zero-sum )
% We use matlab optimization routine 'lsqlin' to compute solution.

   % get dimension m, where M is m x m
   m = length(M);

   % negate M, since the solution method uses losses, not payoffs
   % also make M all positive by adding a large constant w
   % note that game value becomes w instead of 0
     M = -M ;
     w = max(1, -2 * min(min(M))) ;
     M = M + w ;

   % set up C, d so that minimizing sum of squares of p_i
   % is the same as finding least-squares solution of C p = d
     C = eye(m) ;
     d = zeros(m,1) ;

   % Our desired inequalities are  M p >= 1
   % Find A and b so that  Ap <= b is equivalent.
     A =  -M ;
     b =  -ones(m,1) ;

   % We have one equality constraint: sum x_i = 1/w
     Aeq = ones(1,m) ;
     beq = [ 1 / w ] ;

   % lower bounds and upper bounds
     lb = zeros(m,1) ;
     ub = ones(m,1) ;

   % now solve this constrained least squares problem, 
     '(ignore warning about using medium scale method -- this is OK)'
     p = lsqlin(C,d,A,b,Aeq,beq,lb,ub) ;
     
   % and return the optimal mixed strategy that minimizes sum of squares
     p = p * w ;

