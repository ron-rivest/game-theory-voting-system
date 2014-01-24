% gtlp.m

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
%
% (end of license)

function x = gtlp(M)
% Solve two-person zero sum game M of payoffs
% returned value x is *some* optimal mixed strategy

% See   http://en.wikipedia.org/wiki/Zero-sum 
% Algorithm is also given in
% TES Raghavan, "Zero-sum two-person games",
% in Handbook of Game Theory, volume 2,
% Aumann and Hart, eds., (Elsevier,1994), p. 735--759.
% (Problem A on page 740)

   % get m, where M is m x m
   m = length(M);

   % first negate M, since we are dealing with losses not payoffs
   M = - M;

   % now make M all positive by adding a large constant c
   % c = twice the min entry in M
   minitem = min(min(M));
   if minitem <= 0
     M = M - 2 * minitem;
   end

   % Our inequalities are
   %    M x >= 1
   %      x >= 0
   % However, matlab only works with
   %    A x <= b
   % so we need to create A and b appropriately.
   A = [ -M ; -eye(m) ];
   b = [ -ones(m,1); zeros(m,1) ];

   % we wish to minimize f*x, where f is all ones
   f = ones(1,m);

   % now solve this linear program
   x = linprog(f,A,b); 

   % compute the value of the game
   v = 1 / sum(x);

   % and return the optimal mixed strategy
   x = x * v;

