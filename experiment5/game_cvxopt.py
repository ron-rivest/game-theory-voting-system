# game_cvxopt.py
# Ronald L. Rivest and Emily Shen
# March 9, 2010
#
# Solve two-person zero-sum games using CVXOPT LP and QP solvers

"""
** Author:  Ronald L. Rivest and Emily Shen
** Address: Room 32G-692 Stata Center 
**          32 Vassar Street 
**          Cambridge, MA 02139
** Email:   rivest@mit.edu, eshen@csail.mit.edu
** Date:    1/17/10
**
** (The following license is known as "The MIT License")
** 
** Copyright (c) 2010 Ronald L. Rivest and Emily Shen
** 
** Permission is hereby granted, free of charge, to any person obtaining a copy
** of this software and associated documentation files (the "Software"), to deal
** in the Software without restriction, including without limitation the rights
** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
** copies of the Software, and to permit persons to whom the Software is
** furnished to do so, subject to the following conditions:
** 
** The above copyright notice and this permission notice shall be included in
** all copies or substantial portions of the Software.
** 
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
** OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
** THE SOFTWARE.
**
** (end of license)
"""

from cvxopt import matrix, solvers

def identity(n):
    """
    Return identity matrix of size n
    """
    I = matrix(0.0, (n, n))
    I[::n+1] = 1.0
    return I

####################################################################################
### LP solver (finds *some* optimal mixed strategy)
####################################################################################

def lp_solver(payoff):
    """
    Solve zero-sum two-person symmetric game M of payoffs.
    Returned value x is an optimal mixed strategy.
    Uses function lp from cvxopt library
    """
    m = len(payoff)

    # convert payoff matrix to cvxopt matrix object M and negate
    M = matrix(payoff).trans()
    M = -M

    # make M all positive by adding large constant v
    v = max(1.0, -2.0 * min(M))
    M = M + v

    # set up G, h so that M x >= 1 and x >= 0 are equivalent to G x <= h
    G = matrix([-M, -identity(m)])
    h = matrix([-1.0]*m + [0.0]*m)

    # set up objective function
    c = matrix(1.0, (m, 1))

    # solve LP problem
    solvers.options['feastol']=1e-9
    solvers.options['show_progress']=False
    x = solvers.lp(c, G, h)['x'];

    # if any were even slightly negative, round up to zero.
    for i in range(m):
        x[i] = max(0.0,x[i])

    # return an optimal mixed strategy
    # sum of x[i]'s should be 1.0/v.  Normalizing gives probability distribution.
    # This should be equivalent to, but more reliable than, simply multiplying by v.        
    sumx = sum(x)
    x = [ xi / sumx for xi in x]

    return x

####################################################################################
### QP solver (finds *balanced* optimal mixed strategy)
####################################################################################

def qp_solver(payoff):
    """
    Solve zero-sum two-person symmetric game M of payoffs.
    Input matrix M is m x m.
    Return value x that is an optimal mixed strategy that minimizes
    sum of squares of x_i. (I.e, it is ``balanced.'')
    Uses function qp from cvxopt library
    """
    m = len(payoff)

    # convert payoff matrix to cvxopt matrix object M and negate
    M = matrix(payoff).trans()
    M = -M

    # make M all positive by adding large constant v
    v = max(1.0, -2.0 * min(M))
    M = M + v

    # set up P, q so that minimizing sum of squares of p_i is
    # equivalent to minimizing 1/2 x^T P x + q^T x
    P = identity(m)                          # P is m x m
    q = matrix([0.0]*m)                      # q is m x 1

    # set up G, h so that M x >= 1 and x >= 0 are equivalent to G x <= h
    G = matrix([-M, -identity(m)])           # G is 2m x m
    h = matrix([-1.0]*m + [0.0]*m)           # h is 2m x 1

    # set up A, b so that sum_i x_i = 1.0/v is equivalent to A x = b
    A = matrix(1.0, (1, m))                  # A is 1 x m
    b = matrix(1.0/v)                        # b is 1 x 1

    # The following requirement on G and A should also be met, 
    # according to the CVXOPT documentation
    # (1)  rank(A) = p                  (where p = # rows of A)
    # (2)  rank(matrix([P,G,A]) = n     (where n = # columns in G and in A)
    # (this last has P stacked on top of G on top of A)
    # otherwise, the routine terminates with a "singular KKT matrix" error
    # but actually gives fairly good results even when terminating this way.
    # These properties should anyway be met by this code.
    
    # solve constrained least squares problem
    solvers.options['feastol']=1e-6      # slightly relaxed from default (avoids singular KKT messages)
    solvers.options['abstol']= 1e-9      # gives us good accuracy on final result
    solvers.options['show_progress']=False
    x = solvers.qp(P, q, G, h, A, b)['x'];

    # if any were even slightly negative, round up to zero
    for i in range(m):
        x[i] = max(0.0,x[i])

    # return optimal mixed strategy that minimizes sum of squares
    # sum of x[i]'s should be 1.0/v.  Normalizing gives probability distribution.
    # This should be equivalent to, but more reliable than, simply multiplying by v.        
    sumx = sum(x)
    x = [ xi / sumx for xi in x]

    return x

def qp_solver_test():
    """
    One test example that produced a singular KKT error when options were set differently.
    (Example x4_3b)
    """
    M = [ [   0,   0,  20,  -50 ],
          [   0,   0,   0,    0 ],
          [ -20,   0,   0,   30 ],
          [  50,   0, -30,    0  ]]
    print qp_solver(M)

if __name__== "__main__":
    qp_solver_test()
