# vs.py  (voting system code)
# Ronald L. Rivest and Emily Shen
# Updated: 3/29/2010
# 
# Given a set of preference ballots, compute GT winner, etc...
# This is the code used for our paper on the GT voting system.  

"""
** Author:  Ronald L. Rivest and Emily Shen
** Address: Room 32G-692 Stata Center 
**          32 Vassar Street 
**          Cambridge, MA 02139
** Email:   rivest@mit.edu, eshen@csail.mit.edu
** Date:    3/27/2010
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

import math
import os
import random
import string
import sys

import game_cvxopt                  # LP and QP solvers for two-person zero-sum games

########################################################################################
### Some global variables...
########################################################################################

indent = "    "                      # standard indent amount within each output section

########################################################################################
### ALTERNATIVES (CANDIDATES)
########################################################################################

"""
An alternative (or candidate) is represented by an integer or a string.
The only restrictions are that a candidate name should not start with
a left parenthesis, include an embedded blank, an embedded equals sign, or be null.

Examples of acceptable alternative names:
    0   1   A  B  Alice  MaryJones  Bob_Smith  #1  #2

We often use the variable A to denote the set of alternatives for an election.
"""

def print_alternatives(A,election_ID):
    """
    Print out list of candidates, nicely, in sorted order.
    """
    print "%s: Alternatives:"%election_ID
    print indent+string.join([str(alt) for alt in sorted(A)])

########################################################################################
### BALLOTS (TUPLES OF CANDIDATES)
########################################################################################

"""
Since this code is about preferential voting, a ballot consists of a nonempty tuple
(Python immutable list) of candidates.  The first element indicates the most-preferred
candidate.  A ballot is a tuple rather than a list so that it can be used as a hash
table (dict) key.  A ballot in some cases may be "short" (i.e. truncated), and only give 
the initial portion of the voter's preferences.

A ballot may contain an equals sign ("=") as one element.  This is not a candidate,
but indicates that the following candidate is to be ranked the same as the previous
candidate.  Some procedures, such as IRV, ignore the "=" signs.  Note that ballots
("A", "B", "C", "=", "D") and ("A", "B") are semantically equivalent when truncated
ballots are allowed (and the candidates are "A", "B", "C", "D"), but we don't
bother to determine when a profile contains two such equivalent ballots.

Examples of ballots:
    (2, 0, 1)
    ("A", "C", "B")
    ("Mary_Smith", "Bob_Jones", "Sarah_Wilson")
    ("Harry",)
    ("A", "=", "B", "C", "=", "D")

"""

def perms(A,rlo=None,rhi=None):
    """
    This routine is for generating a list of "all possible" ballots 
    for a given set  A of alternatives.  (It does not produce ballots
    containing equals signs.)

    There are typically two cases, depending on whether "short" ballots
    are allowed.

    To get all possible permutations of A: perm(A)

    To get all permutations of lengths between rlo and rhi: perms(A,rlo,rhi)
    (Here perms(A,1,len(A)) gives all nonempty ballots on A, including short
    ones.)

    If A is an integer k, it is interpreted as A = range(k) = [0,1,...,k-1].

    Return all r-permutations of set A, for all r such that
        rlo <= r <= rhi
    If rlo is not given, then rlo = len(A)
       so perms(A) gives all full permutations of A.
    If rhi is not given, then rhi = rlo.

    Examples:

    A = [0,1,2,3]
    perms(A,1,2) = 
      [(0,), (0, 1), (0, 2), (0, 3), 
       (1,), (1, 0), (1, 2), (1, 3), 
       (2,), (2, 0), (2, 1), (2, 3), 
       (3,), (3, 0), (3, 1), (3, 2)]

    perms(3) = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    """

    if type(A)==type(0):  # int
        A = range(A)
    if rlo==None:
        rlo = len(A)
    if rlo==0:
        ans = [tuple()]
    else:
        ans = []
    if rhi==None: 
        rhi = rlo
    if rhi==0:
        return ans
    for i in range(len(A)):
        x = A[i]
        Awox = A[:i] + A[i+1:]              # A without x
        L1 = perms(Awox,max(0,rlo-1),rhi-1)
        L2 = [ (x,)+p for p in L1 ]
        ans.extend( L2 )
    return sorted(ans)

def ballot_OK(ballot,A=None):
    """
    Check the given ballot (a tuple):
       -- has all entries distinct  (except OK to have multiple equals signs)
       -- has no syntactically illegal entries:
            entries are non-null, have no whitespace or equal signs, 
            and don't start with "(".  (Exception: bare "=" is OK.)
       -- [if A is not null]: check that all items are in A
    return True if the ballot looks OK
    """
    if len(ballot)==0:                         # zero-length ballot is legal
        return True
    ballot = [ str(a) for a in ballot ]        # convert all to string form for checking
    for (x,y) in zip(sorted(ballot),sorted(ballot)[1:]):
        if x==y and x!="=": return False       # duplicate found (other than duplicate "=")
    if A != None:                              # if A given, all items must be in A
        for item in ballot:
            if item != "=" and item not in A:
                return False
    for (position,item) in enumerate(ballot):
        if len(item) == 0:                     # null string not allowed as item
            return False
        if item[0]=="(":                       # item can't start with a parenthesis
            return False
        for ws in string.whitespace:           # item can't contain whitespace
            if item.find(ws)>=0:
                return False
        if len(item)>1 and item.find('=')>=0:  # item can't have embedded equals signs
            return False
        if item=="=":
            if position==0:                    # ballot can't start with equals sign
                return False
            if position==len(ballot)-1:        # ballot can't end with an equals sign
                return False
            if ballot[position-1]=="=":        # ballot can't have two equals signs in a row
                return False
    return True                                # Ballot is legal

def ballots_for(B,c,elim):
    """
    Return list of all ballots in B voting for c, once candidates 
    in list elim are eliminated.  (The returned ballots still have
    the eliminated candidates; they are just ignored when determining
    if the ballot is for c.)

    B = list of ballots; each ballot is list of distinct candidates

    c = a specific candidate (e.g. 0)

    elim = list of previously eliminated candidates 

    This routine flushes any "=" appearing in a ballot; IRV treats
    ballots as if the "=" signs didn't appear in them at all.

    Example:
       ballots_for(perms(3), 0, [1]) = [(0, 1, 2), (0, 2, 1), (1, 0, 2)]
    """
    ans = []
    for ballot in B:
        filtered_ballot = filter( lambda x: x != "=" and x not in elim, ballot)
        if len(filtered_ballot)>0 and filtered_ballot[0] == c:
            ans.append(ballot)
    return ans


########################################################################################
### PROFILES AND INPUT ROUTINES
########################################################################################
"""
A profile is a set of ballots, with multiplicities.
A profile is represented as a dict P mapping ballots to nonnegative integer multiplicities.
"""

def alternatives_in_profile(P):
    """
    Return the list of alternatives appearing in profile P, in sorted order.
    """
    A = set()
    for ballot in P:
        for alternative in ballot:
            if alternative != "=":
                A.add(alternative)
    return sorted(list(A))

def number_of_ballots_in_profile(P):
    """
    Return number of ballots in profile P.
    """
    return sum([P[ballot] for ballot in P])

def first_choice_counts(A,P):
    """
    Return list giving first-choice counts, in decreasing order by count
    """
    count = { }
    for a in A:
        count[a] = 0
    for ballot in P:
        if len(ballot)>0:
            a = ballot[0]
            count[a] += P[ballot]
    L = [ (count[a],a) for a in A ]
    L = sorted(L)
    L.reverse()
    return L

def print_first_choice_counts(A,P,election_ID):
    """
    Print out a line containing the first choice counts for each candidate.
    """
    print "%s: Count of number of times each candidate was given as first choice."%election_ID
    L = first_choice_counts(A,P)
    line = indent
    for (cnt,a) in L:
        line = line + "%s (%d) "%(a,cnt)
    print line

def parse_ballot_line(line):
    """
    line = line of text representing a single ballot, with multiplicities.
    default count is 1, but any nonnegative count can be given in parentheses.
    Recommendation is to give count *last*, but it could be anywhere.
    Return (ballot,count)

    Input examples:
       "Charles (10)"              returns ("Charles",), 10
       "Alice Charles (10)"        returns ("Alice","Charles"), 10
       "A = B C (12)"              returns ('A','=','B'), 12
    """
    line = string.split(line)            # split into words, using blanks to separate
    count = 1                            # default count, if none given
    ballot = [ ]
    for word in line:                    # note that word can't be empty string here
        if word[0] == "(":               # (count) 
            if word[-1]!=")":
                print "Illegal count field %s in line %s"%(word,line)
                sys.exit()
            count = int(word[1:-1])      
        else:                
            ballot += [word]             # candidate
    ballot = tuple(ballot)
    if not ballot_OK(ballot):
        print "Illegal ballot:",line
        sys.exit()
    return (ballot,count)

def import_ballot(P,ballot,count):
    """
    Add new ballot (or just increment count for ballot) to profile P.
    P = dict mapping ballots to counts
    ballot = tuple of candidates (possibly with "=" signs.
    count = integer multiplicity for ballot
    """
    if not ballot_OK(ballot):
        print "Illegal ballot:", ballot
        sys.exit()
    if P.has_key(ballot):
        P[ballot] += count
    else:
        P[ballot] = count

def import_lines(P,lines):
    """
    Update profile P to reflect ballots given in lines.
    Here lines is a sequence of text lines from a file, 
    one ballot per line with (optional) multiplicities given in parentheses,
    and comments following '#' characters.
    """
    for line in lines:
        i = line.find("#")        # note that '#' is a comment char, as in python
        if i>=0:
            line = line[:i]       # removes first "#" and everything after it
        if line != "":
            (ballot,count) = parse_ballot_line(line)
            import_ballot(P,ballot,count)

def default_params():
    """
    Return dict of default parameters.
    These can be set to different values in an input file by having a line of the
    form:
        ## parametername parametervalue
    e.g.
        ## missing_preferred_less False
    """
    params = { }

    # The parameter  missing_preferred_less
    # if False:
    #    means that ballot  A B  only expressed preference for A over B
    # if True:
    #    means that ballot  A B  also says that A and B preferred to all unlisted candidates.
    params["missing_preferred_less"] = True

    return params

def import_file(filename,P=None,params=None):
    """
    Update profile P corresponding to given file of ballots.
    File has one line per ballot.
    A line may have multipicities, given in parens.
    It is OK to have same ballot more than once; counts just add.
    On any line, # is a comment character; it and later chars are ignored
    A line of the form:
        ## parametername parametervalue
    will set the named parameter to the given value.  Right now,
    the only parameter allowed is "missing_preferred_less", which
    must be True or False.
    """
    if P == None:
        P = { }
    if params == None:
        params = default_params()
    print "Reading ballots from file:",filename
    file = open(filename,"r")
    text = file.read()
    lines = text.split("\n")
    for line in lines:
        if len(line)>0 and line[0]=="#":               # at least a comment
            print line                                 # print it out
            handle_possible_parameter(line,params)     # perhaps setting a parameter
    import_lines(P,lines)
    return P,params

def coerce(x):
    """
    Here input x should be a string (a parameter value).
    Coerce x to nonnegative integer or True/False if possible.
    Else just return x.
    """
    x=x.strip()
    if len(x)==0:
        return x
    if x.lower()=="true":
        return True
    if x.lower()=="false":
        return False
    for c in x:
        if not c.isdigit():
            return x
    return int(x)

def handle_possible_parameter(line,params,printing_wanted=True):
    """
    Input is a line starting with '#'.
    Handle line if it is of the form
        ## parametername parametervalue
    by setting params[parametername] to parametervalue
    If parametervalue is missing, it is set to True
    It is an error if parametername is not already a key in params
    and if new value is not of same type as old value.
    parametervalue coerced to be True/False or an integer if possible
    """
    if len(line)<2:                # too short to be a command
        return
    if line[:2]!="##":             # it's just a comment
        return
    parameter = line[2:].strip()
    if parameter != "":            # something there
        x = parameter.split(" ",1)    # split off parametername from rest (value)
        parametername = x[0]
        if len(x)>1:
            parametervalue = x[1]
        else:
            parametervalue = "True"
        # coerce parameter value if possible to int or True/False
        parametervalue = coerce(parametervalue)
        # check that parameter name is a parameter than can be assigned (i.e. already in params)
        if not params.has_key(parametername):
            print "Error: `%s' is not a parameter than can be set."%parametername
            sys.exit()
        if type(params[parametername])!=type(parametervalue):
            print "Error: value '%s' does not have proper type for parameter `%s'."%(parametervalue,parametername)
            sys.exit()
        params[parametername] = parametervalue
        print indent+"Parameter `%s' set to `%s'."%(parametername, parametervalue)

def print_profile(P,election_ID,print_by_decreasing_count=True):
    """
    print a profile of ballots, nicely, sorted in order 
    print in order by decreasing ballot count if print_by_decreasing_count is True, 
    otherwise print in alphabetic order
    """
    if print_by_decreasing_count:
        print "%s: Profile of ballots (with multiplicities), in decreasing order by count:"%election_ID
        ballots = [(P[ballot],ballot) for ballot in P.keys()]
        ballots = sorted(ballots)
        ballots.reverse()
        ballots = [ b for (cnt,b) in ballots ]
    else:
        print "%s: Profile of ballots (with multiplicities), in sorted (alphabetic) order:"%election_ID
        ballots = [(ballot,P[ballot]) for ballot in P.keys()]
        ballots = sorted(ballots)
        ballots = [ b for (b,cnt) in ballots ]
    for ballot in ballots:
        line = string.join([ str(alternative) for alternative in ballot ])
        count = P[ballot]
        line += "  (%d)  "%count
        print indent+line
    print indent+"Total count =",number_of_ballots_in_profile(P)

def filter_profile(P,A):
    """
    Return a profile modified by eliminating all candidates not in A.
    Not used now; intended possibly for use later if we say filter profiles
    according to Smith set...
    """
    Pnew = { }                                 
    for ballot in P:
        filtered_ballot = filter( lambda x: x in A, ballot)
        import_ballot(Pnew,filtered_ballot,P[ballot])
    return Pnew


########################################################################################
### RANDOM PROFILE GENERATOR
########################################################################################

def random_hypersphere_point(d):
    """
    Return a random point on a d-dimensional hypersphere.
    Ref: http://mathworld.wolfram.com/HyperspherePointPicking.html
    """
    x = [ random.gauss(0.0,1.0) for i in range(d) ]
    l = math.sqrt(sum([xi**2 for xi in x]))
    x = [ xi / l for xi in x]
    return x

def random_profile(A,ballot_count,dist_type,length_range,seed,printing_wanted=False):
    """
    Return a random profile P, a dict mapping 
    mapping ballots to nonnegative integer.
    P[ballot] is the number of times that ballot occurs.
    ballots with count 0 need not be listed explicitly in P,
    so that if m len(A) is large, the size of P is still 
    just at most n, the number of voters, rather than m!

    dist_type specifies the type of distribution wanted,
    as a list [dist_ID, param1, param2,...] where dist_ID is
      ("uniform",)             -- for uniform distribution over B
      ("geometric",d)          -- for spatial d-dimensional model
      ("hypersphere",d)        -- for points on d-dimensional hypersphere

    length_range is a pair (min_ballot_length,max_ballot_length) [or None]
    describing the allowed ballot lengths (inclusive).

    Ballots are generated at first as full-length ballots.
    Then, if length_range is not None, the ballot is truncated
    to a length randomly chosen in length_range. 

    ballot_count gives the total desired number of ballots.
    Running time is linear in ballot_count; for large ballot
    counts this could perhaps be improved.

    Seed is given so that experiment is reproducible.
    """

    random.seed(seed)

    full_ballots = [ ]

    dist_ID = dist_type[0]
    if dist_ID not in ["uniform","geometric","hypersphere"]:
        print "Illegal distribution descriptor for random profile generator:",dist_ID
        sys.exit()

    if dist_ID == "uniform":
        # ("uniform")
        for i in range(ballot_count):       # toss ballots in randomly
            ballot = A[:]
            random.shuffle(ballot)
            full_ballots.append(ballot)
    elif dist_ID == "geometric":     
        # ("geometric", d)
        d = dist_type[1]                    # dimension of issue space
        # generate candidate vectors (position of each candidate on d issues)
        c = {}
        for a in A:
            c[a] = [ random.random() for j in range(d)]
        if printing_wanted:
            print "Candidates:",
            for a in sorted(A): print "%s:"%a,c[a],
            print
        for i in range(ballot_count):
            # generate voter vector v and issue importance vector s
            v = [ random.random() for j in range(d)]
            s = [ random.random() for j in range(d)]
            if printing_wanted:
                print "Voter %d:"%i,v,s,
            # generate ballot for that voter:
            p = 2                              # for L_p norm
            L = [ ( sum( [ s[j]*(abs(v[j]-c[a][j]))**p for j in range(d) ]), a) for a in A ]
            L = sorted(L)
            ballot = tuple([ a for (x,a) in L ])
            full_ballots.append(ballot)
            if printing_wanted:
                print L, ballot
    elif dist_ID == "hypersphere":
        # ("hypersphere",d)
        d = dist_type[1]
        # generate candidate vectors
        c = {}
        for a in A:
            c[a] = random_hypersphere_point(d)
        if printing_wanted:
            print "Candidates:"
            for a in sorted(A): print "%s:"%a,c[a]
        for i in range(ballot_count):
            v = random_hypersphere_point(d)
            if printing_wanted:
                print "Voter %d:"%i,v
            # generate ballot for that voter:
            # note that the distance used in L_p in d-space, and not around
            # the surface of the sphere, but the candidate orderings are unchanged by this.
            p = 2                              # for L_p norm
            L = [ ( sum( [ abs(v[j]-c[a][j])**p for j in range(d) ]), a) for a in A ]
            L = sorted(L)
            ballot = tuple([ a for (x,a) in L ])
            full_ballots.append(ballot)
            if printing_wanted:
                print L, ballot
    # truncate ballots if desired
    P = { }
    if length_range == None:
        # only full ballots allowed
        min_ballot_length = max_ballot_length = len(A)
    else:
        min_ballot_length,max_ballot_length = length_range
    for ballot in full_ballots:
        ballot = ballot[:random.randint(min_ballot_length,max_ballot_length)]
        ballot = tuple(ballot)
        if ballot in P:
            P[ballot] += 1
        else:
            P[ballot] = 1
    return P

########################################################################################
### PAIRWISE PREFERENCES, MARGINS, AND PAIRWISE COMPARISON GRAPHS
########################################################################################

def pairwise_prefs(A,P,params):
    """
    Return a dict pref that gives for each pair (i,j) of alternatives from A
    the count as to how many voters prefer i to j.
    If params["missing_preferred_less"]==True:
        A short ballot is interpreted as listing the top alternatives only;
        the unlisted alternatives are preferred less than any listed alternative.
    else:
        A short ballot contributes nothing for or against the missing candidates.
    This routine also handles equals signs in ballots.
    """
    pref = { }
    for x in A:
        for y in A:
            pref[(x,y)] = 0
    for ballot in P:
        remaining = A[:]                  # less-preferred candidates remaining
        mentioned = [ ]                   # candidates mentioned so far
        equivalents = [ ]                 # equivalence class, disjoint from mentioned
        last_x = None
        for x in ballot:
            if x != "=":
                remaining.remove(x)
                if last_x == "=":
                    equivalents.append(x)
                else:
                    mentioned.extend(equivalents)
                    equivalents = [ x ]
                for y in mentioned:          # earlier options preferred to x  
                    pref[(y,x)] += P[ballot]
            last_x = x
        if params==None or params["missing_preferred_less"]:  # everything mentioned > everything not
            mentioned.extend(equivalents)
            for x in mentioned:
                for y in remaining:
                    pref[(x,y)] += P[ballot]
    return pref

def pairwise_margins(A,P,params):
    """
    Returns a dict margin that gives for each pair (i,j) of alternatives from A
    the net count of voters that prefer i to j (I.e. the number that prefer i to j
    minus the number that prefer j to i.
    Params is the same as for pairwise_prefs
    """
    pref = pairwise_prefs(A,P,params)
    margin = { }
    for (x,y) in pref:
        margin[(x,y)] = pref[(x,y)] - pref[(y,x)]
    return margin

def print_matrix(A,mat):
    """
    Print matrix mat indexed by pairs of alternatives (from A).
    (Primarily used when mat = pref or mat = margin)
    A = list of labels (candidates/alternatives)  (desired order assumed)
    mat = dict mapping pairs of labels (candidates) to integers.
    """
    width = max( [ len(str(a)) for a in A ] + [ len(str(v)) for v in mat.values() ] )
    print indent + " "*(width+1),
    for a in A:
        print string.rjust(a,width+1),
    print
    for a in A:
        print indent+string.ljust(a,width+1),
        for b in A:
            cnt = 0
            if mat.has_key((a,b)): cnt = mat[(a,b)]
            print string.rjust(str(cnt),width+1),
        print

def print_pairwise_prefs(A,pref,election_ID):
    print "%s: Pairwise preferences (number of voters preferring row over column):"%election_ID
    print_matrix(A,pref)

def print_pairwise_margins(A,margin,election_ID):
    print "%s: Pairwise margins (net number of voters preferring row over column):"%election_ID
    print_matrix(A,margin)

def save_matrix(filanme,A,mat):
    """
    Save matrix mat indexed by pairs of alternatives (from A).
    (This is only used to output margin matrix for possible use within matlab.)
    A = list of candidates/alternatives
    mat = dict mapping pairs of candidates to integers.
    """
    # check that A is sorted, otherwise output is not interpretable.
    assert A == sorted(A)
    file = open(filename,"w")
    # compute width of a cell; big enough for row/column name, or for entry
    width = max( [ len(str(a)) for a in A ] + [ len(str(v)) for v in mat.values() ] )
    for a in A:
        for b in A:
            cnt = 0
            if mat.has_key((a,b)): cnt = mat[(a,b)]
            file.write(string.rjust(str(cnt),width+1)+" ")
        file.write("\n")
    file.close()

########################################################################################
### Tie-breaker values TB
########################################################################################
"""
For voting systems that can produce a tie between two candidates, need some systematic
way to break ties, the same for all voting systems.
Here we establish a value TB[a] for each candidate 'a', 
and always favor the candidate with the *smaller* value.
"""
TB = { }

def setup_TB(A,printing_wanted=True):
    """ 
    Establish numeric tie-breaker values for each candidate a in A.
    Currently these are just the positions of the candidate in the sorted list of names
    so the alphabetically first candidate is always favored.  
    But this routine can be modified (by making use_random_TB_values True)
    to generate random values instead.
    """
    global TB
    TB = { }
    value_list = range(len(A))
    use_random_TB_values = False
    if use_random_TB_values:
        random.shuffle(value_list)
    for a,v in zip(sorted(A),value_list):
        TB[a] = v
    if printing_wanted:
        print "Tie-breaker values (smaller is better):"
        print indent,
        for a in A:
            print str(a)+":"+str(TB[a])+" ",
        print

########################################################################################
### Unanimous
########################################################################################

def unanimous_winner(A,P,params,election_ID,printing_wanted=False):
    """
    Return unanimous winner, if there is one (else return None).
    Unanimous winner has all voters giving winner first place.
    """
    if printing_wanted:
        print "%s: Computing Unanimous winner (if any)."%election_ID
    n = number_of_ballots_in_profile(P)
    L = first_choice_counts(A,P)
    max_count = L[0][0]
    if max_count==n:
        if printing_wanted:
            print indent+"Unanimous winner is",L[0][1]
        return L[0][1]
    print indent+"No Unanimous winner exists."
    return None

########################################################################################
### Majority
########################################################################################

def majority_winner(A,P,params,election_ID,printing_wanted=False):
    """
    Return majority winner, if there is one (else return None).
    Majority winner has majority of voters giving candidate first place.
    """
    if printing_wanted:
        print "%s: Computing Majority winner (if any)."%election_ID
    n = number_of_ballots_in_profile(P)
    L = first_choice_counts(A,P)
    max_count = L[0][0]
    if max_count>n/2:
        if printing_wanted:
            print indent+"Majority winner is",L[0][1]
        return L[0][1]
    print indent+"No Majority winner exists."
    return None

########################################################################################
### Plurality
########################################################################################

def plurality_winners(A,P,params,election_ID,printing_wanted=False):
    """
    Return list of all plurality winners (may be more than one if ties occur).
    """
    if printing_wanted:
        print "%s: Computing Plurality winner(s)."%election_ID
    L = first_choice_counts(A,P)
    max_count = L[0][0]
    winners = sorted([ a for (cnt,a) in L if cnt==max_count ])
    if printing_wanted:
        if len(winners) == 1:
            print indent+"Plurality winner is",winners[0]
        else:
            print indent+"Plurality winners are: ",string.join(map(str,winners))
    return winners

def plurality_winner(A,P,params,election_ID,printing_wanted=False):
    """
    Return one of the plurality winners (the one with the smallest TB value).
    """
    global TB
    winners = plurality_winners(A,P,params,election_ID,printing_wanted=False)
    winner = min( [ (TB[w],w) for w in winners ] )[1]
    if printing_wanted:
        print indent+"plurality potential winners:",
        for w in sorted(winners): print w,
        print ", winner is:",winner
    return winner

########################################################################################
### Condorcet
########################################################################################

def Condorcet_winner(A,P,params,election_ID,printing_wanted=False):
    """
    Return Condorcet winner, if one exists (else return None).
    Condorcet winner strictly beats every other in head-on-head competition.
    """
    if printing_wanted:
        print "%s: Computing Condorcet winner (if any)."%election_ID
    winner = None
    pref = pairwise_prefs(A,P,params)        # pref[(i,j)] gives number preferring i to j
    for a in A:
        if all([ b==a or pref[a,b]>pref[b,a] for b in A ]):  # note a,b = (a,b) in python
            winner = a
            break
    if printing_wanted:
        if winner == None:
            print indent+"No Condorcet winner exists."
        else:
            print indent+"Condorcet winner is",winner
    return winner


########################################################################################
### Borda
########################################################################################

def Borda_winner(A,P,params,election_ID,printing_wanted=False):
    """
    Return Borda winner.
    Score of a candidate is the number of other candidates explicitly
    preferred less on the ballots.  Thus, the score for a candidate
    is just the sum of that candidate's row in the preference matrix.
    Ties broken in favor of candidate with smaller TB value.
    """
    global TB
    if printing_wanted:
        print "%s: Computing Borda winner."%election_ID
    prefs = pairwise_prefs(A,P,params)      # prefs[(i,j)] gives number preferring i to j
    scorelist = [ ]
    for a in A:
        score = sum( [ prefs[a,b] for b in A ] )
        scorelist.append( (score,-TB[a],a) )   # so we favor smaller TB values
    scorelist = sorted( scorelist )
    scorelist.reverse()
    winner = scorelist[0][2]
    if printing_wanted:
        line = indent
        for score,tba,a in scorelist:
            line += "%s:%d "%(a,score)
        print line
        print indent+"Borda winner is",winner
    return winner

########################################################################################
### minimax
########################################################################################

def minimax_winner(A,P,params,election_ID,printing_wanted=False):
    """
    Return minimax winner.
    Minimax winner is one whose worst loss is minimized.
    Breaks ties in favor of candidate with smaller TB value
    """
    global TB                    # tie-breaker values (smaller is better)
    if printing_wanted:
        print "%s: Computing minimax winner."%election_ID
    winner = None
    margin = pairwise_margins(A,P,params)
    for a in A:
        a_score = max( [ margin[b,a] for b in A ] )
        if winner == None or a_score < min_score or \
                (a_score==min_score and TB[a]<TB[winner]):
            min_score = a_score
            winner = a
    if printing_wanted:
        print indent+"minimax winner is",winner
    return winner

########################################################################################
### Smith sets
########################################################################################

def Smith_set(A,P,params,election_ID,printing_wanted=False):
    """
    Compute and return a list of the candidates in the Smith set.
    This is the smallest set of candidates such that every candidate in the
    Smith set beats every candidate not in the Smith set in one-on-one contests.
    In this implementation, "a beats b" if at least half the voters prefer a to b.
    Thus, a beats b and vice versa if they are tied; this gives probably the most
    reasonable notion for a Smith set when there are ties.
    
    The algorithm uses the fact that the Smith set will be the *last*
    strongly connected component discovered by the usual DFS SCC algorithm.

    Here A = set of alternatives (candidates), and
         P = profile (dict mapping ballots to counts).
    """
    if printing_wanted:
        print "%s: Computing Smith set."%election_ID
    pref = pairwise_prefs(A,P,params)        # pref[(i,j)] gives number preferring i to j
    n = number_of_ballots_in_profile(P)
    stack = []
    in_stack = set()
    index = 0                            # DFS node counter 
    I = { }                              # gives indices of vertics
    L = { }                              # gives lowlinks of vertices
    for a in A:
        if not I.has_key(a):             # Start a DFS at each node we haven't seen yet
            (index,scc)=Smith_aux(a,A,index,I,L,stack,in_stack,pref,n)   
    scc = sorted(scc)
    if printing_wanted:
        print indent+"Smith set is: "+string.join(scc)
    return scc

def Smith_aux(a,A,index,I,L,stack,in_stack,pref,n):
    """
    Auxiliary routine for DFS
    """
    # print "Smith_aux",a
    I[a] = index  
    L[a] = index
    index = index + 1
    stack.append(a)                      
    in_stack.add(a)                     # record that it is on stack
    scc = None
    for b in A:
        if b!=a and pref[(a,b)]>=pref[(b,a)]:  # note: ties count as an edge!
            # print "edge:",a,"-->",b
            if not I.has_key(b):            # Was successor b visited?
                index,scc = Smith_aux(b,A,index,I,L,stack,in_stack,pref,n)
                L[a] = min(L[a], L[b])
            elif b in in_stack:          # Was successor b in stack?
                L[a] = min(L[a], I[b])
    if L[a] == I[a]:                    # Is a the root of an SCC?
        scc = [ ]
        while stack[-1]!=a:
            b = stack.pop()
            scc.append(b)
            in_stack.remove(b)
        b = stack.pop()
        scc.append(b)
        in_stack.remove(b)
    return (index,scc)

########################################################################################
### IRV 
########################################################################################

def IRV_count(A,P,elim):
    """
    Return dictionary "count" mapping candidates c to vote counts count[c]
    under given profile, when candidates in list elim considered eliminated

    A = list of alternatives (candidates)
    P = profile mapping B to nonnegative integers
    elim = list of eliminated candidates
    """
    count = { }
    for c in A:
        count[c] = sum([P[b] for b in ballots_for(P.keys(),c,elim)])
    return count

def IRV_winner(A,P,params,election_ID,printing_wanted=False):
    """
    Return IRV winner for a given profile P

    A = list of alternatives (candidates)
    P = profile mapping ballots to non-negative integers
    """
    global TB                       # tie-breaker values (smaller is better)
    if printing_wanted:
        print "%s: Computing IRV winner."%election_ID
    remaining = A[:]                # candidates not yet eliminated
    elim = []                       # candidates eliminated
    while len(remaining)>1:
        count = IRV_count(A,P,elim)
        L = sorted( [ (count[c],-TB[c],c) for c in remaining ] )  
        loser = L[0][2]          # a candidate with smallest count (and larger TB value if tied)
        # note ties broken in favor of eliminating candidate whose name sorts earlier
        remaining.remove(loser)
        elim.append(loser)
        if printing_wanted: 
            print indent+"Round %d votes counts:"%(len(A)-len(remaining)),
            L = sorted([ (count[alt],alt) for alt in count.keys() ])
            L.reverse()
            for (cnt,alt) in L:
                if cnt>0:
                    print "%s=%d"%(alt,cnt),
            print "so",loser,"is eliminated."
    winner = remaining[0]
    if printing_wanted: print indent+"IRV winner is",winner
    return winner

########################################################################################
### Schulze's ``beatpath'' method
########################################################################################

def greaterD( ef, gh):
    """
    Return True if pair ef has greater margin than pair gh.
    A pair is a pair of the form:    (pref[e,f],pref[f,e])
    Schulze says (page 154):
    "(N[e,f],N[f,e]) >_win (N[g,h],N[h,g]) 
    if and only if at least one of the following conditions is satisfied:
    1. N[e,f] > N[f,e] and N[g,h] <= N[h,g].
    2. N[e,f] >= N[f,e] and N[g,h] < N[h,g]. 
    3. N[e,f] > N[f,e] and N[g,h] > N[h,g] and N[e,f] > N[g,h]. 
    4. N[e,f] > N[f,e] and N[g,h] > N[h,g] and N[e,f] = N[g,h] and N[f,e] < N[h,g]."
    """
    nef,nfe = ef
    ngh,nhg = gh
    if nef >  nfe and ngh <= nhg: return True
    if nef >= nfe and ngh <  nhg: return True
    if nef >  nfe and ngh >  nhg and nef > ngh: return True
    if nef >  nfe and ngh >  nhg and nef==ngh and nfe < nhg: return True
    return False

def minD( p, q):
    """
    Return pair with smaller margin.
    """
    if greaterD(p,q):
        return q
    else:
        return p

def maxD( p, q):
    """
    Return pair with larger margin.
    """
    if greaterD(p,q):
        return p
    else:
        return q

def beatpath_potential_winners(A,P,params,election_ID,printing_wanted=False):
    """
    Return list of potential winner(s) according to Schulze's ``beatpath'' method.
    Code adapted from Markus Schulze's August 2009 paper,
    Markus Schulze, "Part 1 of 5: A New Monotonic, Clone-Independent, 
    Reversal Symmetric, and Condorcet-Consistent Single-Winner Election Method    
    pages 27--28.    http://m-schulze.webhop.net/schulze1.pdf
    """
    pref = pairwise_prefs(A,P,params)
    PD = { }
    for i in A:
        for j in A:
            if i != j:
                PD[i,j] = (pref[i,j],pref[j,i])
    for i in A:
        for j in A:
            if i != j:
                for k in A:
                    if i != k and j != k:
                        PD[j,k] = maxD( PD[j,k],  minD( PD[j,i], PD[i,k] ) )
    winners = set(A)
    for i in A:
        for j in A:
            if i != j:
                if greaterD( PD[j,i], PD[i,j] ):
                    winners.discard(i)
    winners = list(winners)
    return winners

def beatpath_winner(A,P,params,election_ID,printing_wanted=False):
    # This is the high-level call to Schulze's ``beatpath'' method
    # It finds all beatpath potential winners, and then returns just one of them.
    # This routine just picks the one with the smallest TB (tie-breaker) value.
    # This is consistent with Schulze's presentation (section 2),
    # but could be improved to follow his (rather complex) recommendations in section 5
    # for tie-breaking.
    global TB
    if printing_wanted:
        print "%s: Computing beatpath winner(s). (Variant based on `winning votes' ordering.)"%election_ID
    winners = beatpath_potential_winners(A,P,params,election_ID,printing_wanted)
    winner = min( [ (TB[w],w) for w in winners ] )[1]      # tie-breaker
    if printing_wanted:
        print indent+"beatpath potential winners:",
        for w in sorted(winners): print w,
        print ", winner is:",winner
    return winner

########################################################################################
### Game theory method GT
########################################################################################
def print_optimal_mixed_strategy(A,x,printing_wanted=False):
    if printing_wanted:
        print indent+"  Optimal mixed strategy =   ",
        for (i,xi) in zip(range(len(A)),x):
            print "%s:%11.6f "%(A[i],xi),
        print
        print indent+"  Sum of squares = ",sum([xi*xi for xi in x])
        print indent+"  Cumulative probabilities = ",
        cp = 0.0
        for (ai,xi) in zip(A,x):
            cp += xi
            print "%s:%11.6f "%(ai,cp),
        print

def gt_optimal_mixed_strategy(A,P,params,election_ID,printing_wanted=False):
    """
    Return optimal balanced mixed strategy for two-person zero-sum game for this election
    uses quadratic programming solver
    """
    margin = pairwise_margins(A,P,params)           # note this is a dict
    if printing_wanted:
        print_matrix(A,margin)
    m = len(A)
    M = [ [0]*m for i in range(m) ]                 # make margin *matrix* (not dict)
    for i in range(m):
        for j in range(m):
            M[i][j] = margin[A[i],A[j]]

    print indent+"Using game_cvxopt.qp_solver (quadratic programming --> balanced soln)"
    qp_x = game_cvxopt.qp_solver(M)                     
    print_optimal_mixed_strategy(A,qp_x,printing_wanted)

    return qp_x

def gt_optimal_mixed_strategy_lp(A,P,params,election_ID,printing_wanted=False):
    """
    Return an optimal mixed strategy for two-person zero-sum game for this election
    """
    margin = pairwise_margins(A,P,params)           # note this is a dict
    if printing_wanted:
        print_matrix(A,margin)
    m = len(A)
    M = [ [0]*m for i in range(m) ]
    for i in range(m):
        for j in range(m):
            M[i][j] = margin[A[i],A[j]]

    print indent+"Using game_cvxopt.lp_solver (linear programming --> soln may be unbalanced)"
    lp_x = game_cvxopt.lp_solver(M)                     
    print_optimal_mixed_strategy(A,lp_x,printing_wanted)

    return lp_x

def non_uniform_picker(x,L):
    """
    Input: L is a nonempty list.
           x is a length of probabilities, as long as L.
           (The elements of x should be nonnegative and sum to 1.)
    Return an element of L, picked with probability as given in x.
    """
    cum_prob = 0.0
    test_value = random.random()
    ans = None
    for prob,cand in zip(x,L):
        cum_prob += prob
        if cum_prob > test_value:
            ans = cand
            break
    if ans == None:
        print "Picker can't pick a value; error!"
        print x, L, test_value, cum_prob
        ans = L[0]
    return ans

def gt_support(A,x):
    """
    Return set of support for probability vector x.
    Here A and x have the same length.  
    A is the list of candidates.
    x is a list of their probabilities.
    Probability is `non-zero' if it is greater than epsilon = 1e-6.
    """
    epsilon = 1e-6
    support = [ ]
    for (xi,a) in zip(x,A):
        if xi >= epsilon:
            support.append(a)
    return support

def gt_winner(A,P,params,election_ID,printing_wanted=False):
    """
    Return winner according to GT voting system.
    """
    if printing_wanted:
        print "%s: Computing GT winner."%election_ID
    x = gt_optimal_mixed_strategy(A,P,params,election_ID,printing_wanted)
    gt_winner = non_uniform_picker(x,A)
    if printing_wanted:
        print indent+"GT winner is",gt_winner, " (randomly chosen according to balanced optimal mixed strategy)."
    return gt_winner

def gtd_winner(A,P,params,election_ID,printing_wanted=False):
    """
    Return winner according to GTD voting system (deterministic version of GT).
    Returns candidate with largest probability in optimal mixed strategy.
    (If there are ties, returns one with smallest TB value.)
    """                                      
    global TB                           # tie-breaker values -- smaller is better
    if printing_wanted:
        print "%s: Computing GTD winner."%election_ID
    x = gt_optimal_mixed_strategy(A,P,params,election_ID,printing_wanted)
    gtd_winner = max( [ (x[i],-TB[a],a) for i,a in enumerate(A)] )[2]
    if printing_wanted:
        print indent+"GTD winner is",gtd_winner, " (a candidate with max probability in optimal mixed strategy)."
    return gtd_winner

def gts_winners(A,P,params,election_ID,printing_wanted=False):
    """
    Return set of support for GT voting system.
    """                                      
    if printing_wanted:
        print "%s: Computing GTS winners."%election_ID
    x = gt_optimal_mixed_strategy(A,P,params,election_ID,printing_wanted)
    gts_winners = gt_support(A,x)
    if printing_wanted:
        print indent+"GTS winners are",gts_winners, " (candidates with positive probability in optimal mixed strategy)."
    return gts_winners


########################################################################################
### TEST ROUTINES
########################################################################################

usage = \
"""
Usage: python vs.py file_1 file_2 ... file_k

       Process election data in files file_1 ... file_k
       as if each were for a separate election.

       Data in each file can have comments or comment lines starting with #
       Comments lines (that have '#' as their first character) are printed out.

       Each data line is then like:
           A B C D (23)
       meaning 23 votes prefer A to B to C to D.  (The number can be 0, and
       can be anywhere in the line.)  The line may contain an equals-sign,
       indicating that a voter is indifferent between the preceding candidate
       and the following candidate, e.g.:
           A = B C D = E (5)
       asserts that five voters are indifferent between A and B, and between
       D and E, but prefer A (or B) to C to D (or E).
       A data line can have a comment, following a comment character '#'.

       Data lines may be truncated; unlisted alternatives are less preferred.
       unless a line of the following form is present at the beginning.
       ## missing_preferred_less False

       A filename of the form  "-runoff"  causes a runoff to be made
       between two voting systems (see code for which ones).

       A filename of the form "-compare" runs experiments comparing 
       various voting systems on simulated profiles.
"""

def test_one(filename):
    """
    Run all routines on the given file.
    """
    print "-"*80
    print "-"*80

    P,params = import_file(filename)
    test_P(P,params,filename)

def test_P(P,params,election_ID):
    """
    Run all routines on the given profile.
    """
    A = alternatives_in_profile(P)
    setup_TB(A)                               # establish tie-breaker values
    pref = pairwise_prefs(A,P,params)         # pref[(i,j)] gives number preferring i to j
    margin = pairwise_margins(A,P,params)

    print_alternatives(A,election_ID)
    print_profile(P,election_ID)
    print_first_choice_counts(A,P,election_ID)
    print_pairwise_prefs(A,pref,election_ID)
    print_pairwise_margins(A,margin,election_ID)

    unanimous_winner(A,P,params,election_ID,printing_wanted=True)
    majority_winner(A,P,params,election_ID,printing_wanted=True)
    plurality_winners(A,P,params,election_ID,printing_wanted=True)
    Condorcet_winner(A,P,params,election_ID,printing_wanted=True)
    Borda_winner(A,P,params,election_ID,printing_wanted=True)
    minimax_winner(A,P,params,election_ID,printing_wanted=True)
    Smith = Smith_set(A,P,params,election_ID,printing_wanted=True)
    IRV_winner(A,P,params,election_ID,printing_wanted=True)
    beatpath_winner(A,P,params,election_ID,printing_wanted=True)
    gt_winner(A,P,params,election_ID,printing_wanted=True)

    # code stub that might be expanded or used someday...
    filter_by_Smith_set_wanted = False
    if filter_by_Smith_set_wanted:
        print "Now filtering by Smith set."
        PSmith = filter_profile(P,Smith)
        prefSmith = pairwise_prefs(Smith,PSmith,params)
        print_profile(PSmith,election_ID)
        print_pairwise_prefs(Smith,prefSmith,election_ID)
        # more to add here...

def runoff(fname,f,gname,g,printing_wanted=True):
    """
    Compare method f to method g.
    Here fname and gname are strings giving the names of the two methods.
         f and g take args A,P,params and return a winner.
    Profile is generated randomly, subject to not having Condorcet winner.
    """ 
    election_ID = "runoff"
    number_condorcet = 0
    m = 5                            # number of candidates
    trials = 2000                    # number of simulated elections
    ballot_count = 100               # number of ballots per simulated election
    ballot_distribution = ("hypersphere",3)     # points on a sphere
    ballot_lengths = None            # full ballots wanted
    params = None                    # no special ballot treatment
    condorcet_OK = True              # if True, proceed even if there is a Condorcet winner

    A = list(string.uppercase[:m])   # candidates are A B C ...
    setup_TB(A)                      # setup tie-breaker values
    if printing_wanted:
        print "Number of candidates =",m
        print "Number of ballots per election trial =",ballot_count
        print "ballot_distribution:",ballot_distribution
        print "ballot min/max lengths:",ballot_lengths
        print "Allow profiles with Condorcet winners:",condorcet_OK
    N_xy = 0
    N_yx = 0
    for trial in range(trials):
        trial_counter = 0
        while True:                   # look for profile with generalized tie
            seed = trial*100000 + trial_counter
            P = random_profile(A,ballot_count,
                               ballot_distribution,
                               ballot_lengths,
                               seed
                               )
            if condorcet_OK or len(Smith_set(A,P,params,election_ID)) > 1:
                break
            trial_counter += 1
        if Condorcet_winner(A,P,params,election_ID,printing_wanted=False) != None:
            number_condorcet += 1
        prefs = pairwise_prefs(A,P,params)
        print_profile(P,election_ID)
        x = g(A,P,params,election_ID,printing_wanted=True)         # typically GT
        y = f(A,P,params,election_ID,printing_wanted=True)         # other method
        N_xy += prefs[(x,y)]
        N_yx += prefs[(y,x)]
        print "Trial %4d: Total number preferring %s over %s = %6d," \
              " Total number preferring %s over %s = %6d"%(trial,gname,fname,N_xy,fname,gname,N_yx)
    if N_xy > 0:
        print "%s / %s = %7.4f"%(fname,gname,float(N_yx)/float(N_xy))
    print "number of trials = ",trials
    print "number having Condorcet winner = ",number_condorcet

def L1_dist(A, B):
    return sum([abs(ai-bi) for ai,bi in zip(A,B)])

def agree(x,y):
    """
    x and y are either strings, or lists of strings.
    Return True if they are equal strings, or one
    string is contained in other list, or lists intersect.
    Used to say when output of two voting systems "agree", 
    even if one or both of them outputs lists of winners.
    """
    if type(x)==type(str()): x = [x]
    if type(y)==type(str()): y = [y]
    xset = set(x)
    for yj in y:
        if yj in xset: return True
    return False

def compare_methods(qs, printing_wanted=True):
    """
    Compare methods in qs to each other (and to GT and GTD).
    qs contains a list of (qname, q) pairs, where qname is a string giving
    the name of the method, and q takes args A,P, params and returns a winner.
    Profiles are generated randomly, and may have a Condorcet winner.
    (Currently we do not filter out those profiles having a Condorcet winner.)
    """
    election_ID = "compare"
    m = 5                            # number of candidates
    trials = 10000                   # number of simulated elections
    ballot_count = 100               # ballots per simulated election
    ballot_distribution = ("hypersphere",3)   # points on a sphere
    ballot_lengths = None            # full ballots wanted
    params = None                    # no special ballot treatments
    condorcet_OK = True              # proceed even if there is a Condorcet winner

    A = list(string.uppercase[:m])   # candidates are 'A' 'B' 'C' ...
    setup_TB(A)                      # establish tie-breaker values
    num_methods = len(qs)

    if printing_wanted:
        print "Number of candidates =",m
        print "Number of ballots per election trial =",ballot_count
        print "ballot_distribution:",ballot_distribution
        print "ballot min/max lengths:",ballot_lengths
        print "Allow profiles with Condorcet winners:",condorcet_OK

    trial_counter = 0
    number_condorcet = 0
    num_optimal_mixed_strategy_unique = 0
    Nagree = { }
    Nprefs = { }
    Nmargins = { }
    for (qiname,qi) in qs:
        for (qjname, qj) in qs:
            Nagree[qiname,qjname] = 0
            Nprefs[qiname,qjname] = 0
            Nmargins[qiname,qjname] = 0
    for trial in range(trials):
        print "Trial %4d:"%trial
        # generate random profile
        while True:
            trial_counter += 1
            seed = trial_counter
            P = random_profile(A,ballot_count,
                               ballot_distribution,
                               ballot_lengths,
                               seed
                               )
            has_condorcet = (Condorcet_winner(A,P,params,election_ID,
                                               printing_wanted=False) != None)
            if condorcet_OK or not has_condorcet:
                break
        if printing_wanted:
            print_profile(P,election_ID)
        if has_condorcet:
            number_condorcet += 1
        prefs = pairwise_prefs(A,P,params)
        margins = pairwise_margins(A,P,params)
        # Generate optimal mixed strategy, GT winner, GTD winner
        lp_p = gt_optimal_mixed_strategy_lp(A,P,params,election_ID,printing_wanted)
        p = gt_optimal_mixed_strategy(A,P,params,election_ID,printing_wanted)
        if L1_dist(lp_p,p) < 0.02:
            num_optimal_mixed_strategy_unique += 1
            if printing_wanted:
                print indent+"LP and QP give same solution to GT"
        else:
            if printing_wanted:
                print indent+"LP and QP give different solutions to GT"
        # iterate through all methods
        w = [ None ] * len(qs)     # for each method, a winner, or a list of winners
        for (i,(qname, q)) in enumerate(qs):
            w[i] = q(A,P,params,election_ID,printing_wanted=True)
        # score each method relative to the other
        for (i,(qiname, qi)) in enumerate(qs):
            for (j,(qjname, qj)) in enumerate(qs):
                # agreement
                if agree(w[i],w[j]):
                    Nagree[qiname,qjname] += 1
                # preferences and margins
                if type(w[i])==type(str()) and type(w[j])==type(str()):
                    Nprefs[qiname,qjname]+=prefs[w[i],w[j]]
                    Nmargins[qiname,qjname]+=margins[w[i],w[j]]

    print "--------------------------------------------------------------------------------------"
    print "\nnumber of trials = ",trials
    print "number of profiles generated = ", trial_counter
    print "number having Condorcet winner = ", number_condorcet
    print "number of times LP and QP gave same solution to GT = ", num_optimal_mixed_strategy_unique
    method_names = [ qname for (qname,q) in qs ]
    print "Nagree:"
    print_matrix(method_names,Nagree)
    print "Nprefs:"
    print_matrix(method_names,Nprefs)
    print "Nmargins:"
    print_matrix(method_names,Nmargins)

        
if __name__ == "__main__":
    if len(sys.argv)==0:
        print usage
        sys.exit()
    for filename in sys.argv[1:]:
        if filename == "-runoff":
            # runoff("IRV",IRV_winner,"GT",gt_winner) 
            # runoff("IRV",IRV_winner,"beatpath",beatpath_winner) 
            # runoff("Borda",Borda_winner,"GT",gt_winner) 
            # runoff("beatpath",beatpath_winner,"GT",gt_winner) 
            # runoff("Borda",Borda_winner,"beatpath",beatpath_winner) 
            runoff("beatpath",beatpath_winner,"GTD",gtd_winner) 
            sys.exit()
        if filename == "-compare":
            compare_methods([ ("plurality",plurality_winner),
                              ("IRV", IRV_winner), 
                              ("Borda", Borda_winner),
                              ("minimax",minimax_winner),
                              ("beatpath", beatpath_winner),
                              ("GTS",gts_winners),              # set of support
                              ("GTD",gtd_winner),               # deterministic
                              ("GT",gt_winner)                  # randomized
                              ])
            sys.exit()
        # If we get here, filename is indeed a file name
        print "-"*80
        print "-"*80
        P,params = import_file(filename)
        election_ID = os.path.basename(os.path.splitext(filename)[0])
        test_P(P,params,election_ID)
        # save margin matrix (not really useful now, but was, when we used matlab)
        A = alternatives_in_profile(P)
        margin = pairwise_margins(A,P,params)
        if filename[-4:]==".txt":
            filename = filename[:-4]+".margins"
        else:
            filename = filename + ".margins"
        save_matrix(filename,A,margin)
    print "Done."
