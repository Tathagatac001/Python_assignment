
#[(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)]
[(j,i) for i in xrange(1,4) for j in xrange(1,4)]

#[[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]]
[[i,i+1,i+2,i+3] for i in xrange(2,6)]

#[[2], [3], [4], [3], [4], [5], [4], [5], [6]]
[[i+j] for i in xrange(2,5) for j in xrange(0,3)]

#['x', 'y', 'z', 'xx', 'yy', 'zz', 'xxx', 'yyy', 'zzz', 'xxxx', 'yyyy', 'zzzz']
[letter*i for i in xrange(1,5) for letter in 'xyz']

#['x', 'xx', 'xxx', 'xxxx', 'y', 'yy', 'yyy', 'yyyy', 'z', 'zz', 'zzz', 'zzzz']
[letter*i for letter in 'xyz' for i in xrange(1,5)]

#['A', 'C', 'A', 'D', 'G', 'I', ’L’, ‘ D’]
[ letter for letter in 'ACADGILD']
