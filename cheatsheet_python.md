## statistics.mode()
most recuring element

## statistics.multimode()
most recuring element (list if multiple)

## random.randrange(1,100,2)
rand odd number

## math.comb(n, k)
Return the number of ways to choose k items from n items without repetition and without order.
Evaluates to n! / (k! * (n - k)!) when k <= n and evaluates to zero when k > n.
Also called the binomial coefficient because it is equivalent to the coefficient of k-th term in polynomial expansion of (1 + x)ⁿ.

## math.perm(n, k=None)
Return the number of ways to choose k items from n items without repetition and with order.
Evaluates to n! / (n - k)! when k <= n and evaluates to zero when k > n.
If k is not specified or is None, then k defaults to n and the function returns n!.   

## math.fsum
floating point accurate

## math.gcd(*integers)
Return the greatest common divisor of the specified integer arguments.
If any of the arguments is nonzero, then the returned value is the largest positive integer that is a divisor of all arguments.
If all arguments are zero, then the returned value is 0. gcd() without arguments returns 0.

## math.lcm()
Return the least common multiple of the specified integer arguments.
If all arguments are nonzero, then the returned value is the smallest positive integer that is a multiple of all arguments.
If any of the arguments is zero, then the returned value is 0. lcm() without arguments returns 1.

## math.prod()
product of all items in an iterable + start value (1)

## Merge sort
```python
def merge_sort(lista):
    def larger(x, y): return x<y
    if len(lista) > 1:
        a = len(lista)//2
        lewa = lista[:a]
        prawa = lista[a:]
        merge_sort(lewa)
        merge_sort(prawa)

        l, p, i = (0, 0, 0)
        while l < len(lewa) and p < len(prawa):
            if larger(lewa[l], prawa[p]):
                lista[i] = lewa[l]
                l += 1
            else:
                lista[i] = prawa[p]
                p += 1
            i += 1
        
        while l < len(lewa):
            lista[i] = lewa[l]
            l += 1
            i += 1

        while p < len(prawa):
            lista[i] = prawa[p]
            p += 1
            i += 1
```
---------------------------------
```python
def nwd(x,y): // or just math.gcd()
    if x > 0:
        return nwd(y, x%y)
    else:
        return x
# if more than 2 numbers:
# nwd(a,b,c) == nwd(nwd(a,b),c)
```
---------------------------------
```python
def nww(x,y): // or just math.lcm() >= 3.9
    return (x*y)/nwd(x,y)


# Function to find the partition position
```
---------------------------------
## Quick Sort
```python
def quickSort(array, low, high):
    if low < high:
 
        # Find pivot element such that
        # element smaller than pivot are on the left
        # element greater than pivot are on the right
        pi = partition(array, low, high)
 
        # Recursive call on the left of pivot
        quickSort(array, low, pi - 1)
 
        # Recursive call on the right of pivot
        quickSort(array, pi + 1, high)
```
### partition 
```python
def partition(array, low, high):
 
    # choose the rightmost element as pivot
    pivot = array[high]
 
    # pointer for greater element
    i = low - 1
 
    # traverse through all elements
    # compare each element with pivot
    for j in range(low, high):
        if array[j] <= pivot:
 
            # If element smaller than pivot is found
            # swap it with the greater element pointed by i
            i = i + 1
 
            # Swapping element at i with element at j
            (array[i], array[j]) = (array[j], array[i])
 
    # Swap the pivot element with the greater element specified by i
    (array[i + 1], array[high]) = (array[high], array[i + 1])
 
    # Return the position from where partition is done
    return i + 1
 
# function to perform quicksort
```
---------------------------------
```python
def SieveOfEratosthenes(num):
    prime = [True for _ in range(num+1)]
    # boolean array
    p = 2
    while (p * p <= num):
  
        # If prime[p] is not
        # changed, then it is a prime
        if (prime[p] == True):
  
            # Updating all multiples of p
            for i in range(p * p, num+1, p):
                prime[i] = False
        p += 1

    array = []
    for p in range(2, num+1):
        if prime[p]:
            array.append(p)
    return array
    
    # albo:
    # return prime

```
---------------------------------
## Is prime
```python
def isPrime(n):
    if n <= 1: return False
    if n <= 3: return True

    if n % 3 == 0 or n % 2 == 0: return False

    i = 11
    while n >= i * i:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True

# for num in SieveOfEratosthenes(1000):
for num in range(100):
    print(num, isPrime(num))
```
---------------------------------
```python
def primeFactorization(n):
    factors = []
    for i in range(2,n + 1):
        if n % i == 0:
            count = 1
            for j in range(2,(i//2 + 1)):
                if(i % j == 0):
                    count = 0
                    break
            if(count == 1):
                factors.append(i)
    return factors
```
---------------------------------
```python
print(primeFactorization(1024))

def TwoColumnsTextImport():
    for i in r:
        col1.append(int(i.split(" ")[0]))
        col2.append(int(i.split(" ")[1]))
```
---------------------------------
## Depth First Search
```python
def dfs(visited, graph, node):  #function for dfs 
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)
```
---------------------------------
## Path finding
```python
def BFS_SP(graph, start, goal):
    explored = []
     
    # Queue for traversing the
    # graph in the BFS
    queue = [[start]]
     
    # If the desired node is
    # reached
    if start == goal:
        print("Same Node")
        return
     
    # Loop to traverse the graph
    # with the help of the queue
    while queue:
        path = queue.pop(0)
        node = path[-1]
         
        # Condition to check if the
        # current node is not visited
        if node not in explored:
            neighbours = graph[node]
             
            # Loop to iterate over the
            # neighbours of the node
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                 
                # Condition to check if the
                # neighbour node is the goal
                if neighbour == goal:
                    print("Shortest path = ", *new_path)
                    return
            explored.append(node)
 
    # Condition when the nodes
    # are not connected
    print("So sorry, but a connecting"\
                "path doesn't exist :(")
    return
```
---------------------------------
## Edmonds-Karp Algorithm
```python
def max_flow(C, s, t):
    n = len(C)  # C is the capacity matrix
    F = [[0] * n for i in range(n)]
    path = bfs(C, F, s, t)
    #  print path
    while path is not None:
        flow = min(C[u][v] - F[u][v] for u, v in path)
        for u, v in path:
            F[u][v] += flow
            F[v][u] -= flow
        path = bfs(C, F, s, t)
    return sum(F[s][i] for i in range(n))
```

### find path by using BFS
```python
def bfs(C, F, s, t):
    queue = [s]
    paths = {s: []}
    if s == t:
        return paths[s]
    while queue:
        u = queue.pop(0)
        for v in range(len(C)):
            if C[u][v]-F[u][v] > 0 and v not in paths:
                paths[v] = paths[u]+[(u, v)]
                if v == t:
                    return paths[v]
                queue.append(v)
    return None
```
---------------------------------
## LCS: longest common subsequence
```python
def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)
 
    # declaring the array for storing the dp values
    L = [[None]*(n + 1) for i in range(m + 1)]
 
    """Following steps build L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
 
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]
```
 
### Driver program to test the above function
```python
X = "AGGTAB"
Y = "GXTXAYB"
print("Length of LCS is ", lcs(X, Y))
```

### A Naive recursive Python implementation of LCS problem
```python 
def lcs(X, Y, m, n):
 
    if m == 0 or n == 0:
       return 0;
    elif X[m-1] == Y[n-1]:
       return 1 + lcs(X, Y, m-1, n-1);
    else:
       return max(lcs(X, Y, m, n-1), lcs(X, Y, m-1, n));
 
 
# Driver program to test the above function
X = "AGGTAB"
Y = "GXTXAYB"
print ("Length of LCS is ", lcs(X, Y, len(X), len(Y)))
```
---------------------------------
## Two pointers Technique
Find if There is a Pair in A[0..N-1] with Given Sum

Using Two-pointers Technique
 
### Method
```python
def isPairSum(A, N, X):
 
    # represents first pointer
    i = 0
 
    # represents second pointer
    j = N - 1
 
    while(i < j):
       
        # If we find a pair
        if (A[i] + A[j] == X):
            return True
 
        # If sum of elements at current
        # pointers is less, we move towards
        # higher values by doing i += 1
        elif(A[i] + A[j] < X):
            i += 1
 
        # If sum of elements at current
        # pointers is more, we move towards
        # lower values by doing j -= 1
        else:
            j -= 1
    return 0
 ```
 ```python
# array declaration
arr = [2, 3, 5, 8, 9, 10, 11]
 
# value to search
val = 17
 
print(isPairSum(arr, len(arr), val))
```
### Python Program Illustrating Naive Approach to
Find if There is a Pair in A[0..N-1] with Given Sum
### Method
```python
def isPairSum(A, N, X):
 
    for i in range(N):
        for j in range(N):
 
            # as equal i and j means same element
            if(i == j):
                continue
 
            # pair exists
            if (A[i] + A[j] == X):
                return True
 
            # as the array is sorted
            if (A[i] + A[j] > X):
                break
 
    # No pair found with given sum
    return 0
 ```
 ```python
# Driver code
arr = [2, 3, 5, 8, 9, 10, 11]
val = 17
 
print(isPairSum(arr, len(arr), val))
```
---------------------------------
## KMP Algorithm
```python
def KMPSearch(pat, txt):
    M = len(pat)
    N = len(txt)
 
    # create lps[] that will hold the longest prefix suffix
    # values for pattern
    lps = [0]*M
    j = 0 # index for pat[]
 
    # Preprocess the pattern (calculate lps[] array)
    computeLPSArray(pat, M, lps)
 
    i = 0 # index for txt[]
    while i < N:
        if pat[j] == txt[i]:
            i += 1
            j += 1
 
        if j == M:
            print ("Found pattern at index", str(i-j))
            j = lps[j-1]
 
        # mismatch after j matches
        elif i < N and pat[j] != txt[i]:
            # Do not match lps[0..lps[j-1]] characters,
            # they will match anyway
            if j != 0:
                j = lps[j-1]
            else:
                i += 1

def computeLPSArray(pat, M, lps):
    len = 0 # length of the previous longest prefix suffix
 
    lps[0] # lps[0] is always 0
    i = 1
 
    # the loop calculates lps[i] for i = 1 to M-1
    while i < M:
        if pat[i]== pat[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            # This is tricky. Consider the example.
            # AAACAAAA and i = 7. The idea is similar
            # to search step.
            if len != 0:
                len = lps[len-1]
 
                # Also, note that we do not increment i here
            else:
                lps[i] = 0
                i += 1
```
```python
txt = "ABABDABACDABABCABAB"
pat = "ABABCABAB"
KMPSearch(pat, txt)
```
### example
```
Input:  txt[] = "THIS IS A TEST TEXT"
        pat[] = "TEST"
Output: Pattern found at index 10

Input:  txt[] =  "AABAACAADAABAABA"
        pat[] =  "AABA"
Output: Pattern found at index 0
        Pattern found at index 9
        Pattern found at index 12
```
---------------------------------
## Python Program for Lowest Common Ancestor in a Binary Tree
O(n) solution to find LCS of two given values n1 and n2
Or just use meet in the middle technique yourself
```python
# A binary tree node
class Node:
    # Constructor to create a new binary node
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
 
# Finds the path from root node to given root of the tree.
# Stores the path in a list path[], returns true if path
# exists otherwise false
def findPath(root, path, k):
 
    # Baes Case
    if root is None:
        return False
 
    # Store this node is path vector. The node will be
    # removed if not in path from root to k
    path.append(root.key)
 
    # See if the k is same as root's key
    if root.key == k:
        return True
 
    # Check if k is found in left or right sub-tree
    if ((root.left != None and findPath(root.left, path, k)) or
            (root.right != None and findPath(root.right, path, k))):
        return True
 
    # If not present in subtree rooted with root, remove
    # root from path and return False
 
    path.pop()
    return False
 
# Returns LCA if node n1 , n2 are present in the given
# binary tree otherwise return -1
def findLCA(root, n1, n2):
 
    # To store paths to n1 and n2 fromthe root
    path1 = []
    path2 = []
 
    # Find paths from root to n1 and root to n2.
    # If either n1 or n2 is not present , return -1
    if (not findPath(root, path1, n1) or not findPath(root, path2, n2)):
        return -1
 
    # Compare the paths to get the first different value
    i = 0
    while(i < len(path1) and i < len(path2)):
        if path1[i] != path2[i]:
            break
        i += 1
    return path1[i-1]
 ```
 ```python 
# Driver program to test above function
if __name__ == '__main__':
     
    # Let's create the Binary Tree shown in above diagram
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.left.left = Node(4)
    root.left.right = Node(5)
    root.right.left = Node(6)
    root.right.right = Node(7)
     
    print("LCA(4, 5) = %d" % (findLCA(root, 4, 5,)))
    print("LCA(4, 6) = %d" % (findLCA(root, 4, 6)))
    print("LCA(3, 4) = %d" % (findLCA(root, 3, 4)))
    print("LCA(2, 4) = %d" % (findLCA(root, 2, 4)))
```
---------------------------------
## Two-pointer techinque
The two-pointer technique is a commonly used algorithmic technique to solve problems that involve traversing a sequence, such as arrays or linked lists, by using two pointers or indices that move through the sequence in different ways. One pointer moves faster than the other, or they move in opposite directions, until they both meet at a certain condition or goal.

Here's a simple example of the two-pointer technique in Python:

Problem statement: Given an array of integers, find two numbers such that they add up to a specific target number.

```python
def two_sum(nums, target):
    # Sort the array in ascending order
    nums.sort()
    
    # Initialize two pointers, one at the beginning and one at the end
    left, right = 0, len(nums) - 1
    
    # Move the pointers until they meet or until the sum is found
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            # Move the left pointer to the right to increase the sum
            left += 1
        else:
            # Move the right pointer to the left to decrease the sum
            right -= 1
    
    # If no sum is found, return an empty array
    return []
```

In this example, we first sort the array to make it easier to find the sum. Then we initialize two pointers at the beginning and end of the array. We move the pointers inwards until we find a sum equal to the target, or until they meet. If the sum is less than the target, we move the left pointer to the right to increase the sum. If the sum is greater than the target, we move the right pointer to the left to decrease the sum. Finally, if no sum is found, we return an empty array.

For example, given the array `[2, 7, 11, 15]` and the target `9`, the function would return `[0, 1]` because `2 + 7 = 9`.

---------------------------------
## knapsack, the bag thingy
This is the memoization approach of
0 / 1 Knapsack in Python in simple
we can say recursion + memoization = DP
```python 
def knapsack(wt, val, W, n):
 
    # base conditions
    if n == 0 or W == 0:
        return 0
    if t[n][W] != -1:
        return t[n][W]
 
    # choice diagram code
    if wt[n-1] <= W:
        t[n][W] = max(
            val[n-1] + knapsack(
                wt, val, W-wt[n-1], n-1),
            knapsack(wt, val, W, n-1))
        return t[n][W]
    elif wt[n-1] > W:
        t[n][W] = knapsack(wt, val, W, n-1)
        return t[n][W]
 
# Driver code
if __name__ == '__main__':
    profit = [60, 100, 120]
    weight = [10, 20, 30]
    W = 50
    n = len(profit)
     
    # We initialize the matrix with -1 at first.
    t = [[-1 for i in range(W + 1)] for j in range(n + 1)]
    print(knapsack(weight, profit, W, n))
```
### Program for 0-1 Knapsack problem
Returns the maximum value that can
be put in a knapsack of capacity W
 
```python
def knapSack(W, wt, val, n):
    K = [[0 for x in range(W + 1)] for x in range(n + 1)]
 
    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i-1] <= w:
                K[i][w] = max(val[i-1]
                              + K[i-1][w-wt[i-1]],
                              K[i-1][w])
            else:
                K[i][w] = K[i-1][w]
 
    return K[n][W]
 
 
# Driver code
if __name__ == '__main__':
    profit = [60, 100, 120]
    weight = [10, 20, 30]
    W = 50
    n = len(profit)
    print(knapSack(W, weight, profit, n))
```

### simplest:
```python
def knapSack(W, wt, val, n):
     
    # Making the dp array
    dp = [0 for i in range(W+1)]
 
    # Taking first i elements
    for i in range(1, n+1):
       
        # Starting from back,
        # so that we also have data of
        # previous computation when taking i-1 items
        for w in range(W, 0, -1):
            if wt[i-1] <= w:
                 
                # Finding the maximum value
                dp[w] = max(dp[w], dp[w-wt[i-1]]+val[i-1])
     
    # Returning the maximum value of knapsack
    return dp[W]
 
 
# Driver code
if __name__ == '__main__':
    profit = [60, 100, 120]
    weight = [10, 20, 30]
    W = 50
    n = len(profit)
    print(knapSack(W, weight, profit, n))
```
