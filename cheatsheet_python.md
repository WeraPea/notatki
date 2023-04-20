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
    for i in range(2,n + 1):
        if n % i == 0:
            count = 1
            for j in range(2,(i//2 + 1)):
                if(i % j == 0):
                    count = 0
                    break
            if(count == 1):
                print(i)
    #TODO
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
