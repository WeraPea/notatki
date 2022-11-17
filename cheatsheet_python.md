import statistics

statistics.mode()

    most recuring element

statistics.multimode()

    most recuring element (list if multiple)

random.randrange(1,100,2)
    
    rand odd number

math.comb(n, k) math.perm() <- better cares about order

    Return the number of ways to choose k items from n items without repetition and without order.

    Evaluates to n! / (k! * (n - k)!) when k <= n and evaluates to zero when k > n.

    Also called the binomial coefficient because it is equivalent to the coefficient of k-th term in polynomial expansion of (1 + x)ⁿ.

    Raises TypeError if either of the arguments are not integers. Raises ValueError if either of the arguments are negative.
    
math.fsum

    floating point accurate

math.gcd(*integers)

    Return the greatest common divisor of the specified integer arguments.
    If any of the arguments is nonzero, then the returned value is the largest positive integer that is a divisor of all arguments.
    If all arguments are zero, then the returned value is 0. gcd() without arguments returns 0.

math.lcm()
       
    Return the least common multiple of the specified integer arguments.
    If all arguments are nonzero, then the returned value is the smallest positive integer that is a multiple of all arguments.
    If any of the arguments is zero, then the returned value is 0. lcm() without arguments returns 1.

math.prod()

    product of all items in an iterable + start value (1)
  


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

def nwd(x,y): // or just math.gcd()
    if x > 0:
        return nwd(y, x%y)
    else:
        return x
# if more than 2 numbers:
# nwd(a,b,c) == nwd(nwd(a,b),c)

def nww(x,y): // or just math.lcm() >= 3.9
    return (x*y)/nwd(x,y)


# Function to find the partition position
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

print(primeFactorization(1024))
```
