import numpy as np
import random


def uniform(N, k):
    """Uniform distribution of 'N' items into 'k' groups."""
    dist = []
    avg = N / k
                       
    for i in range(k):
        dist.append(int((i + 1) * avg) - int(i * avg))
                                  
    random.shuffle(dist)
    return dist


def normal(N, k):
    """Normal distribution of 'N' items into 'k' groups."""
    dist = []
                       
    for i in range(k):
        x = i - (k - 1) / 2
        dist.append(int(N * (np.exp(-x) / (np.exp(-x) + 1)**2)))
                    
    remainder = N - sum(dist)
    dist = list(np.add(dist, uniform(remainder, k)))
                                      
    return dist
