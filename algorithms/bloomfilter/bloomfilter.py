import random
from bitarray import bitarray
from prime import generateLargePrime

class BloomFilter(object):

    def __init__(self, bitarr_size, hash_count):
        """
        Args:
            bitarr_size: size of a bit array to use
            hash_count: number of hash functions
        """
        self.m = bitarr_size
        self.k = hash_count
        self.arr = bitarray(self.m) 
        
        # init hash functions, we use (ax+b) mod p mod m
        self.hfs = self.gen_hash_params()

    def gen_hash_params(self):
        """ generates a, b and p and returns array of tuples"""
        m_bitlength = self.m.bit_length()
        hfs = []
        for _ in xrange(self.k):
            p = generateLargePrime(m_bitlength + 7)
            a = random.randint(1, p)
            b = random.randint(1, p)
            hfs.append((a, b, p))
        return hfs

    def hash_value(self, x, h_num):
        """ get value of h_num-th hash function """
        a, b, p = self.hfs[h_num]
        return (a * x + b) % p % self.m
        

    def insert(self, item):
        """ insert item(so far num) in a bloom filter """
        for h_num in xrange(self.k):
            val = self.hash_value(item, h_num)
            self.arr[val] = True

    def contains(self, item):
        """ determine if the item possibly exists """
        for h_num in xrange(self.k):
            val = self.hash_value(item, h_num)
            if not self.arr[val]:
                return False
        else:
            return True

