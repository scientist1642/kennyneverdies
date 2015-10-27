import random
import math
from bloomfilter import BloomFilter

bf = BloomFilter(101, 5)
bf.insert(56)
assert not bf.contains(99)
assert bf.contains(56)


bitsize = 16000000
bf_num_count =  1000000
test_num_count = 100000000

hash_count = int(math.log(2) * bitsize / bf_num_count)
print 'number of hash functions to use:', hash_count

bf = BloomFilter(bitsize, hash_count) 

bf_nums = set()
for _ in xrange(bf_num_count):
    rand_int = random.randint(0, 2**64)
    bf_nums.add(rand_int)
    bf.insert(rand_int)

false_alarms = 0
for _ in xrange(test_num_count):
    rand_num = random.randint(0, 2**64)
    if bf.contains(rand_num) and not (rand_num in bf_nums):
        false_alarms += 1

print "false alarms ", false_alarms 
print "ratio",  false_alarms / float(test_num_count)
# prints 
# 11 - optiomal hash functions
# false alarms  46017
# ratio 0.00046017



# 8 - not optimal

# false alarms  57690
# ratio 0.0005769
