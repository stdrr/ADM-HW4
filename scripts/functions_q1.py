# hash function 
def hash_function (string, coeff):
    
    total_sum = 0
    
    for i in range(0,32):
        total_sum += coeff[i]*ord(string[i]) 
    
    hash_value = total_sum % (pow(2,32))
    
    return hash_value




# Functions for the hyperLogLog structure 

# this function gives the position of the leftmost 1, e.g. pho(101000) = 3 
def pho (binary):
    index = binary.rfind('1') + 1
    
    return len(binary)-index

# this function converts numbers in their 32-bit binary representation, e.g. convert_to_32bit(98765754) = 00000101111000110000101110111010
def convert_to_32bit(num):
    binary_num = bin(num)[2:].zfill(32) 
    
    return binary_num


# this function splits a number in two numbers: one made up of the first 6 figure and the second made up of the remaining figures, 
# e.g. splitHash_6 (1010010110110) = (101001, 0110110)
def splitHash_6 (binary):
    name_buck = binary[:6]
    hash_buck = binary[6:]
    
    return int('0b' + name_buck, 2), hash_buck




# this function creates the buckets of a 6-bit HyperLogLog structure so will return a list like [5, 8, 9, ... 
def hyperLogLog (data, coeff):
    
    buckets = [0]*2**6
    for x in data[0]: # read the data sequentially
        hash_value = hash_function(x, coeff)   # create the corresponding hash value
        binary_hash_value = convert_to_32bit(hash_value)  # find its binary representation
        name_buck, hash_buck = splitHash_6 (binary_hash_value)   # split the binary representation in (bucket_number, rest_of_number)
        
        buckets[name_buck] = max(buckets[name_buck], pho(hash_buck))  # update the buckets with the highest count of zeroes 
        
    return buckets



# this function gives an estimate of the cardinality of a set given the buckets of the HyperLogLog structure
def cardinality_hyperLogLog (buckets):
 
    length = len(buckets)
    alpha = 2*0.709
    
    tot = 0 
    for bucket in buckets:
        tot += pow(2, -bucket)
    
    
    return alpha*(length**2)/tot