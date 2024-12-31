import os
import sys
import marshal
import itertools
import argparse
from operator import itemgetter
from functools import partial
from collections import Counter
import time

try:
    import cPickle as pickle
except:
    import pickle

termchar = 7  # I assume the byte 7 does not appear in the input file


class HeapPQ:
    def __init__(self, nodes=None):
        if nodes is None:
            nodes = []
        self._entries = nodes

    def insert(self, node):
        self._entries.append(node)
        self.upheap(len(self._entries) - 1)

    def parent(self, i):
        return (i - 1) // 2

    def children(self, i):
        left = 2 * i + 1
        right = 2 * i + 2
        return range(left, min(len(self._entries), right + 1))

    def swap(self, a, b):
        L = self._entries
        L[a], L[b] = L[b], L[a]

    def upheap(self, i):
        L = self._entries
        parent = self.parent(i)
        if i > 0 and L[i] < L[parent]:
            self.swap(i, parent)
            self.upheap(parent)

    def findmin(self):
        return self._entries[0].item

    def removemin(self):
        L = self._entries
        item = L[0]
        L[0] = L[-1]
        L.pop()
        self.downheap(0)
        return item

    def downheap(self, i):
        L = self._entries
        children = self.children(i)
        if children:
            child = min(children, key=lambda x: L[x])
            if L[child] < L[i]:
                self.swap(i, child)
                self.downheap(child)

    def _heapify(self):
        n = len(self._entries)
        for i in reversed(range(n)):
            self.downheap(i)

    def __len__(self):
        return len(self._entries)


def ibwt(msg):
    # I would work with a bytearray to store the IBWT output
    n = len(msg)

    count = [0] * 256
    for char in msg:
        count[char] += 1

    cum_count = [0] * 256
    for i in range(1, 256):
        cum_count[i] = cum_count[i - 1] + count[i - 1]

    next_array = [0] * n
    for i in range(n):
        char = msg[i]
        next_array[i] = cum_count[char]
        cum_count[char] += 1

    result = [0] * n
    index = 0
    for i in range(n):
        result[i] = msg[index]
        index = next_array[index]

    result.remove(termchar)

    return bytearray(result[::-1])


# Burrows-Wheeler Transform fncs
def radix_sort(values, key, step=0):
    sortedvals = []
    radix_stack = []
    radix_stack.append((values, key, step))

    while len(radix_stack) > 0:
        values, key, step = radix_stack.pop()
        if len(values) < 2:
            for value in values:
                sortedvals.append(value)
            continue

        bins = {}
        for value in values:
            bins.setdefault(key(value, step), []).append(value)

        for k in sorted(bins.keys()):
            radix_stack.append((bins[k], key, step + 1))
    return sortedvals


# memory efficient BWT
def bwt(msg):
    def bw_key(text, value, step):
        return text[(value + step) % len(text)]

    msg = msg + termchar.to_bytes(1, byteorder='big')

    bwtM = bytearray()

    rs = radix_sort(range(len(msg)), partial(bw_key, msg))
    for i in rs:
        bwtM.append(msg[i - 1])

    return bwtM[::-1]


# move-to-front encoding fncs
def mtf(msg):
    # Initialise the list of characters (i.e. the dictionary)
    dictionary = bytearray(range(256))

    # Transformation
    compressed_text = bytearray()
    rank = 0

    # read in each character
    for c in msg:
        rank = dictionary.index(c)  # find the rank of the character in the dictionary
        compressed_text.append(rank)  # update the encoded text

        # update the dictionary
        dictionary.pop(rank)
        dictionary.insert(0, c)

    # dictionary.sort() # sort dictionary
    return compressed_text  # Return the encoded text as well as the dictionary


# inverse move-to-front
def imtf(compressed_msg):
    compressed_text = compressed_msg
    dictionary = bytearray(range(256))

    decompressed_img = bytearray()

    # read in each character of the encoded text
    for i in compressed_text:
        # read the rank of the character from dictionary
        decompressed_img.append(dictionary[i])

        # update dictionary
        e = dictionary.pop(i)
        dictionary.insert(0, e)

    return decompressed_img  # Return original string


def build_decoder_ring(huffmantree, code='', huffman_encoder_map=None, huffman_decoder_map=None):
    if huffman_decoder_map is None:
        huffman_decoder_map = dict()
    if huffman_encoder_map is None:
        huffman_encoder_map = dict()
    if huffmantree:
        if not huffmantree.left and not huffmantree.right:
            huffman_decoder_map[code] = huffmantree.character
            huffman_encoder_map[huffmantree.character] = code
        if huffmantree.left:
            build_decoder_ring(huffmantree.left, code + '0', huffman_encoder_map, huffman_decoder_map)
        if huffmantree.right:
            build_decoder_ring(huffmantree.right, code + '1', huffman_encoder_map, huffman_decoder_map)
    return huffman_encoder_map, huffman_decoder_map


class Node:
    def __init__(self, frequency, character, left=None, right=None):
        self.frequency = frequency
        self.left = left
        self.right = right
        self.character = character
        self.huffmancode = ''

    def __lt__(self, other):
        return self.frequency < other.frequency


def generate_frequency_table(msg):
    frequency_table = dict()
    for char in msg:
        if char in frequency_table:
            frequency_table[char] += 1
        else:
            frequency_table[char] = 1
    return frequency_table


# This takes a sequence of bytes over which you can iterate, msg,
# and returns a tuple (enc,\ ring) in which enc is the ASCII representation of the 
# Huffman-encoded message (e.g. "1001011") and ring is a ``decoder ring'' needed
# to decompress that message.
def encode(msg):
    msg_frequency_table = generate_frequency_table(msg)
    # Create min heap with nodes for each char in msg_frequency_table
    # Create huffman tree
    nodes = []
    for i in msg_frequency_table:
        nodes.append(Node(msg_frequency_table[i], i))
    nodes = HeapPQ(nodes)
    # build huffman tree
    while len(nodes) > 1:
        left = nodes.removemin()
        right = nodes.removemin()
        left.huffmancode = 0
        right.huffmancode = 1
        nodes.insert(Node(left.frequency + right.frequency, left.character + right.character, left, right))
    # Store huffman decoder from tree to hashmap
    huffman_encoder_map, huffman_decoder_map = build_decoder_ring(nodes.removemin())
    # using huffman map, find encoded message
    encoded_msg = ''
    for char in msg:
        encoded_msg += huffman_encoder_map[char]
    return encoded_msg, huffman_decoder_map


# This takes a string, cmsg, which must contain only 0s and 1s, and your
# representation of the ``decoder ring'' ring, and returns a bytearray msg which 
# is the decompressed message. 
def decode(cmsg, decoderRing, padding):
    # Creates an array with the appropriate type so that the message can be decoded.
    cmsg = ''.join(format(byte, '08b') for byte in cmsg)
    if padding != 0:
        cmsg = cmsg[:len(cmsg) - (8 - padding)]
    current_str = ''
    decoded_bytes = []
    for i in range(len(cmsg)):
        current_str += cmsg[i]
        if current_str in decoderRing:
            decoded_bytes.append(decoderRing[current_str])
            current_str = ''
    byteMsg = bytearray(decoded_bytes)
    return byteMsg


# This takes a sequence of bytes over which you can iterate, msg, and returns a tuple (compressed, ring)
# in which compressed is a bytearray (containing the Huffman-coded message in binary, 
# and ring is again the ``decoder ring'' needed to decompress the message.
def compress(msg, useBWT):
    if useBWT:
        msg = bwt(msg)
        msg = mtf(msg)
    encoded_msg, decoder_ring = encode(msg)
    length = len(encoded_msg)
    padding = length % 8
    if padding != 0:
        pad = (8 - padding) * '0'
        encoded_msg += pad
    encoded_bytes = int(encoded_msg, 2).to_bytes(len(encoded_msg) // 8, byteorder='big')
    compressed = bytearray(encoded_bytes)
    return compressed, decoder_ring, padding


# This takes a sequence of bytes over which you can iterate containing the Huffman-coded message, and the
# decoder ring needed to decompress it.  It returns the bytearray which is the decompressed message. 
def decompress(msg, decoderRing, useBWT, padding):
    # Creates an array with the appropriate type so that the message can be decoded.
    decompressedMsg = decode(msg, decoderRing, padding)
    if useBWT:
        decompressedMsg = imtf(decompressedMsg)
        decompressedMsg = ibwt(decompressedMsg)
    return decompressedMsg


with open('sample.txt', 'rb') as f:
    text = f.read()

start_gzip_compress_text = time.time()
compressed_text, image_decoder_ring, image_padding = compress(text, True)
end_gzip_compress_text = time.time()
print(f'gzip text compression time is {end_gzip_compress_text-start_gzip_compress_text}')
print(f'gzip text compression ratio is {sys.getsizeof(text)/sys.getsizeof(compressed_text)}')
start_gzip_decompress_text = time.time()
gzip_decompressed_text = decompress(compressed_text, image_decoder_ring, True, image_padding)
end_gzip_decompress_text = time.time()
print(f'gzip text decompression time is {end_gzip_decompress_text-start_gzip_decompress_text}')

if __name__ == '__main__':

    # argparse is an excellent library for parsing arguments to a python program
    parser = argparse.ArgumentParser(description='<Insert a cool name for your compression algorithm> compresses '
                                                 'binary and plain text files using the Burrows-Wheeler transform, '
                                                 'move-to-front coding, and Huffman coding.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-c', action='store_true', help='Compresses a stream of bytes (e.g. file) into a bytes.')
    group.add_argument('-d', action='store_true', help='Decompresses a compressed file back into the original input')
    group.add_argument('-v', action='store_true', help='Encodes a stream of bytes (e.g. file) into a binary string'
                                                       ' using Huffman encoding.')
    group.add_argument('-w', action='store_true', help='Decodes a Huffman encoded binary string into bytes.')
    parser.add_argument('-i', '--input', help='Input file path', required=True)
    parser.add_argument('-o', '--output', help='Output file path', required=True)
    parser.add_argument('-b', '--binary', help='Use this option if the file is binary and therefore '
                                               'do not want to use the BWT.', action='store_true')

    args = parser.parse_args()

    compressing = args.c
    decompressing = args.d
    encoding = args.v
    decoding = args.w

    infile = args.input
    outfile = args.output
    useBWT = not args.binary

    assert os.path.exists(infile)

    if compressing or encoding:
        fp = open(infile, 'rb')
        sinput = fp.read()
        fp.close()
        if compressing:
            msg, tree, padding = compress(sinput, useBWT)
            fcompressed = open(outfile, 'wb')
            marshal.dump((pickle.dumps(tree), msg, padding), fcompressed)
            fcompressed.close()
        else:
            msg, tree = encode(sinput)
            print(msg)
            fcompressed = open(outfile, 'wb')
            marshal.dump((pickle.dumps(tree), msg), fcompressed)
            fcompressed.close()
    else:
        fp = open(infile, 'rb')
        pck, msg, padding = marshal.load(fp)
        tree = pickle.loads(pck)
        fp.close()
        if decompressing:
            sinput = decompress(msg, tree, useBWT, padding)
        else:
            sinput = decode(msg, tree, padding)
            print(sinput)
        fp = open(outfile, 'wb')
        fp.write(sinput)
        fp.close()
