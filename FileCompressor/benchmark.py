import os
import zlib
import gzip
import bz2
import lzma
import time
import sys

original_size_text = os.path.getsize('sample.txt')
# compressed_size = os.path.getsize('test.huf')
# uncompressed_size = os.path.getsize('uncompressed_text.txt')
# print(f"The Compression Ratio is {original_size_text/compressed_size}")

original_size_image = os.path.getsize('sample.bmp')
# compressed_size = os.path.getsize('test.huf')
# uncompressed_size = os.path.getsize('uncompressed_image.png')
# print(f"The Compression Ratio is {original_size_image/compressed_size}")

with open('sample.bmp', 'rb') as f:
    image = f.read()

with open('sample.txt', 'rb') as f:
    text = f.read()

start_gzip_compress_image = time.time()
gzip_compressed_image = lzma.compress(image)
end_gzip_compress_image = time.time()
print(f'gzip image compression time is {end_gzip_compress_image-start_gzip_compress_image}')
print(f'gzip image compression ratio is {sys.getsizeof(image)/sys.getsizeof(gzip_compressed_image)}')
start_gzip_decompress_image = time.time()
gzip_decompressed_image = lzma.decompress(gzip_compressed_image)
end_gzip_decompress_image = time.time()
print(f'gzip image decompression time is {end_gzip_decompress_image-start_gzip_decompress_image}')


start_gzip_compress_text = time.time()
gzip_compressed_text = lzma.compress(text)
end_gzip_compress_text = time.time()
print(f'gzip text compression time is {end_gzip_compress_text-start_gzip_compress_text}')
print(f'gzip text compression ratio is {sys.getsizeof(text)/sys.getsizeof(gzip_compressed_text)}')
start_gzip_decompress_text = time.time()
gzip_decompressed_text = lzma.decompress(gzip_compressed_text)
end_gzip_decompress_text = time.time()
print(f'gzip text decompression time is {end_gzip_decompress_text-start_gzip_decompress_text}')



