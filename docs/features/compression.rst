Compression Output
=====================

Fuse supports on-the-fly compression when writing output files. This is useful for large wordlists where disk space and I/O efficiency matter.

Use the ``-z`` (or ``--compress``) flag together with ``-o``:

.. code-block:: bash

   # Gzip compression (balanced speed/ratio)
   $ fuse '/l{5}' -z gzip -o wordlist.txt.gz

   # LZMA compression (best ratio, slower)
   $ fuse '/l{5}' -z lzma -o wordlist.txt.xz

   # Bzip2 compression (middle ground)
   $ fuse '/l{5}' -z bzip2 -o wordlist.txt.bz2

Compression is applied during generation:

* No intermediate uncompressed file is created
* Output is streamed directly into the compressor
* Memory usage remains minimal


Flush Threshold and Performance
-------------------------------

When using compression, the ``-k`` / ``--flush-threshold`` parameter becomes critical for performance tuning.

This value defines how many bytes are buffered before being flushed to the output stream (and consequently to the compressor).

- Smaller values (e.g., ``64KB``):

  - Lower memory usage
  - More frequent flushes
  - ❌ Worse compression ratio
  - ❌ Higher CPU overhead

- Larger values (e.g., ``1MB`` or more):

  - Better compression efficiency (more data per compression block)
  - Fewer I/O operations
  - ✔️ Higher throughput
  - ❌ Increased memory usage

.. code-block:: bash

   # Optimized for compression efficiency
   $ fuse '/l{6}' -z lzma -k 2MB -o output.txt.xz

   # Optimized for low memory environments
   $ fuse '/l{6}' -z gzip -k 128KB -o output.txt.gz

**Important**: Compression algorithms benefit from larger contiguous data blocks.  
Using a very low ``--flush-threshold`` can significantly degrade compression ratio and overall performance.


Compression Comparison
----------------------

+----------+---------------------+------------------+------------------+-----------------------------+
| Format   | Compression Ratio   | Speed            | CPU Usage        | Recommended Use Case        |
+==========+=====================+==================+==================+=============================+
| gzip     | Medium              | Fast             | Low              | General purpose, fast I/O   |
+----------+---------------------+------------------+------------------+-----------------------------+
| bzip2    | High                | Medium           | Medium           | Better compression balance  |
+----------+---------------------+------------------+------------------+-----------------------------+
| lzma     | Very High           | Slow             | High             | Maximum compression         |
+----------+---------------------+------------------+------------------+-----------------------------+


Compression Level
-----------------

You can control the compression strength using ``-l`` / ``--compresslevel``.

The accepted range depends on the selected format:

- ``gzip``: ``0–9`` (``0`` = no compression, ``9`` = best compression)
- ``bzip2``: ``1–9`` (higher = better compression, slower)
- ``lzma``: ``0–9`` (preset levels, default is ``6``)

.. code-block:: bash

   # Faster compression (lower ratio)
   $ fuse '/l{5}' -z gzip -l 1 -o fast.txt.gz

   # Maximum compression
   $ fuse '/l{5}' -z bzip2 -l 9 -o compact.txt.bz2

   # LZMA high compression preset
   $ fuse '/l{5}' -z lzma -l 9 -o ultra.txt.xz

If ``--compresslevel`` is not specified, each algorithm uses its own default:

- ``gzip``: ``9``
- ``bzip2``: ``9``
- ``lzma``: ``6``

**Note**: Lower levels improve speed but reduce compression efficiency.


Practical Guidance
------------------

- Use ``gzip`` for most cases (best trade-off)
- Use ``lzma`` when disk space is critical
- Use ``bzip2`` for a balance between compression ratio and speed
- Increase ``--flush-threshold`` when prioritizing compression efficiency
- Decrease it in constrained environments or streaming scenarios


Notes
-----

- File extensions are not enforced but should match the format:

  - ``.gz`` for gzip
  - ``.bz2`` for bzip2
  - ``.xz`` for lzma

- Compression may become CPU-bound when using multiple workers (``-w``)

- Extremely low flush thresholds can negate the benefits of compression
