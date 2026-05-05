CLI Reference & Flags
=====================

Fuse comes packed with a robust options suite designed for production pipelines.

Usage Syntax
------------

.. code-block:: text

   usage: fuse [options] <expression> [<files...>]

Options
-------

**Core Options**

* ``-o <path>, --output <path>``: Writes the generated wordlist reliably into a target file.
* ``-f <path>, --file <path>``: Instead of an inline expression, runs a ``.fuse`` definition file.
* ``-q, --quiet``: Disable progress bars and metric statistics. Great for `bash` pipes.
* ``-n, --non-interactive``: Runs without interactive prompts.
* ``-d <word>, --delimiter <word>``: Replaces the default newline (``\n``) delimiter with custom strings. Optional strings like ``\0`` can be used for zero-byte split integration.

**Performance & Scaling**

* ``-b <bytes>, --write-buffer <bytes>``: Explicitly sets write buffer size (e.g. ``50MB``, ``1GB``) for IO optimization.
* ``-w <1-64>, --workers <1-64>``: Distributes combinatorial operations across ``N`` processes. Default is 1.
* ``-k <bytes>, --flush-threshold <bytes>``: Sets byte threshold before flushing output buffer. Default is 512KB.
* ``-z <format>, --compress <format>``: Compress output using specified format. Available: ``gzip``, ``bzip2``, ``lzma``.

**Filtration & Range**

* ``-F <regex>, --filter <regex>``: Pre-evaluates tokens using standard Python regex before outputting.
* ``-S <word>, --start <word>``: Starts writing specifically from ``<word>`` forward.
* ``-E <word>, --end <word>``: Stops execution, ending precisely on ``<word>``.
