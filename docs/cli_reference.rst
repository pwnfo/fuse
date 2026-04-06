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
* ``-s <word>, --separator <word>``: Replaces the default newline (``\n``) separator with custom strings. Optional strings like ``\0`` can be used for zero-byte split integration.

**Performance & Scaling**

* ``-b <bytes>, --buffer <bytes>``: Explicitly sets chunk buffer targets (e.g. ``50MB``, ``1GB``) for IO optimization.
* ``-w <1-64>, --workers <1-64>``: Distributes combinatorial operations across ``N`` processes. Default is 1.

**Filtration**

* ``-F <regex>, --filter <regex>``: Pre-evaluates tokens using standard Python regex before outputting.
* ``--from <word>``: Starts writing specifically from ``<word>`` forward.
* ``--to <word>``: Caps execution explicitly ending precisely on ``<word>``.
