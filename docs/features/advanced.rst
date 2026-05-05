Advanced Concepts
=================

Expression Files (.fuse)
------------------------
For very complex generations, you can author ``.fuse`` files instead of invoking the CLI. These files allow defining explicit aliases and sequentially joining outputs from multiple sub-expressions. 

**Syntax Overview**

- Comments start with ``#`` (it must be followed by a space to be interpreted as a comment).
- ``%define name pattern``: Replace ``$name;`` with ``pattern`` throughout the rest of the file.
- ``%include filename.txt``: Expressly opens a file relative to the ``.fuse`` script or an absolute path.
- **Important**: When you declare a ``%include``, that file is bound to the ``^`` placeholder in the **very next expression line**. It does not persist globally. You can declare multiple ``%define`` lines consecutively to bind to multiple ``^`` placeholders in the next expression.
- Any other (non-empty) line is treated as an expression.

**Example payloads.fuse**:

.. code-block:: text

   # Define Reusable Payload Aliases
   %define DIGIT #[0-9]
   %define BASE_URL (https://example.com/api/)
   
   # Include a dictionary text file from previous runs
   %include default_paths.txt

   # Expression Generation
   $BASE_URL;^\?id=$DIGIT; # Example: https://example.com/api/account?id=1
   $BASE_URL;v$DIGIT; # Example: https://example.com/api/v1

Run using ``-f`` or ``--file``:

.. code-block:: bash

   fuse -f payloads.fuse

Smart Skipping & Chunking
-------------------------
Large permutations quickly hit constraints. Fuse addresses this through algorithmic seeking. Instead of creating combinations starting from `A` waiting until it hits your target, Fuse calculates precisely where a specific target begins and resumes generation from there optimally.

You can segment workloads using ``-S/--start`` and ``-E/--end``.

.. code-block:: bash

   $ fuse '/l{4}' -S abcd -E wxyz
   abce
   abcf
   ...
   wxyz

This logic applies cleanly natively even when distributing across threads.

Multi-threading
---------------------------
You can specify multiple workers via ``-w <int>``. 
Fuse intelligently delegates disjoint segments of the permutation space to each worker.

.. code-block:: bash
   
   # using 3 different workers to write
   $ fuse '[/l/d]{5}' -w 3 -o output.txt


Regex Output Filtering
----------------------
Do you want to impose restrictions on expressions like "Strings must start with ``3`` or ``5``"?
Generate the superset pattern and then redirect or filter it using ``-F "REGEX"``.

.. code-block:: bash
   
   $ fuse '[/l/d]{3}' -F '^(3|5)'

*Warning: Filtering evaluates dynamically, meaning permutations discarded incur minor performance penalties due to skipped outputs.*
