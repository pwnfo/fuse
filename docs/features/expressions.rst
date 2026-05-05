Expressions & Classes
=====================

At the core of Fuse is the expression parser. 

Literal Characters
------------------
Literal characters produce themselves. For example, ``admin`` literally yields ``admin``. If you concatenate it with numbers, it produces combinations.

.. code-block:: bash
   
   $ fuse 'admin/d'
   admin0
   admin1
   ...
   admin9

Built-in Character Classes
--------------------------
To easily reference standard character sets, Fuse provides built-in tokens initialized with a forward slash ``/``.

========= ======================== =====================================
Symbol    Meaning                  Example / Resulting Output          
========= ======================== =====================================
``/l``    Letters                  ``a–z`` and ``A–Z``
``/a``    Lowercase letters        ``a–z``
``/A``    Uppercase letters        ``A–Z``
``/d``    Digits                   ``0–9``
``/D``    Non-zero digits          ``1–9``
``/h``    Hexadecimal (lower)      ``0–9``, ``a–f``
``/H``    Hexadecimal (upper)      ``0–9``, ``A–F``
``/s``    Whitespace               `` `` (Space)
``/o``    Octal digits             ``0–7``
``/p``    Special characters       ``!@#$%^&*()-_+=``
``/N``    Newline                  ``\n``
========= ======================== =====================================

Custom Classes and Unions
-------------------------
You can construct your own custom character classes by wrapping items in brackets ``[...]``.
For example, ``[abc]`` generates the characters ``a``, ``b``, or ``c``.

**Mixing Built-in Classes inside Brackets**
Built-in tokens naturally expand *inside* brackets.

.. code-block:: bash

   $ fuse '[/d/a_]'
   # Yields all digits, lowercase letters, and underscores: 0, 1..., a, b..., _

**Unions (Alternatives)**
Use the pipe character ``|`` to separate full-word alternatives inside brackets. Each option acts as a discrete, indivisible block.

.. code-block:: bash
   
   $ fuse '[admin|root|123]'
   admin
   root
   123

This allows grouping multiple payloads logically.

Literal Groups
--------------
If you need an entire word sequence to be treated strictly as a single permutation unit without interpreting inner unions, you can wrap it in literal parenthesis ``(...)``. This is highly useful when combined with quantifiers.

.. code-block:: bash

   $ fuse '(admin){3}'
   adminadminadmin

Quantifiers
-----------
Control repetition using brace syntax. The syntax supports minimum and maximum repetitions.

* ``{N}`` — Repeat exactly **N** times.
* ``{min,max}`` — Repeat between **min** and **max** times (inclusive).
* ``?`` — Sugar syntax for exactly **0 or 1** time (optional).

Examples:

.. code-block:: bash

   $ fuse '[XYZ]{3}'
   # Yields: XXX, XXY, XXZ, XYX... (27 permutations)

   $ fuse '[XYZ]{1,2}'
   # Yields: X, Y, Z, XX, XY, XZ... (12 permutations)

   $ fuse '(admin)?[12]'
   # Yields: 1, 2, admin1, admin2

Escaping
--------
Need to output a reserved token like ``/d``, ``[``, ``}``, or ``^``? Escape it using a backslash ``\``.

.. code-block:: bash

   $ fuse '\/d/d'
   /d0
   /d1
   ...
