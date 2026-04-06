Ranges & Placeholders
=====================

In addition to static characters, Fuse enables dynamically injecting external sources and numerical ranges into expressions.

Numeric Ranges
--------------
Rather than using ``[1|2|3|...|100]``, you can use the built-in numeric range syntax ``#[start-end:step]``.
These ranges can be placed anywhere inside an expression.

* **Ascending**: ``#[1-5]`` yields ``1, 2, 3, 4, 5``
* **Descending**: ``#[5-1]`` yields ``5, 4, 3, 2, 1``
* **With Steps**: ``#[2-10:2]`` yields ``2, 4, 6, 8, 10``

Examples:

.. code-block:: bash

   $ fuse 'user#[1-10]'
   user1
   user2
   ...
   user10

File Placeholders
-----------------
You can inject contents from external wordlists directly into a Fuse generation using the caret placeholder ``^``. Each caret consumes one sequential file argument passed to the CLI.

.. code-block:: bash

   $ fuse '^-^' firstnames.txt lastnames.txt
   Bob-Smith
   Bob-Doe
   Ana-Smith
   Ana-Doe

**File and Range Quantifiers**
Remember that placeholders and ranges are native tokens! You can apply `Quantifiers <expressions.html#quantifiers>`_ to them directly to repeat configurations.

.. code-block:: bash

   $ fuse '^{2}' colors.txt
   RedRed
   RedBlue
   ...

   $ fuse '#[1-3]{3}'
   111
   112
   ...

Fuse calculates permutations gracefully, properly nesting iterations. Since files are parsed as native tokens, memory usage is kept extremely low, reading the file securely without caching the entire permutations tree in RAM.

Inline Macros
-------------
Don't want to create an external file just for a small set of words? You can supply inline payload arrays by prefixing arguments with ``//``. This treats the argument itself as a pattern replacement for ``^``.

.. code-block:: bash

   $ fuse 'login-^-^' '//[user|admin]' '//[1|2]'
   login-user-1
   login-user-2
   login-admin-1
   login-admin-2
