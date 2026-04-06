Welcome to Fuse's documentation!
================================

Generate wordlists from concise expressions.

Fuse is a blazing fast and robust wordlist generator that parses character classes, quantifiers, files, and numeric ranges. It brings a "regex-like" paradigm to generating precise datasets, allowing offensive security professionals and developers to generate specific password lists, payloads, or permutations from a compact syntax.

.. toctree::
   :maxdepth: 2
   :caption: Installation & Setup:

   installation
   cli_reference

.. toctree::
   :maxdepth: 2
   :caption: Features & Syntax:

   features/expressions
   features/ranges_files
   features/advanced

.. toctree::
   :maxdepth: 2
   :caption: Links:

   GitHub Repository <https://github.com/pwnfo/fuse>

Why Fuse?
---------
* **Compact Patterns**: Describe millions of words in just a few characters.
* **Fast**: Employs mathematically optimal scaling algorithms yielding tens of thousands of words per second.
* **Smart Seeking**: Instead of streaming from the beginning, Fuse can seek directly to a specific word using ``--from`` and ``--to``, keeping memory usage constant and small.
* **Multi-threaded**: Distribute generation intelligently across up to 64 cores.
* **Scriptable**: Robust ``.fuse`` syntax allows breaking rules into easily readable macros and modules.
