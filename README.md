# Fuse — Wordlist Generator

Generate wordlists from concise expressions. Fuse parses character classes, quantifiers, file inputs and combines them into wordlists suitable for password lists, testing, or data generation.

---

## Table of Contents

* [Quick start](#quick-start)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)

  * [Expression basics](#expression-basics)
  * [Character classes](#character-classes)
  * [Custom classes and unions](#custom-classes-and-unions)
  * [Quantifiers](#quantifiers)
  * [Files and placeholders](#files-and-placeholders)
  * [Escaping special characters](#escaping-special-characters)
* [Examples](#examples)
* [License](#license)

---

## Quick start

Run `fuse` with an expression to generate words. Expressions combine built-in character classes, custom classes, quantifiers and file placeholders.

```bash
$ fuse "/l{2,3}"
# sample output: aa, ab, ac, ..., Zy, Zz, ZA, ZB, ..., ZZ
```

---

## Features

* Built-in character classes (letters, digits, hex, etc.)
* Custom character classes (single characters or word alternatives)
* Quantifiers (`{n}`, `{min,max}`, `?`) for repetition and optional tokens
* File placeholders (`^`) to inject external wordlists
* Escaping for literal special characters

---

## Installation

Install Fuse directly from PyPI or GitHub:

### From PyPI (recommended)
```
pip install fuse-cli
```

### From GitHub
```
git clone https://github.com/pwnfo/fuse.git
cd fuse
pip install .
```
---

## Usage

```
fuse [options] <expression> [<files...>]
```

`<expression>` is a compact string that specifies how words are generated. `[<files...>]` can contain filenames or inline expressions (see below).

### Expression basics

* Literal characters produce themselves.
* Built-in classes and bracketed classes produce one item per position.
* Concatenation forms full words: each position picks one value from its class or token.

### Character classes

Common built-in classes (examples):

```
/l  - letters (a–z, A–Z)
/a  - lowercase letters (a–z)
/A  - uppercase letters (A–Z)
/d  - digits (0–9)
/h  - hexadecimal (0–9, a–f)
/s  - space
/o  - octal digits (0-7)
/p  - special characters
/N  - break line (\n)
```

Use these directly in expressions, for example: `/l/l` generates two-letter combinations where each letter may be lower or upper case.

**Sample**

```
$ fuse "/l/l"
# sample output: aa, ab, ac, ..., ZY, ZZ
```

### Custom classes and unions

You can create classes with bracket syntax `[...]`. Inside brackets you may list characters or use `|` to separate full-word alternatives.

```
[hello]     # characters: h, e, l, l, o
[admin|pass|12345678]  # whole-word alternatives
```

**Notes**

* `[abc]` selects one character from the set `a`, `b` or `c`.
* `[admin|root]` will treat `admin` and `root` as alternatives (each alternative is inserted as a multi-character token).

### Quantifiers

* `{N}` — repeat exactly `N` times
* `{min,max}` — repeat between `min` and `max` times (inclusive)
* `?` — previous token is optional (zero or one)

Examples:

```
$ fuse "[XYZ]{3}"
# sample: XXX, XXY, ..., ZZZ

$ fuse "[XYZ]{2,5}"
# sample: XY, XZ, ..., XYZXY

$ fuse "Ryan?/d"
# sample: Rya0, Rya1, ..., Ryan9

$ fuse "[XYZ]?Ryan"
# sample: Ryan, XRyan ... ZRyan
```

### Files and placeholders

Use `^` in an expression as a placeholder for filenames supplied after the expression. Each `^` consumes the next file argument and iterates its lines where the placeholder appears.

```
$ fuse "^/d" names.txt
# sample: Bob0, Bob1, ..., Ana0, Ana1, ...

$ fuse "^-^" names.txt years.txt
# sample: Bob-1990, Ana-1991, Ryan-1992, ...
```

You may also supply inline expressions in place of filenames by prefixing them with `//`. This treats the following token as an expression source instead of a path.

```
$ fuse "Ryan^" "//_/d"
# sample: Ryan_0, Ryan_1, ..., Ryan_9
```

### Escaping special characters

Use a backslash `\` to escape special characters when you need them literally.

```
$ fuse "\/d/d"
# sample: /d/0, /d/1, ..., /d/9
```

---

## Examples

```
# letter words length 2 to 4
$ fuse "/l{2,4}"
# sample: aa, ab, ..., ZZZZ

# 3-byte words mixing letters and digits
$ fuse "[/l/d]{3}"
# sample: 0A0, 0A1, ..., Z9Z

# joining two files with a separator
$ fuse "^:^" names.txt pass.txt
# sample: alice:admin, bob:8888, ...
```

> *Use `fuse --help` to get more CLI options*

---

## License

Licensed under the MIT License, see [LICENSE](./LICENSE) for more information.