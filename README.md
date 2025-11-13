<p align="center">
  <span>Generate wordlists from concise expressions — a compact "regex" for wordlists</span>
  <br>
</p>

<p align="center">
  <a href="#installation">Installation</a>
  &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp;
  <a href="#usage">Usage</a>
  &nbsp;&nbsp;&nbsp;
</p>

<p align="center">
<img src="images/demo.png" alt="demo"/>
</p>

## Description

**Fuse** - Wordlist Generator

A tool to generate wordlists from compact expressions. Fuse parses character classes, quantifiers, file placeholders, and combines them into outputs useful for password lists, testing, or data generation.

Core features:

* Short expressions describing large word sets
* Support for built-in and custom character classes
* Placeholders for files and inline expressions
* Quantifiers, numeric ranges, and escaping special characters

---

## Installation

> **Recommended**: install using `pipx` or `pip` for the PyPI version.

| Method                           | Notes                                                                     |
| -------------------------------- | ------------------------------------------------------------------------- |
| `pipx install fuse-cli`          | `pip` can be used instead of `pipx`                                       |
| Clone and install from GitHub    | `git clone https://github.com/pwnfo/fuse.git && cd fuse && pip install .` |

---

## Usage

```
usage: fuse [options] <expression> [<files...>]

Fuse v0.1.0

  -h, --help            show this help message and exit
  -v, --version         show version message and exit
  -o, --output <path>   write the wordlist in the file
  -f, --file <path>     files with different expressions
  -q, --quiet           use quiet mode
  -b, --buffer <bytes>  buffer size in wordlist generation
  -s, --separator <sep>
                        separator beetwen entries
  --from <word>         start writing the wordlist with <word>
  --to <word>           ends writing the wordlist with <word>

Customizable wordlist generator with advanced pattern.
Developed by Ryan R. <pwnfo@proton.me>
```

### Expression basics

* Literal characters produce themselves.
* Built-in classes and bracketed classes `[...]` produce one item per position.
* Concatenation combines positions: each position picks one value from its token and concatenates.

Example:

```
$ fuse "/l{2,3}"
# output: aa, ab, ac, ..., ZY, ZZ
```

### Character classes

| Symbol | Meaning                 |
| ------ | ----------------------- |
| `/l`   | letters (a–z, A–Z)      |
| `/a`   | lowercase letters (a–z) |
| `/A`   | uppercase letters (A–Z) |
| `/d`   | digits (0–9)            |
| `/h`   | hexadecimal (0–9, a–f)  |
| `/s`   | space                   |
| `/o`   | octal digits (0–7)      |
| `/p`   | special characters      |
| `/N`   | newline (`\n`)          |

Example: `/l/l` generates all two-letter combinations (upper and lower case).

### Custom classes and unions

* `[abc]` selects **one character** from `a`, `b`, or `c`.
* Use `|` to separate full-word alternatives (each treated as a multi-character token):

  * `[admin|root|123]` inserts `admin` OR `root` OR `123` at that point.

### Quantifiers

* `{N}` — repeat exactly N times
* `{min,max}` — repeat between min and max times (inclusive)
* `?` — optional (0 or 1 time)

Examples:

```
$ fuse "[XYZ]{3}"         # XXX, XXY, ..., ZZZ
$ fuse "[XYZ]{2,5}"       # XY, XZ, ..., XYZXY
$ fuse "Ryan?/d"          # Rya0, Rya1, ..., Ryan9
$ fuse "[XYZ]?Ryan"       # Ryan, XRyan, YRyan, ZRyan
```

### Numeric ranges

* `#[1-10]` → generates 1,2,3,4,5,6,7,8,9,10
* `#[1-10:2]` → generates 1,3,5,7,9
* `#[2-10:2]` → generates 2,4,6,8,10

These numeric ranges can be used in any position of an expression.

### Files and placeholders

Use `^` in an expression as a placeholder for the next file argument. Each `^` consumes one file and iterates over its lines:

```
$ fuse "^/d" names.txt
# output: Bob0, Bob1, ..., Ana0, Ana1, ...

$ fuse "^-^" names.txt years.txt
# output: Bob-1990, Ana-1991, Ryan-1992, ...
```

Prefix a filename with `//` to treat it as an inline expression instead of a file path.

### Escaping special characters

Use `\` to escape special characters.

```
$ fuse "\\/d/d"
# output: /d/0, /d/1, ..., /d/9
```

---

## Examples

```
$ fuse "/l{2,4}"                # words with 2 to 4 letters
$ fuse "[/l/d]{3}"              # 3-char words mixing letters and digits
$ fuse "^:^" names.txt pass.txt  # combine two files with separator
$ fuse "user#[1-5]"             # user1, user2, user3, user4, user5
$ fuse "id#[0-10:2]"            # id0, id2, id4, id6, id8, id10
```

Use `-o output.txt` to save the output to a file.

---

## Credits

Developed by <a href="https://github.com/pwnfo">**Ryan R.**</a>

---

## License

MIT © Ryan R.

---
