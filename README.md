<p align="center">
  <br>
  <a href="https://github.com/pwnfo/fuse" target="_blank"><img src="images/icon.png" width="30%" alt="fuse"/></a>
  <br>
  <span>Pattern-based wordlist generation tool</span>
  <br>
</p>

<p align="center">
  <a href="#installation">Installation</a>
  &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp;
  <a href="#usage">Usage</a>
  &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp;
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
<img width="80%" src="images/demo.png" alt="demo"/>
</p>

## Installation

> [!NOTE]  
> It is **recommended** to install using `pipx` or `pip` for the PyPI version.

> [!NOTE]  
> The PyPI package has been renamed from `fuse-tool` to `fuse-generator`.

| Method | Notes |
| - | - |
| `pipx install fuse-generator` | `pip` may be used in place of `pipx` |
| `git clone https://github.com/pwnfo/fuse.git && cd fuse && pip install .` | Clone and install directly from GitHub |


## General usage

To generate a wordlist from a simple expression:
```bash
fuse "/l{2,4}"
```

To combine files with generators:
```bash
fuse "^:^" names.txt pass.txt
```

Outputs can be manipulated, filtered, and saved.

```console
General Options:
  -h, --help            show this help message and exit
  -v, --version         show version message and exit
  -o, --output <path>   write the wordlist in the file
  -f, --file <path>     file with different expressions
  -q, --quiet           use quiet mode
  -n, --non-interactive
                        disable interactive prompt before execution

Generation Options:
  -d, --delimiter <string>
                        delimiter between entries
  -b, --write-buffer <size>
                        write buffer size
  -w, --workers <1-64>  number of workers (default is 1)
  -F, --filter <regex>  filter generated words using a regex
  -k, --flush-threshold <size>
                        byte threshold before flushing output (default is 512KB)
  -z, --compress <format>
                        compress output (available: gzip, bzip2 and lzma)
  -l, --compresslevel <level>
                        compression level (depends on selected format)
  -S, --start <word>    start writing the wordlist from <word>
  -E, --end <word>      end writing the wordlist at <word>
```

### Expression basics

* Literal characters produce themselves.
* Built-in classes and bracketed classes `[...]` produce one item per position.
* Concatenation combines positions: each position picks one value from its token and concatenates.

Example:
```bash
$ fuse "/l{2,3}"
# output: aa, ab, ac, ..., ZY, ZZ
```

### Character classes

| Symbol | Meaning                          |
| ------ | -------------------------------- |
| `/l`   | letters (a–z, A–Z)               |
| `/a`   | lowercase letters (a–z)          |
| `/A`   | uppercase letters (A–Z)          |
| `/d`   | digits (0–9)                     |
| `/D`   | non-zero digits (1–9)            |
| `/h`   | lowercase hexadecimal (0–9, a–f) |
| `/H`   | uppercase hexadecimal (0–9, A–F) |
| `/s`   | space                            |
| `/o`   | octal digits (0–7)               |
| `/p`   | special characters               |
| `/N`   | newline (`\n`)                   |

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
```bash
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
```bash
$ fuse "^/d" names.txt
# output: Bob0, Bob1, ..., Ana0, Ana1, ...

$ fuse "^-^" names.txt years.txt
# output: Bob-1990, Ana-1991, Ryan-1992, ...
```

Prefix a filename with `//` to treat it as an inline expression instead of a file path.

### Value bindings

Use `<@name=expr>` to evaluate an expression **once per output line** and store it, then reuse it with `<@name>`.
Unlike concatenation, this introduces a dependency, so no cartesian product is created between the definition and its references.

```bash
$ fuse '<@d=/d>-<@d>'
# output: 0-0, 1-1, ..., 9-9

$ fuse '<@x=^>:<@x>' words.txt
# output: foo:foo, bar:bar, ...

$ fuse '<@n=/d{2}>_<@n>'
# output: 00_00, 01_01, ..., 99_99
```

### Compression

Fuse supports on-the-fly compression when writing output files.

```bash
# gzip (fast, balanced)
fuse "/l{5}" -z gzip -o wordlist.txt.gz

# lzma (best compression)
fuse "/l{5}" -z lzma -o wordlist.txt.xz

# bzip2 (middle ground)
fuse "/l{5}" -z bzip2 -o wordlist.txt.bz2
```

### Escaping special characters

Use `\` to escape special characters.
```bash
$ fuse "\/d/d"
# output: /d/0, /d/1, ..., /d/9
```

## Contributing

We welcome contributions to Fuse! Whether it's adding new features, improving documentation, or fixing bugs, your help is appreciated. 
Feel free to open an issue or submit a pull request on our GitHub repository at `pwnfo/fuse`.

## Star History

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=pwnfo/fuse&type=Date&theme=dark" />
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=pwnfo/fuse&type=Date" />
  <img alt="Fuse Project Star History Chart" src="https://api.star-history.com/svg?repos=pwnfo/fuse&type=Date" />
</picture>

## License

MIT © Ryan R. &lt;pwnfo@proton.me&gt;
