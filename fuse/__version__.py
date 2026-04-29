from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("fuse-generator")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
