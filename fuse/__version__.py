from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("fuse-tool")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
