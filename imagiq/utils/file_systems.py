import os
import shutil
import pathlib
from zipfile import ZipFile
import tarfile


def path_split(filepath):
    """Splits a file path into all of its parts.

    For example, '/dir/subdir/a/b/c/file.txt' should be splitted into
    ['dir', 'subdir', 'a', 'b', 'c', 'file.txt']

    Arguments:
        filepath: File path to split.
    Returns:
        Splitted parts
    """
    parts = []
    while True:
        splitted = os.path.split(filepath)
        if splitted[0] == filepath:
            parts.insert(0, splitted[0])
            break
        elif splitted[1] == filepath:
            parts.insert(0, splitted[1])
            break
        else:
            filepath = splitted[0]
            parts.insert(0, splitted[1])
    return parts


def mkdir(directory):
    """Checks if a directory exists and makes if not.

    Arguments:
        directory: Path to the directory to be created.
    """
    # split the path into all of its pieces
    paths = path_split(directory)
    curr_dir = ""
    for path in paths:
        curr_dir = os.path.join(curr_dir, path)  # subdirectory
        if not os.path.exists(curr_dir):
            os.mkdir(curr_dir)


def copy(src, dst):
    """Copy a directory or a file.
    Arguments:
        src: Path to the directory/file to be copied from.
        dst: Destination.
    """
    if os.path.exists(src):
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        print("[File Systems]: " + src + " not found. Skipping it.")


def move(src, dst):
    """Move a directory or a file.
    Arguments:
        src: Path to the directory/file to be moved from.
        dst: Destination.
    """
    if os.path.exists(src):
        shutil.move(src, dst)
    else:
        print("[File Systems]: " + src + " not found. Skipping it.")


def rename(src, dst):
    """Rename a directory or a file.
    Arguments:
        src: Path to the directory/file to be renamed.
        dst: New name.
    """
    if os.path.exists(src):
        os.rename(src, dst)
    else:
        print("[File Systems]: " + src + " not found. Skipping it.")


def extract_archive(archive_path, extract_to=None, cleanup=False):
    """Extracts an archive file.

    Arguments:
        archive_path: Path to the archive file.
        extract_to: Directory path where the archive file is to be extracted.
        cleanup: If True, the archive file will be removed after extracting it.
    Returns:
        Path to the downloaded file/folder
    """
    extension = pathlib.Path(archive_path).suffixes
    if not extension:
        raise ValueError("Invalid file format.")
    elif extension[0] == ".zip":
        with ZipFile(archive_path, "r") as zip:
            if extract_to is None:
                zip.extractall()
            else:
                zip.extractall(extract_to)
        if cleanup is True:
            os.remove(archive_path)
    elif extension[0] == ".tar":
        with tarfile.open(archive_path) as tar:
            if extract_to is None:
                tar.extractall()
            else:
                tar.extractall(extract_to)
    else:
        raise ValueError("Invalid file format.")


def remove(path):
    """Remove a file or a directory (and all of its contents).
    Arguments:
        path: Path to the directory/file to remove.
    """
    if not os.path.exists(path):
        print("[File Systems]: " + path + " not found. Skipping it.")
        return
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)
