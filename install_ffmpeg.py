import shutil
import subprocess
import os
import stat
import urllib.request
import tarfile

def has_ffmpeg():
    return shutil.which("ffmpeg") is not None

if not has_ffmpeg():
    url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    local_tar = "ffmpeg.tar.xz"
    urllib.request.urlretrieve(url, local_tar)

    # extract
    import lzma
    import tarfile
    with lzma.open(local_tar) as f:
        with tarfile.open(fileobj=f) as tar:
            tar.extractall("ffmpeg_bin")

    # find ffmpeg inside extracted dir
    for root, dirs, files in os.walk("ffmpeg_bin"):
        if "ffmpeg" in files:
            ffmpeg_path = os.path.join(root, "ffmpeg")
            os.chmod(ffmpeg_path, os.stat(ffmpeg_path).st_mode | stat.S_IEXEC)
            # prepend to PATH for current process
            os.environ["PATH"] = f"{root}:{os.environ['PATH']}"
            break
