import platform
import sys

def is_linux_kernel_6_11():
    if sys.platform != "linux":
        return False

    kernel_version = platform.release()  # e.g., '6.11.0-24-generic'
    version_parts = kernel_version.split(".")
    try:
        major = int(version_parts[0])
        minor = int(version_parts[1])
        return major == 6 and minor == 11
    except (IndexError, ValueError):
        return False