# pip install pefile
import os
import pefile
import hashlib
import string
import subprocess
from statistics import mean


def extract_printable_strings(file_bytes, min_len=5):
    result = []
    current = ''
    for byte in file_bytes:
        try:
            char = chr(byte)
            if char in string.printable and char not in '\n\r\t':
                current += char
            else:
                if len(current) >= min_len:
                    result.append(current)
                current = ''
        except:
            continue
    if len(current) >= min_len:
        result.append(current)
    return result


def get_trid_info(filepath, timeout=30, trid_path=None):
    """
    Extract file type information using TRiD.
    
    Args:
        filepath: Path to the file
        timeout: Maximum execution time in seconds
        trid_path: Path to TRiD executable (optional)
    
    Returns:
        Tuple of (file_type, probability)
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return "Not found", 0.0
    
    # Use the provided TRiD path or just 'trid' if not specified
    trid_command = [trid_path] if trid_path else ['trid']
    if trid_path:
        trid_command = [trid_path]
    else:
        trid_command = ['trid']
    
    try:
        result = subprocess.run(
            trid_command + [filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        output = result.stdout.splitlines()
        for line in output:
            if "%" in line:
                parts = line.strip().split()
                prob = float(parts[0].replace('%', ''))
                file_type = " ".join(parts[1:])
                return file_type, prob
    except subprocess.TimeoutExpired:
        print(f"TRiD analysis timed out after {timeout} seconds")
    except FileNotFoundError:
        print("TRiD not found. Please install TRiD from https://mark0.net/soft-trid-e.html")
    except Exception as e:
        print(f"TRiD error: {e}")
    
    return "Unknown", 50.0


def extract_features_from_file(filepath):
    features = {}

    # Load file bytes first
    try:
        with open(filepath, 'rb') as f:
            file_bytes = f.read()
        features["sha256"] = hashlib.sha256(file_bytes).hexdigest()
        features["size"] = len(file_bytes)
    except:
        return {
            "vsize": None, "imports": 0, "exports": 0,
            "has_debug": 0, "has_tls": 0, "has_resources": 0,
            "has_relocations": 0, "has_signature": 0, "symbols": 0,
            "sha256": "", "size": 0, "numstrings": 0, "avlength": 0,
            "printables": 0, "paths": 0, "urls": 0, "registry": 0, "MZ": 0,
            "file_type_trid": "Unknown", "file_type_prob_trid": 50
        }

    # Try PE parsing
    try:
        pe = pefile.PE(data=file_bytes, fast_load=True)

        features["vsize"] = pe.OPTIONAL_HEADER.SizeOfImage
        try:
            features["imports"] = sum(len(entry.imports) for entry in pe.DIRECTORY_ENTRY_IMPORT)
        except:
            features["imports"] = 0

        try:
            features["exports"] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
        except:
            features["exports"] = 0

        features["has_debug"] = int(hasattr(pe, 'DIRECTORY_ENTRY_DEBUG'))
        features["has_tls"] = int(hasattr(pe, 'DIRECTORY_ENTRY_TLS'))
        features["has_resources"] = int(hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'))
        features["has_relocations"] = int(hasattr(pe, 'DIRECTORY_ENTRY_BASERELOC'))
        features["has_signature"] = int(pe.OPTIONAL_HEADER.DATA_DIRECTORY[4].Size > 0)
        features["symbols"] = 0  # Optional
    except:
        features.update({
            "vsize": None,
            "imports": 0,
            "exports": 0,
            "has_debug": 0,
            "has_tls": 0,
            "has_resources": 0,
            "has_relocations": 0,
            "has_signature": 0,
            "symbols": 0
        })

    # String-based features
    strings_found = extract_printable_strings(file_bytes)
    features["numstrings"] = len(strings_found)
    features["avlength"] = mean([len(s) for s in strings_found]) if strings_found else 0
    features["printables"] = sum(1 for b in file_bytes if chr(b) in string.printable)
    features["paths"] = sum(1 for s in strings_found if s.lower().startswith("c:\\"))
    features["urls"] = sum(1 for s in strings_found if "http://" in s.lower() or "https://" in s.lower())
    features["registry"] = sum(1 for s in strings_found if "HKEY_" in s)
    features["MZ"] = file_bytes.count(b"MZ")


    # File type (via TRiD CLI)
    # Specify the direct path to your TRiD executable
    trid_path = r"C:\Users\eitan\OneDrive\Desktop\Eitan\Setups\trid_w32\trid.exe"  # Update this path to where you installed TRiD
    file_type, prob = get_trid_info(filepath, trid_path=trid_path)
    features["file_type_trid"] = file_type
    features["file_type_prob_trid"] = prob

    return features


if __name__ == "__main__":
    # Example usage
    file_path = "demo-files/notepad_copy.exe"
    features = extract_features_from_file(file_path)
    from pprint import pprint
    pprint(features)
