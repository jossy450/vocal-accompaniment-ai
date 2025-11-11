def generate_accompaniment_with_fallback(vocal_path, style, complexity, instruments, features, output_dir):
    # TODO: connect to real model services
    import shutil
    band_path = vocal_path.replace("vocal", "band")
    shutil.copy(vocal_path, band_path)  # placeholder copy
    return band_path
