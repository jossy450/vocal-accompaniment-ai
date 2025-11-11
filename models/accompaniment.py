def generate_accompaniment_with_fallback(vocal_path, style, complexity, instruments, features, output_dir):
    # TODO: call real model here
    import shutil
    band_path = vocal_path.replace("v", "b").replace("_", "_band_")
    shutil.copy(vocal_path, band_path)
    return band_path
