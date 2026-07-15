import sys


def _explicit_cli_keys():
    keys = set()
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            continue
        key = arg[2:].split("=", 1)[0].replace("-", "_")
        if key:
            keys.add(key)
    return keys


def merge_hparams(args, config):
    params = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
    explicit_keys = _explicit_cli_keys()
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if hasattr(args, key) and key not in explicit_keys:
                    setattr(args, key, value)

    return args
