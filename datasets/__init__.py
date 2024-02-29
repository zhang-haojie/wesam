def call_load_dataset(cfg):
    name, type = cfg.dataset, cfg.load_type

    key = name.split("-")[0]
    module_name = f"datasets.{key}"

    if type == "load":
        function_name = "load_datasets"
    elif type == "soft":
        function_name = "load_datasets_soft"
    elif type == "visual":
        function_name = "load_datasets_visual"

    if cfg.prompt == "coarse":
        function_name = function_name + "_" + "coarse"

    exec(f"from {module_name} import {function_name}")
    func = eval(function_name)
    return func
