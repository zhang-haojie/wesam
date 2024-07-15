def call_load_dataset(cfg):
    name = cfg.dataset

    key = name.split("-")[0]
    module_name = f"datasets.{key}"
    function_name = "load_datasets"

    if cfg.visual:
        function_name = function_name + "_" + "visual"

    if cfg.prompt == "coarse":
        function_name = function_name + "_" + "coarse"

    exec(f"from {module_name} import {function_name}")
    func = eval(function_name)
    return func


def call_load_dataset_prompt(cfg):
    name = cfg.dataset

    key = name.split("-")[0]
    module_name = f"datasets.{key}"
    function_name = "load_datasets"

    function_name = function_name + "_" + "prompt"

    exec(f"from {module_name} import {function_name}")
    func = eval(function_name)
    return func


def call_load_dataset_val(cfg):
    name = cfg.dataset

    key = name.split("-")[0]
    module_name = f"datasets.{key}"
    function_name = "load_datasets"

    function_name = function_name + "_" + "val"

    exec(f"from {module_name} import {function_name}")
    func = eval(function_name)
    return func
