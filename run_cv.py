import importlib.util

if __name__ == "__main__":

    # load parameters from file
    params_file = sys.argv[1]
    spec = importlib.util.spec_from_file_location("module.name", params_file)
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    model_params = params.model_params[model_type]
    params = params.params

    