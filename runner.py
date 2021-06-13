import well_model
import UniflocVBA.v7_25.python_api as python_api_7_25
import UniflocVBA.v7_28.python_api as python_api_7_28
import shutil
import time


path =  r'.' + '\\'

def esp_design_wrapper(arg_list):
    params, pumps_ids, pumps_heads, pi_mc, dist_p_res, debug, num_simulations, vba_version, thread_num = arg_list
    #time.sleep(thread_num)
    if vba_version == '7.25':
        path_to_vba = path + r'UniflocVBA\v7_25\UniflocVBA_7.xlam'

    else:
        path_to_vba = path + r'UniflocVBA\v7_28\UniflocVBA_7_28.xlam'

    new_path = path_to_vba.replace('.xlam', f"_{thread_num}.xlam")
    shutil.copy(path_to_vba, new_path)

    if vba_version == '7.25':
        print(f"Поток {thread_num} vba = 7.25")
        api = python_api_7_25.API(new_path)
        api_new = None
    else:
        print(f"Поток {thread_num} vba = 7.28")
        api = None
        api_new = python_api_7_28.API(new_path)

    results = well_model.run_design(params, pumps_ids, pumps_heads, pi_mc, dist_p_res,
                                    debug=debug, num_simulations=num_simulations, api=api,
                         api_new=api_new,
                         vba_version=vba_version, thread_number=thread_num, path=path + '\\calc_new')
    return results


