def get_dataset_path(env_name, mode, debug):
    key = env_name + '_' + mode
    if debug:
        key = key + '_debug'
    d = {
        'MultiTool-v1_local_debug': 'data/autobot/1024_gen_multitool/',
        'MultiTool-v1_local': 'data/autobot/1024_gen_multitool/',

        'Roll-v1_local_debug': 'data/autobot/1024_gen_multitool/',
        'Roll-v1_local': 'data/autobot/1024_gen_multitool/',

        'Cut-v1_local_debug': 'data/autobot/1024_gen_multitool/',
        'Cut-v1_local': 'data/autobot/1024_gen_multitool/',

        'SmallScoop-v1_local_debug': 'data/autobot/1024_gen_multitool/',
        'SmallScoop-v1_local': 'data/autobot/1024_gen_multitool/',

        'LargeScoop-v1_local_debug': 'data/autobot/1024_gen_multitool/',
        'LargeScoop-v1_local': 'data/autobot/1024_gen_multitool/',

    }
    return d[key]


def get_bc_policy_path(actor_type, mode):
    key = actor_type + '_' + mode
    d = {
        'conditioned_local': '<BC_POLICY_PATH>',
    }
    return d[key]