def get_eval_traj(cached_state_path, plan_step=3, exclude_idxes=[], eval_subset=False, mode='all'):
    """Test set"""
    if 'multitool' in cached_state_path:
        if eval_subset:
            init_v = [0, 611, 814, 1006]
        else:
            if mode == 'all':
                init_v = [0,    13,   14,   20,   40,   53,   59,   83,   106,  110,
                        611,  619,  631,  661,  665,  666,  702,  705,  718,  721,
                        814,  816,  819,  834,  859,  854,  875,  876,  880,  884,
                        1006, 1022, 1025, 1028, 1059, 1034, 1035, 1038, 1050, 1053]
            elif mode == 'roll':
                init_v = [0,    13,   14,   20,   40,   53,   59,   83,   106,  110,
                          0,    13,   14,   20,   40,   53,   59,   83,   106,  110,
                          0,    13,   14,   20,   40,   53,   59,   83,   106,  110]
            elif mode == 'cut':
                init_v = [611,  619,  631,  661,  665,  666,  702,  705,  718,  721,
                          611,  619,  631,  661,  665,  666,  702,  705,  718,  721,
                          611,  619,  631,  661,  665,  666,  702,  705,  718,  721]
            elif mode == 'small_scoop':
                init_v = [814,  816,  819,  834,  859,  854,  875,  876,  880,  884,
                          814,  816,  819,  834,  859,  854,  875,  876,  880,  884,
                          814,  816,  819,  834,  859,  854,  875,  876,  880,  884]
                # init_v = [814]
            elif mode == 'large_scoop':
                init_v = [1006, 1022, 1025, 1028, 1059, 1034, 1035, 1038, 1050, 1053,
                          1006, 1022, 1025, 1028, 1059, 1034, 1035, 1038, 1050, 1053,
                          1006, 1022, 1025, 1028, 1059, 1034, 1035, 1038, 1050, 1053]
            else:
                raise NotImplementedError
        target_v = init_v.copy()
        init_v = [v for v in init_v if v not in exclude_idxes]
        target_v = [v for v in target_v if v not in exclude_idxes]
        return init_v, target_v
    else:
        raise NotImplementedError

def get_eval_skill_trajs(cached_state_path, tid):
    if '1215_cutrearrange' in cached_state_path:
        if tid == 0:
            init_v = [0, 3, 6, 9, 12]
            target_v = [0, 3, 6, 9, 12]
        else:
            init_v = [1, 2, 4, 8, 10]
            target_v = [1, 2, 4, 8, 10]
        return init_v, target_v
    elif '0202_liftspread' in cached_state_path:
        if tid == 0:
            init_v = [27, 35, 38, 46, 48]
            target_v = [109, 112, 143, 166, 185]
        else:
            init_v = [104, 104, 106, 106, 107]
            target_v = [27, 35, 38, 46, 48]
        return init_v, target_v
    elif '0202_gathermove' in cached_state_path:
        if tid == 0:
            init_v = [0, 2, 4, 8, 16]
            target_v = [1, 3, 5, 9, 17]
        else:
            init_v = [1, 3, 5, 9, 17]
            target_v = [102, 103, 102, 103, 104]
        return init_v, target_v
    elif 'cutrearrangespread' in cached_state_path:
        if tid == 0:
            init_v = [0, 1, 2, 3, 4]
            target_v = [0, 1, 2, 3, 4]
        elif tid == 1:
            init_v = [200, 201, 202, 203, 204]
            target_v = [200, 201, 202, 203, 204]
        else:
            init_v = [400, 401, 402, 403, 404]
            target_v = [400, 401, 402, 403, 404]
        return init_v, target_v
    else:
        raise NotImplementedError