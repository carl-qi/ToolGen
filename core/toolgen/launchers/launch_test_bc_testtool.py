import time
import click
from core.toolgen.test_traj_bc import run_task
from core.diffskill.dataset_path import get_bc_policy_path, get_dataset_path

env_name_to_mode = {
    'Roll-v1': 'roll',
    'Cut-v1': 'cut',
    'SmallScoop-v1': 'small_scoop',
    'LargeScoop-v1': 'large_scoop'
}

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '<EXP_PREFIX>'
    from chester.run_exp import run_experiment_lite, VariantGenerator
    vg = VariantGenerator()
    vg.add('env_name', ['Roll-v1', 'Cut-v1', 'SmallScoop-v1', 'LargeScoop-v1'])
    vg.add('cached_state_path', ['datasets/1025_multitool'])
    vg.add('actor_type', ['conditioned'])
    vg.add('resume_path', lambda actor_type: [get_bc_policy_path(actor_type, mode)])
    vg.add('dataset_path', lambda env_name: [get_dataset_path(env_name, mode, debug)])
    vg.add('pointflow_resume_path', ['<POINTFLOW_RESUME_PATH>'])
    vg.add('fit_training_tool', [False])
    vg.add('num_tools', [3])
    vg.add('train_tool_idxes', [[0, 1, 2]])
    vg.add('use_wandb', [True] if not debug else [False])
    vg.add('batch_size', [5])  # Same as the number of positive pairs
    vg.add('il_eval_freq', [1])
    vg.add('use_contact', [False])
    vg.add('il_num_epoch', [0])
    vg.add('env_loss_type', ['chamfer']) #, 'emd'])
    vg.add('input_mode', ['pc'])
    vg.add('img_size', [128])
    vg.add('wandb_group', [exp_prefix])
    vg.add('debug', [debug])
    vg.add('opt_reset_pose', [True])
    vg.add('opt_traj_deltas', [True])
    vg.add('eval_traj_mode', lambda env_name : [env_name_to_mode[env_name]])
    vg.add('eval_subset', [False])
    vg.add('use_gt_tool', [True])

    if debug:
        exp_prefix += '_debug'
    print('Number of configurations: ', len(vg.variants()))

    sub_process_popens = []
    for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 1:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        compile_script = wait_compile = None

        cur_popen = run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break


if __name__ == '__main__':
    main()
