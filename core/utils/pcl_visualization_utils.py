import socket

from core.diffskill.utils import batch_pred

if 'autobot' not in socket.gethostname() and 'seuss' not in socket.gethostname() and 'compute' not in socket.gethostname():
    import open3d as o3d

import matplotlib
import matplotlib.cm
import numpy as np
from plotly.subplots import make_subplots
import plotly.colors as pc
import plotly.graph_objects as go
import torch
cmap = matplotlib.cm.get_cmap('Spectral')
# Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
cam_center = [0.5, 0.3, 0.6]  # look_at target
cam_eye = [0.5, 0.6, 1.0]  # camera position
cam_up = [0, -0.5, 0]  # camera orientation


def draw_unit_box():
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[0, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def draw_frame():
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])


def set_camera(vis, camera_path):
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters(camera_path)
    ctr.convert_from_pinhole_camera_parameters(parameters)


def visualize_point_cloud(pcls):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(draw_frame())
    vis.add_geometry(draw_unit_box())
    # if len(pcls.shape) == 2:
    #     pcls = np.expand_dims(pcls, 0)
    pcds = []
    for idx, pcl in enumerate(pcls):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        color = cmap(idx / len(pcls))[:3]
        pcd.colors = o3d.utility.Vector3dVector(np.ones_like(pcl) * color)
        pcds.append(pcd)
        vis.add_geometry(pcd)
    set_camera(vis, 'core/utils/camera_info.json')
    vis.run()


def visualize_point_cloud_plt(pcl, view_init=(140, -90)):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    matplotlib.rcParams['figure.dpi'] = 100
    fig = plt.figure(figsize=(3, 3))
    canvas = FigureCanvas(fig)
    cmap = plt.get_cmap("tab10")
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], s=5, color=cmap(0))
    # ax1.view_init(elev=130, azim=270)
    #
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    ax1.view_init(*view_init)
    # ax1.set_xlim3d(0.2, 0.8)
    # ax1.set_ylim3d(0, 0.6)
    # ax1.set_zlim3d(0.2, 0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.tight_layout()
    canvas.draw()  # draw the canvas, cache the renderer
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.close()
    return image


def visualize_point_cloud_batch(lst_pcl, dpi=100, view_init=(140, -90)):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    matplotlib.rcParams['figure.dpi'] = dpi
    fig = plt.figure(figsize=(3, 3))
    canvas = FigureCanvas(fig)

    images = []
    width, height = fig.get_size_inches() * fig.get_dpi()
    width = int(width)
    height = int(height)
    cmap = plt.get_cmap("tab10")
    ax1 = fig.add_subplot(111, projection='3d')

    # ax1.view_init(elev=130, azim=270)
    #
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    ax1.view_init(*(view_init))
    # ax1.set_xlim3d(0.2, 0.8)
    # ax1.set_ylim3d(0, 0.6)
    # ax1.set_zlim3d(0.2, 0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.tight_layout()

    for pcl in lst_pcl:
        q = ax1.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], s=5, color=cmap(0))
        canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        images.append(image)
        q.remove()
    plt.close()
    return images

def visualize_pcl_policy_input(pcl, tool_pcl, goal_pcl, view_init=(140, -90)):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    matplotlib.rcParams['figure.dpi'] = 100
    fig = plt.figure(figsize=(3, 3))
    canvas = FigureCanvas(fig)
    cmap = plt.get_cmap("tab10")
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(pcl[:, 0], pcl[:, 1], pcl[:, 2], s=5, color=cmap(0))
    if tool_pcl is not None:
        ax1.scatter(tool_pcl[:, 0], tool_pcl[:, 1], tool_pcl[:, 2], s=5, color=cmap(1))
    ax1.scatter(goal_pcl[:, 0], goal_pcl[:, 1], goal_pcl[:, 2], s=5, color=cmap(2))
    # ax1.view_init(elev=130, azim=270)
    #
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_zlim(0, 1)
    ax1.view_init(*view_init)
    # ax1.set_xlim3d(0.2, 0.8)
    # ax1.set_ylim3d(0, 0.6)
    # ax1.set_zlim3d(0.2, 0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.tight_layout()
    canvas.draw()  # draw the canvas, cache the renderer
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.close()
    return image

def _adjust_camera(f):

    _adjust_camera_angle(f)


def _adjust_camera_angle(f):
    """Adjust default camera angle if desired.
    For default settings: https://plotly.com/python/3d-camera-controls/
    """
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0., y=0., z=0.),
        eye=dict(x=1., y=-1., z=1.)
    )
    f.update_layout(scene_camera=camera)

def _3d_scene(data):
    # Create a 3D scene which is a cube w/ equal aspect ratio and fits all the data.
    assert data.shape[1] == 3
    # Find the ranges for visualizing.
    mean = data.mean(axis=0)
    max_x = np.abs(data[:, 0] - mean[0]).max()
    max_y = np.abs(data[:, 1] - mean[1]).max()
    max_z = np.abs(data[:, 2] - mean[2]).max()
    all_max = max(max(max_x, max_y), max_z)
    scene = dict(
        xaxis=dict(nticks=10, range=[mean[0] - all_max, mean[0] + all_max]),
        yaxis=dict(nticks=10, range=[mean[1] - all_max, mean[1] + all_max]),
        zaxis=dict(nticks=10, range=[mean[2] - all_max, mean[2] + all_max]),
        aspectratio=dict(x=1, y=1, z=1),
    )
    return scene

def _segmentation_traces(
    data,
    labels,
    labelmap=None,
    scene="scene",
    sizes=None,
):
    # Colormap.
    colors = np.array(pc.qualitative.Alphabet)
    # Keep track of all the traces.
    traces = []
    for label in np.unique(labels):
        subset = data[np.where(labels == label)]
        color = colors[label % len(colors)]
        if sizes is None:
            subset_sizes = 4
            if subset.shape[0] == 1:
                subset_sizes = 10
        else:
            subset_sizes = sizes[np.where(labels == label)]
        if labelmap is not None:
            legend = labelmap[label]
        else:
            legend = str(label)
        # print(subset.shape)
        traces.append(
            go.Scatter3d(
                mode="markers",
                marker={"size": subset_sizes, "color": color, "line": {"width": 0}},
                x=subset[:, 0],
                y=subset[:, 1],
                z=subset[:, 2],
                name=legend,
                scene=scene,
            )
        )
    return traces

def plot_scene_with_contact(action_pos, anchor_pos, contact):
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "scene"}],
        ],
        subplot_titles=(
            "Alignment",
            "Action weight",
        ),
    )
    # Segmentation.
    pos = torch.cat([action_pos, anchor_pos, contact], dim=0)
    labels = torch.zeros(len(pos)).int()
    labels[: len(action_pos)] = 0
    labels[len(action_pos) : len(action_pos) + len(anchor_pos)] = 1
    labels[len(action_pos) + len(anchor_pos) :] = 2
    labelmap = {0: "tool", 1: "dough", 2: "Pred"}
    fig.add_traces(_segmentation_traces(pos, labels, labelmap, "scene1"))
    fig.update_layout(
        scene1=_3d_scene(pos),
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(xanchor="left", x=0, yanchor="top", y=0.75),
        title="",
    )
    return fig


def plot_scene_with_frames(quaternions):
    import pytorch3d.transforms as transforms
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "scene"}],
        ],
    )
    u, v, w = torch.FloatTensor([0.01, 0, 0]), torch.FloatTensor([0, 0.01, 0]), torch.FloatTensor([0, 0, 0.01])
    u = transforms.quaternion_apply(quaternions[:, 3:7], u)
    v = transforms.quaternion_apply(quaternions[:, 3:7], v)
    w = transforms.quaternion_apply(quaternions[:, 3:7], w)
    # u += quaternions[:, :3]
    # v += quaternions[:, :3]
    # w += quaternions[:, :3]
    fig.add_traces(go.Cone(
    x=quaternions[:, 0],
    y=quaternions[:, 1],
    z=quaternions[:, 2],
    # x=torch.zeros((len(quaternions))),
    # y=torch.zeros((len(quaternions))),
    # z=torch.zeros((len(quaternions))),
    u=u[:, 0],
    v=u[:, 1],
    w=u[:, 2],
    sizemode="absolute",
    sizeref=0.1))

    # fig.add_traces(go.Cone(
    # x=quaternions[:, 0],
    # y=quaternions[:, 1],
    # z=quaternions[:, 2],
    # # x=torch.zeros((len(quaternions))),
    # # y=torch.zeros((len(quaternions))),
    # # z=torch.zeros((len(quaternions))),
    # u=v[:, 0],
    # v=v[:, 1],
    # w=v[:, 2],
    # sizemode="absolute",
    # sizeref=1))
    # fig.update_layout(
    #     scene1=_3d_scene(pos),
    #     showlegend=True,
    #     margin=dict(l=0, r=0, b=0, t=40),
    #     legend=dict(xanchor="left", x=0, yanchor="top", y=0.75),
    #     title="",
    # )
    return fig

def visualize_model_output(tool_particles, obs_pc, goal_pc,
                           model_head,
                           images,
                           rotation_matrices_gt=None,
                           final_prediction=None,):
    """Display distributions over SO(3).
    Args:
        vision_model: The model which produces a feature vector to hand to IPDF.
        model_head: The IPDF model.
        images: A list of images.
        rotation_matrices_gt: A list of [N, 3, 3] tensors, representing the ground
        truth rotation matrices corresponding to the images.
    Returns:
        A tensor of images to output via Tensorboard.
    """
    import tensorflow as tf
    import tensorflow_graphics.geometry.transformation as tfg
    import matplotlib.pyplot as plt
    return_images = []
    num_to_display = 1

    query_rotations = model_head.get_closest_available_grid(
        model_head.number_eval_queries)
    probabilities = []
    for image in images:
        import pytorch3d.transforms as transforms
        poses = transforms.matrix_to_quaternion(torch.FloatTensor(query_rotations)).cuda()
        poses = torch.cat([torch.zeros((len(poses), 3)).cuda(), poses], dim=1)
        probs = batch_pred(model_head.get_r_score, {'poses': poses, 'tool_pc': [tool_particles], 'obs_pc': [obs_pc], 'goal_pc':[goal_pc]}, batch_size=5)
        probs = torch.nn.Softmax(dim=0)(probs).detach().cpu().numpy()
        probabilities.append(probs)
    probabilities = np.array(probabilities)

    inches_per_subplot = 4
    # canonical_rotation = np.float32(tfg.rotation_matrix_3d.from_euler([0.0]*3))
    # canonical_rotation = tfg.rotation_matrix_3d.inverse(rotation_matrices_gt)[0]
    # canonical_rotation = rotation_matrices_gt[0]
    canonical_rotation = np.eye(3)
    for image_index in range(num_to_display):
        print(image_index)
        fig = plt.figure(figsize=(3*inches_per_subplot, inches_per_subplot),
                        dpi=100)
        gs = fig.add_gridspec(1, 3)
        fig.add_subplot(gs[0, 0])
        plt.imshow(images[image_index])
        plt.axis('off')
        ax2 = fig.add_subplot(gs[0, 1:], projection='mollweide')
        return_fig = visualize_so3_probabilities(
            query_rotations,
            probabilities[image_index],
            rotation_matrices_gt[image_index],
            ax=ax2,
            fig=fig,
            display_threshold_probability=1e-2 / query_rotations.shape[0],
            canonical_rotation=canonical_rotation,
            final_prediction=final_prediction[image_index] if final_prediction is not None else None)
        return_images.append(return_fig)
    return tf.concat(return_images, 0)

def visualize_so3_probabilities(rotations,
                                probabilities,
                                rotations_gt=None,
                                ax=None,
                                fig=None,
                                display_threshold_probability=0,
                                to_image=True,
                                show_color_wheel=True,
                                canonical_rotation=np.eye(3),
                                final_prediction=None):
    """Plot a single distribution on SO(3) using the tilt-colored method.
    Args:
        rotations: [N, 3, 3] tensor of rotation matrices
        probabilities: [N] tensor of probabilities
        rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
        ax: The matplotlib.pyplot.axis object to paint
        fig: The matplotlib.pyplot.figure object to paint
        display_threshold_probability: The probability threshold below which to omit
        the marker
        to_image: If True, return a tensor containing the pixels of the finished
        figure; if False return the figure itself
        show_color_wheel: If True, display the explanatory color wheel which matches
        color on the plot with tilt angle
        canonical_rotation: A [3, 3] rotation matrix representing the 'display
        rotation', to change the view of the distribution.  It rotates the
        canonical axes so that the view of SO(3) on the plot is different, which
        can help obtain a more informative view.
    Returns:
        A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
    """
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import tensorflow_graphics.geometry.transformation as tfg

    def _show_single_marker(ax, rotation, marker, edgecolors=True,
                            facecolors=False):
        eulers = tfg.euler.from_rotation_matrix(rotation)
        xyz = rotation[:, 0]
        tilt_angle = eulers[0]
        longitude = np.arctan2(xyz[0], -xyz[1])
        latitude = np.arcsin(xyz[2])

        color = cmap(0.5 + tilt_angle / 2 / np.pi)
        ax.scatter(longitude, latitude, s=2500,
                edgecolors=color if edgecolors else 'none',
                facecolors=facecolors if facecolors else 'none',
                marker=marker,
                linewidth=4)

    if ax is None:
        fig = plt.figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111, projection='mollweide')
    if rotations_gt is not None and len(tf.shape(rotations_gt)) == 2:
        rotations_gt = rotations_gt[tf.newaxis]
    if final_prediction is not None and len(tf.shape(final_prediction)) == 2:
        final_prediction = final_prediction[tf.newaxis]

    display_rotations = rotations @ canonical_rotation
    cmap = plt.cm.hsv
    scatterpoint_scaling = 4e3
    eulers_queries = tfg.euler.from_rotation_matrix(display_rotations)
    xyz = display_rotations[:, :, 0]
    tilt_angles = eulers_queries[:, 0]

    longitudes = np.arctan2(xyz[:, 0], -xyz[:, 1])
    latitudes = np.arcsin(xyz[:, 2])

    which_to_display = (probabilities > display_threshold_probability).squeeze(1)

    if rotations_gt is not None:
        # The visualization is more comprehensible if the GT
        # rotation markers are behind the output with white filling the interior.
        display_rotations_gt = rotations_gt @ canonical_rotation

        for rotation in display_rotations_gt:
            _show_single_marker(ax, rotation, 'o')
        # Cover up the centers with white markers
        for rotation in display_rotations_gt:
            _show_single_marker(ax, rotation, 'x', edgecolors=False,
                            facecolors='#000000')
    if final_prediction is not None:
        # The visualization is more comprehensible if the GT
        # rotation markers are behind the output with white filling the interior.
        display_rotations_pred = final_prediction @ canonical_rotation

        for rotation in display_rotations_pred:
            _show_single_marker(ax, rotation, 'o')
        # Cover up the centers with white markers
        for rotation in display_rotations_pred:
            _show_single_marker(ax, rotation, '+', edgecolors=False,
                            facecolors='#000000')
    # Display the distribution
    ax.scatter(
        longitudes[which_to_display],
        latitudes[which_to_display],
        s=scatterpoint_scaling * probabilities[which_to_display],
        c=cmap(0.5 + tilt_angles[which_to_display] / 2. / np.pi))

    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if show_color_wheel:
        # Add a color wheel showing the tilt angle to color conversion.
        ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection='polar')
        theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
        radii = np.linspace(0.4, 0.5, 2)
        _, theta_grid = np.meshgrid(radii, theta)
        colormap_val = 0.5 + theta_grid / np.pi / 2.
        ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
        ax.set_yticklabels([])
        ax.set_xticklabels([r'90$\degree$', None,
                            r'180$\degree$', None,
                            r'270$\degree$', None,
                            r'0$\degree$'], fontsize=14)
        ax.spines['polar'].set_visible(False)
        plt.text(0.5, 0.5, 'Tilt', fontsize=14,
                horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)

    if to_image:
        return plot_to_image(fig)
    else:
        return fig

def plot_to_image(figure):
    import io
    import matplotlib.pyplot as plt
    import tensorflow as tf
    """Converts matplotlib fig to a png for logging with tf.summary.image."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    plt.close(figure)
    buffer.seek(0)
    image = tf.image.decode_png(buffer.getvalue(), channels=4)
    return image[tf.newaxis]

def create_pcl_plot(scene_pts, gt_tool_pts, init_tool_pts, final_tool_pts):
    """Create flow plot to show on wandb, current points + fitting tool points"
    """
    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[
            [{"type": "scene"}],
        ],
        subplot_titles=(
            "Tool Fitment",
        ),
    )
    scene_pts = scene_pts.detach().cpu()
    gt_tool_pts = gt_tool_pts.detach().cpu()
    init_tool_pts = init_tool_pts.detach().cpu()
    final_tool_pts = final_tool_pts.detach().cpu()

    pos = torch.cat([scene_pts, gt_tool_pts, init_tool_pts, final_tool_pts], dim=0)
    labels = torch.zeros(len(pos)).int()
    labels[:len(scene_pts)] = 0
    labels[len(scene_pts):len(scene_pts)+len(gt_tool_pts)] = 1
    labels[len(scene_pts)+len(gt_tool_pts):len(scene_pts)+len(gt_tool_pts)+len(init_tool_pts)] = 2
    labels[len(scene_pts)+len(gt_tool_pts)+len(init_tool_pts):] = 3
    labelmap = {0: 'scene', 1: 'gt_tool', 2: 'init_tool', 3: 'final_tool'}
    fig.add_traces(_segmentation_traces(pos, labels, labelmap, "scene1"))
    fig.update_layout(
        scene1=_3d_scene(pos),
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(xanchor="left", x=0, yanchor="top", y=0.75),
    )
    return fig

def create_flow_plot(pts, flow, sizeref=4.0, flow_scale = 1., args=None):
    """Create flow plot to show on wandb, current points + (predicted) flow.
    Note: tried numerous ways to add titles and it strangely seems hard. To
    add more info, I'm adjusting the names we supply to the scatter plot and
    increasing its `hoverlabel`. Only for 3D flow!
    """
    from core.toolgen.visualization.primitives import pointcloud
    flow = flow / flow_scale


    # Our pointcloud now should include all points adaptively
    scene = _3d_scene(pts)

    pts_name = 'scene points'
    flow_name = 'tool_flows'

    # Shrink layout, otherwise we get a lot of whitespace.
    layout = go.Layout(
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )

    f = go.Figure(layout=layout)
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene1", name=pts_name)
    )
    ts = _flow_traces_v2(pts[:len(flow)], flow, sizeref=sizeref, scene="scene1", name=flow_name)
    for t in ts:
        f.add_trace(t)
    f.update_layout(scene1=scene)
    f.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

    _adjust_camera(f)
    return f


def _flow_traces_v2(
    pos, flows, sizeref=0.05, scene="scene", flowcolor="red", name="flow",
    pour_water=False,
):
    x_lines = list()
    y_lines = list()
    z_lines = list()

    # Handle 6D flow GT case, present in the pointwise before SVD ablation
    if flows.shape[1] == 6:
        flows[:, :3] += flows[:, 3:]
        flows = flows[:, :3]

    # normalize flows:
    nonzero_flows = (flows == 0.0).all(axis=-1)
    n_pos = pos[~nonzero_flows]
    n_flows = flows[~nonzero_flows]

    n_dest = n_pos + n_flows * sizeref

    for i in range(len(n_pos)):
        x_lines.append(n_pos[i][0])
        y_lines.append(n_pos[i][2])
        z_lines.append(n_pos[i][1])
        x_lines.append(n_dest[i][0])
        y_lines.append(n_dest[i][2])
        z_lines.append(n_dest[i][1])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    lines_trace = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        scene=scene,
        line=dict(color=flowcolor, width=10),
        name=name,
        hoverlabel=dict(namelength=50)
    )

    head_trace = go.Scatter3d(
        x=n_dest[:, 0] if pour_water else -n_dest[:, 0],
        y=n_dest[:, 2],
        z=n_dest[:, 1],
        mode="markers",
        marker={"size": 3, "color": "darkred"},
        scene=scene,
        showlegend=False,
    )

    return [lines_trace, head_trace]