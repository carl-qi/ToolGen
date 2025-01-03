from typing import Dict, Optional, Union

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from core.toolgen.visualization.primitives import pointcloud, sphere, vector

def dcp_sg_plot(action_pos, anchor_pos, t_gt, t_pred, R_gt, R_pred, act_weight):
    oacp, oanp = action_pos.detach().cpu(), anchor_pos.detach().cpu()
    tg = t_gt.detach().cpu().squeeze()
    tp = t_pred.detach().cpu().squeeze()
    Rg = R_gt.detach().cpu().squeeze()
    Rp = R_pred.detach().cpu().squeeze()

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
    oacp_goal_gt = oacp @ Rg.T + tg
    oacp_goal_pred = oacp @ Rp.T + tp
    pos = torch.cat([oanp, oacp, oacp_goal_gt, oacp_goal_pred], dim=0)
    labels = torch.zeros(len(pos)).int()
    labels[: len(oanp)] = 0
    labels[len(oanp) : len(oanp) + len(oacp)] = 1
    labels[len(oanp) + len(oacp) : len(oanp) + 2 * len(oacp)] = 2
    labels[len(oanp) + 2 * len(oacp) : len(oanp) + 3 * len(oacp)] = 3
    labelmap = {0: "anchor", 1: "actor_start", 2: "actor_gt", 3: "actor_pred"}

    fig.add_traces(_segmentation_traces(pos, labels, labelmap, "scene1"))
    fig.update_layout(
        scene1=_3d_scene(pos),
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(xanchor="left", x=0, yanchor="top", y=0.75),
    )

    # Embedding.

    if act_weight is not None:
        aw = act_weight.detach().cpu().squeeze()
        fig.add_trace(
            pointcloud(
                oacp.T, downsample=1, colors=aw, scene="scene2", size=10, colorbar=True
            )
        )
        fig.update_layout(scene2=_3d_scene(oacp))

    return fig

def _3d_scene(data) -> Dict:
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
    data: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
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
        else:
            subset_sizes = sizes[np.where(labels == label)]
        if labelmap is not None:
            legend = labelmap[label]
        else:
            legend = str(label)
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


def _embedding_traces(
    embeddings: np.ndarray,
    labels: np.ndarray,
    classes,
    centers,
    labeldict=None,
    margin: float = 0.5,
    show_centers=True,
    scene="scene",
):
    # If the points are in the margin, they're larger, whereas the points outside
    # the margin are smaller.
    sizes = np.zeros_like(labels).astype(int) + 10
    if show_centers:
        for cl, center in zip(classes, centers):
            # Get the indices for that class.
            ixs = np.where(labels == cl)
            # Calculate the distances.
            dists = np.linalg.norm(embeddings[ixs] - center, axis=1)
            # Points inside their class margin are bigger.
            sizes[ixs[0][np.where(dists <= margin)]] = 20

    traces = _segmentation_traces(
        data=embeddings, labels=labels, labelmap=labeldict, sizes=sizes, scene=scene
    )
    if show_centers:
        cols = np.array(pc.qualitative.Alphabet)
        for cl, center in zip(classes, centers):
            center = center.cpu().numpy()
            col = cols[cl % len(cols)]
            traces.append(
                sphere(center[0], center[1], center[2], margin, 0.25, col, scene)
            )
    return traces


def _articulation_trace(
    origin, direction, color="red", joint_type=0, scene="scene", name="joint"
):

    x, y, z = origin.tolist()
    u, v, w = direction.tolist()

    return vector(x, y, z, u, v, w, color, scene=scene, name=name)


def _isolated_articulation_traces(
    parent_pos: np.ndarray,
    child_pos: np.ndarray,
    joint_type: int,
    origin: np.ndarray,
    direction: np.ndarray,
    parent_name: Optional[str] = None,
    child_name: Optional[str] = None,
    scene="scene",
):
    # Isolated because it's only the parent/child parts.

    art_labels = np.concatenate(
        [np.zeros(len(parent_pos)), np.ones(len(child_pos))]
    ).astype(int)
    art_pts = np.concatenate([parent_pos, child_pos])

    label_dict = None
    if parent_name is not None and child_name is not None:
        label_dict = {0: f"(parent) {parent_name}", 1: f"(child) {child_name}"}

    # Create the base segmentation.
    traces = _segmentation_traces(art_pts, art_labels, label_dict, scene)

    # Add the articulation.
    traces.append(_articulation_trace(origin, direction, joint_type, scene))

    return traces


def _dense_articulation(pos, origins, directions, art_types, scene="scene"):
    rot_dirs = torch.cross(directions.float(), (pos - origins).float())
    dirs = torch.zeros((len(pos)), 3)
    dirs[torch.where(art_types == 0)] = rot_dirs[torch.where(art_types == 0)]
    dirs[torch.where(art_types == 1)] = directions.float()[torch.where(art_types == 1)]

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    u, v, w = dirs[:, 0], dirs[:, 1], dirs[:, 2]

    return go.Cone(
        x=x,
        y=y,
        z=z,
        u=u,
        v=v,
        w=w,
        colorscale="Blues",
        sizemode="absolute",
        showscale=False,
        sizeref=40,
        name="articulations",
        scene=scene,
    )


def _dense_articulation_traces(
    pos, y, origins, directions, art_types, label_dict, scene="scene"
):
    traces = _segmentation_traces(pos, y, label_dict, scene)
    traces.append(_dense_articulation(origins, directions, art_types, scene))
    return traces


def _flow_trace(pos, flows, sizeref, scene="scene"):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    u, v, w = flows[:, 0], flows[:, 1], flows[:, 2]

    return go.Cone(
        x=x,
        y=y,
        z=z,
        u=u,
        v=v,
        w=w,
        colorscale="Blues",
        sizemode="absolute",
        showscale=True,
        sizeref=sizeref,
        name="flow",
        scene=scene,
    )


def _flow_traces_v2(
    pos, flows, sizeref=0.05, scene="scene", flowcolor="red", name="flow"
):
    x_lines = list()
    y_lines = list()
    z_lines = list()

    # normalize flows:
    nonzero_flows = (flows == 0.0).all(axis=-1)
    n_pos = pos[~nonzero_flows]
    n_flows = flows[~nonzero_flows]

    n_dest = n_pos + n_flows * sizeref

    for i in range(len(n_pos)):
        x_lines.append(n_pos[i][0])
        y_lines.append(n_pos[i][1])
        z_lines.append(n_pos[i][2])
        x_lines.append(n_dest[i][0])
        y_lines.append(n_dest[i][1])
        z_lines.append(n_dest[i][2])
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
    )

    head_trace = go.Scatter3d(
        x=n_dest[:, 0],
        y=n_dest[:, 1],
        z=n_dest[:, 2],
        mode="markers",
        marker={"size": 3, "color": "darkred"},
        scene=scene,
        showlegend=False,
    )

    return [lines_trace, head_trace]


# Just a single pointcloud.
def pointcloud_fig(data: np.ndarray, downsample=5, colors=None, size=3, colorbar=False):
    fig = go.Figure()
    fig.add_trace(
        pointcloud(
            data.T, downsample, colors, scene="scene1", size=size, colorbar=colorbar
        )
    )
    fig.update_layout(scene1=_3d_scene(data), showlegend=False)
    return fig


def segmentation_fig(
    data: np.ndarray, labels: np.ndarray, labelmap=None, sizes=None, fig=None
):
    # Create a figure.
    if fig is None:
        fig = go.Figure()

    fig.add_traces(_segmentation_traces(data, labels, labelmap, "scene1", sizes))

    fig.update_layout(
        scene1=_3d_scene(data),
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=1.0, y=0.75),
    )

    return fig


def embedding_fig(
    embeddings: np.ndarray,
    labels: np.ndarray,
    classes,
    centers,
    labeldict=None,
    margin: float = 0.5,
    show_centers=True,
):
    fig = go.Figure()
    fig.add_traces(
        _embedding_traces(
            embeddings,
            labels,
            classes,
            centers,
            labeldict,
            margin,
            show_centers,
            scene="scene1",
        )
    )
    fig.update_layout(scene1=_3d_scene(embeddings), showlegend=True)

    # Hide the dumb colorscale.
    fig.update(layout_coloraxis_showscale=False)
    return fig


def class_hist_fig(hist: Dict[int, float], labelmap: Dict[int, str]):
    cmap = np.array(pc.qualitative.Alphabet)
    colors = [cmap[cl % len(cmap)] for cl in hist.keys()]
    return go.Figure(
        data=[
            go.Bar(
                x=[
                    labelmap[key] for key in hist.keys()
                ],  # Relabel with the class names.
                y=list(hist.values()),
                marker_color=colors,
            )
        ]
    )


# def show_data(data: PartNetData):
#     f = segmentation_fig(data.pos.numpy(), data.y.numpy(), data.name_dict)
#     f.show()


def isolated_articulation_fig(
    parent_pos: np.ndarray,
    child_pos: np.ndarray,
    joint_type: int,
    origin: np.ndarray,
    direction: np.ndarray,
    parent_name: Optional[str] = None,
    child_name: Optional[str] = None,
    fig: Optional[go.Figure] = None,
):
    if fig is None:
        fig = go.Figure()

    traces = _isolated_articulation_traces(
        parent_pos,
        child_pos,
        joint_type,
        origin,
        direction,
        parent_name,
        child_name,
        "scene1",
    )
    fig.add_traces(traces)

    art_pts = np.concatenate([parent_pos, child_pos])
    fig.update_layout(scene1=_3d_scene(art_pts))

    return fig


def masked_articulation_fig(
    pos,
    mask,
    origin,
    direction,
    child_name=None,
    fig: Optional[go.Figure] = None,
):
    if fig is None:
        fig = go.Figure()

    labelmap = {0: "non_moving", 1: child_name if child_name is not None else "child"}

    traces = _segmentation_traces(
        pos, mask.squeeze(), labelmap=labelmap, scene="scene1"
    )
    fig.add_traces(traces)

    traces = _articulation_trace(
        torch.as_tensor(origin), torch.as_tensor(direction), scene="scene1"
    )
    fig.add_traces(traces)

    # fig.update_layout(scene1=_3d_scene(pos))
    return fig


def dense_articulation_fig(pos, y, origins, directions, art_types, label_dict):
    f = go.Figure()
    f.add_traces(
        _dense_articulation_traces(
            pos, y, origins, directions, art_types, label_dict, "scene1"
        )
    )
    f.update_layout(scene1=_3d_scene(pos))
    return f


def flow_fig(pos, flows, sizeref, use_v2=False):
    if not use_v2:
        f = go.Figure()
        f.add_trace(pointcloud(pos.T, downsample=1, scene="scene1"))
        f.add_trace(_flow_trace(pos, flows, sizeref, scene="scene1"))
        f.update_layout(scene1=_3d_scene(pos))
    else:
        f = go.Figure()
        f.add_trace(pointcloud(pos.T, downsample=1, scene="scene1"))
        ts = _flow_traces_v2(pos, flows, sizeref=sizeref, scene="scene1")
        for t in ts:
            f.add_trace(t)
        f.update_layout(scene1=_3d_scene(pos))

    return f