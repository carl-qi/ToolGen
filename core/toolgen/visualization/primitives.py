from typing import Any, Dict

import numpy as np
from plotly import graph_objects as go


def pointcloud(
    T_chart_points: np.ndarray,
    downsample=5,
    colors=None,
    scene="scene",
    name=None,
    size=3,
    colorbar=False,
) -> go.Scatter3d:
    marker_dict: Dict[str, Any] = {"size": size}
    if colors is not None:
        try:
            a = [f"rgb({r}, {g}, {b})" for r, g, b in colors][::downsample]
            marker_dict["color"] = a
        except:
            marker_dict["color"] = colors[::downsample]

            if colorbar:
                marker_dict["colorbar"] = dict(thickness=20)
    return go.Scatter3d(
        x=T_chart_points[0, ::downsample],
        y=T_chart_points[2, ::downsample],
        z=T_chart_points[1, ::downsample],
        mode="markers",
        marker=marker_dict,
        scene=scene,
        name=name,
    )


def sphere(x, y, z, r, opacity, color, scene="scene"):
    phi = np.linspace(0, 2 * np.pi, 20)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 20)
    phi, theta = np.meshgrid(phi, theta)

    xs = np.cos(theta) * np.sin(phi) * r + x
    ys = np.cos(theta) * np.cos(phi) * r + y
    zs = np.sin(theta) * r + z

    return go.Surface(
        x=xs,
        y=ys,
        z=zs,
        colorscale=[[0, color], [1, color]],
        opacity=opacity,
        showscale=False,
        scene=scene,
    )


def vector(x, y, z, u, v, w, color, scene="scene", name="vector"):
    return go.Scatter3d(
        x=[x, x + u],
        y=[y, y + v],
        z=[z, z + w],
        line=dict(color=color, width=10),
        scene=scene,
        name=name,
    )
