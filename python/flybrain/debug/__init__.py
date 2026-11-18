"""Debug helpers (SideChannels, path viz, …)."""

from flybrain.debug.mind_log_channel import (
    KIND_DEEP,
    KIND_MIND,
    MIND_LOG_CHANNEL_ID,
    MindLogDebugChannel,
    format_mind_panel,
    pack_mind_log_message,
    parse_mind_log_message,
)
from flybrain.debug.route_path_channel import (
    ROUTE_PATH_CHANNEL_ID,
    RoutePathDebugChannel,
    build_graph_snapshot,
    pack_graph_message,
    pack_path_message,
    parse_graph_message,
    parse_path_message,
    parse_pose_message,
)

__all__ = [
    "KIND_DEEP",
    "KIND_MIND",
    "MIND_LOG_CHANNEL_ID",
    "MindLogDebugChannel",
    "ROUTE_PATH_CHANNEL_ID",
    "RoutePathDebugChannel",
    "build_graph_snapshot",
    "format_mind_panel",
    "pack_graph_message",
    "pack_mind_log_message",
    "pack_path_message",
    "parse_graph_message",
    "parse_mind_log_message",
    "parse_path_message",
    "parse_pose_message",
]
