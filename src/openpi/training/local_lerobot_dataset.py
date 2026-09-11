"""Compatibility for sparse episode IDs in the pinned LeRobot version."""

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class SplitLeRobotDataset(LeRobotDataset):
    """Translate original episode IDs only when indexing subset-local bounds.

    Keep original IDs for video filenames and metadata. The pinned LeRobot
    implementation constructs compact episode_data_index arrays for a subset,
    but _get_query_indices indexes those arrays with the original episode ID.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_positions = None
        if self.episodes is not None:
            self.episode_positions = {episode: position for position, episode in enumerate(self.episodes)}
            for episode, position in self.episode_positions.items():
                start = int(self.episode_data_index["from"][position])
                end = int(self.episode_data_index["to"][position])
                if (
                    int(self.hf_dataset[start]["episode_index"]) != episode
                    or int(self.hf_dataset[end - 1]["episode_index"]) != episode
                ):
                    raise ValueError("LeRobot subset order differs from episode bounds; refusing ambiguous windows")

    def _get_query_indices(self, idx, ep_idx):
        position = ep_idx if self.episode_positions is None else self.episode_positions[ep_idx]
        return super()._get_query_indices(idx, position)
