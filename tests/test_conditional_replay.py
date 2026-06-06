import torch

from polaris_ebm.scripts.current.conditional_replay import LabelReplayBuffer


def test_label_replay_buffer_returns_label_matched_states_and_restarts():
    replay = LabelReplayBuffer(capacity=6, image_shape=(3, 4, 4), num_classes=3)

    labels = torch.tensor([1, 1, 2], dtype=torch.long)
    state0, stats0 = replay.sample(labels)
    assert torch.equal(state0.y.cpu(), labels)
    assert int(stats0["matched_hits"]) == 0
    assert int(stats0["restarts"]) == 3

    updated_x = torch.randn((3, 3, 4, 4))
    replay.update_slots(state0._replace(x=updated_x, steps_done=torch.tensor([5, 5, 7], dtype=torch.long)))

    state1, stats1 = replay.sample(torch.tensor([1, 2], dtype=torch.long))
    assert torch.equal(state1.y.cpu(), torch.tensor([1, 2], dtype=torch.long))
    assert int(stats1["matched_hits"]) == 2
    assert int(stats1["restarts"]) == 0


def test_label_replay_buffer_state_dict_roundtrip_preserves_labels():
    replay = LabelReplayBuffer(capacity=4, image_shape=(3, 4, 4), num_classes=2)
    state, _ = replay.sample(torch.tensor([0, 1], dtype=torch.long))
    replay.update_slots(state._replace(steps_done=torch.tensor([3, 4], dtype=torch.long)))

    clone = LabelReplayBuffer(capacity=4, image_shape=(3, 4, 4), num_classes=2)
    clone.load_state_dict(replay.state_dict())
    restored, stats = clone.sample(torch.tensor([0, 1], dtype=torch.long))

    assert torch.equal(restored.y.cpu(), torch.tensor([0, 1], dtype=torch.long))
    assert int(stats["matched_hits"]) == 2
    assert torch.equal(restored.steps_done.cpu(), torch.tensor([3, 4], dtype=torch.long))

