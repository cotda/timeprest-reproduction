import pytest

from timeprest.schedule import (build_schedule, render_grid, steady_state_v, trace_versions,
                                version_difference_formula)

from timeprest.reference import FIG2, fig2_grid


@pytest.mark.parametrize("W,N", sorted(FIG2))
def test_matches_paper_fig2(W, N):
    expected = fig2_grid(W, N)
    got = render_grid(build_schedule(W, N, 12), W, 18)
    assert got == expected


@pytest.mark.parametrize("W,N", sorted(FIG2))
def test_inflight_cap_does_not_change_nf1b(W, N):
    assert render_grid(build_schedule(W, N, 12), W) == render_grid(build_schedule(W, N, 12, None), W)


def test_1f1b_two_stages_is_pipedream_pattern():
    grid = render_grid(build_schedule(2, 1, 6), 2, 10)
    assert grid[0] == "1A 2A . 1 3A 2 4A 3 5A 4".split()
    assert grid[1] == ". 1A 1 2A 2 3A 3 4A 4 5A".split()


@pytest.mark.parametrize("W,N,K", [(2, 3, 7), (3, 2, 5), (4, 4, 3), (2, 1, 9), (5, 3, 4)])
def test_every_op_once_and_dependencies(W, N, K):
    ops = build_schedule(W, N, K)
    f = {(o.stage, o.mb, o.micro): o.slot for o in ops if o.kind == "F"}
    b = {(o.stage, o.mb): o.slot for o in ops if o.kind == "B"}
    assert len(f) == W * N * K and len(b) == W * K
    assert len(ops) == len(f) + len(b)
    for (s, i, j), t in f.items():
        if s > 0:
            assert f[(s - 1, i, j)] < t
    for (s, i), t in b.items():
        if s == W - 1:
            assert all(f[(s, i, j)] < t for j in range(N))
        else:
            assert b[(s + 1, i)] < t
    # one op per stage per slot
    assert len({(o.stage, o.slot) for o in ops}) == len(ops)


def test_version_difference_condition_eq2():
    """Paper Eq.(2): v = 1 iff W <= N + 1."""
    for W in range(2, 9):
        for N in range(2, 8):
            ops = build_schedule(W, N, 30)
            v = steady_state_v(trace_versions(ops, W, N))
            assert (v == 1) == (W <= N + 1), (W, N, v)


@pytest.mark.parametrize("W,N", [(W, N) for W in range(2, 6) for N in range(2, 6)])
def test_version_difference_formula_eq3(W, N):
    """Paper Eq.(3) matches the simulated schedule for all W <= 5 (the paper's range)."""
    ops = build_schedule(W, N, 30)
    assert steady_state_v(trace_versions(ops, W, N)) == version_difference_formula(W, N)


def test_multiple_sequences_fig2a():
    """W=4, N=2: updates propagate along {1,3,5,..} and {2,4,6,..} (paper §3.4)."""
    ops = build_schedule(4, 2, 12)
    tr = trace_versions(ops, 4, 2)
    # backward of mini-batch i (1-based) uses weights last updated by mini-batch i-2
    for i in range(3, 11):
        assert tr.committed_at_bwd[i - 1] == i - 2


def test_fig2b_micro_batches_c_d_use_new_version():
    """Paper p.4: in Fig.2b micro-batches 3C, 3D start with weights updated by mini-batch 1."""
    tr = trace_versions(build_schedule(4, 4, 6), 4, 4)
    assert tr.fwd_version[(0, 2, 0)] == 0 and tr.fwd_version[(0, 2, 1)] == 0
    assert tr.fwd_version[(0, 2, 2)] == 1 and tr.fwd_version[(0, 2, 3)] == 1


def test_pipedream_stash_counts():
    """PipeDream without vertical sync stashes W - s versions at stage s."""
    for W in (2, 3, 4):
        tr = trace_versions(build_schedule(W, 1, 20), W, 1, vertical_sync=False, backward_version="stashed")
        assert tr.max_versions == [W - s for s in range(W)]


def test_timeprest_backward_uses_live_weights_when_v1():
    """With W <= N+1 the committed version equals every stage's live version at backward time,
    so no horizontal stash is needed."""
    for W, N in [(2, 3), (3, 2), (4, 3), (4, 4)]:
        ops = build_schedule(W, N, 15)
        tr = trace_versions(ops, W, N, True, "committed")
        for idx, o in enumerate(ops):
            if o.kind == "B":
                assert all(tr.bwd_version[(o.stage, o.mb, j)] == tr.live_at_op[idx] for j in range(N))


def test_vertical_sync_forward_versions_equal_across_stages():
    ops = build_schedule(3, 2, 10)
    tr = trace_versions(ops, 3, 2, True, "committed")
    for (s, i, j), k in tr.fwd_version.items():
        assert k == tr.fwd_version[(0, i, j)]
