from app.core.gpu_preflight import VramStats, choose_job_gpu_plan


def test_choose_job_gpu_plan_mid_budget():
    stats = VramStats(total_mib=24576, used_mib=15000, free_mib=9576, source="test")
    plan = choose_job_gpu_plan("trueai", "conservative", stats)
    assert 0.55 <= plan.vram_fraction <= 0.85
    assert plan.frames_per_chunk <= 48


def test_choose_job_gpu_plan_low_budget_enables_mitigations():
    stats = VramStats(total_mib=12288, used_mib=10500, free_mib=1788, source="test")
    plan = choose_job_gpu_plan("trueai", "conservative", stats)
    assert plan.attention_slicing
    assert plan.vae_slicing
    assert plan.vae_tiling
    assert plan.batch_size == 1


def test_oom_retry_level_makes_plan_more_conservative():
    stats = VramStats(total_mib=24576, used_mib=12000, free_mib=12576, source="test")
    p0 = choose_job_gpu_plan("trueai", "conservative", stats, oom_retry_level=0)
    p2 = choose_job_gpu_plan("trueai", "conservative", stats, oom_retry_level=2)
    assert p2.vram_fraction <= p0.vram_fraction
    assert p2.frames_per_chunk <= p0.frames_per_chunk
