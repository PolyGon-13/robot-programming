import gymnasium as gym

gym.register(
    id="Twolink-MarkerTrack-v0",
    entry_point=f"{__name__}.marker_track_env:MarkerTrackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.marker_track_env:MarkerTrackEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:MarkerTrackPPORunnerCfg",
    },
)
