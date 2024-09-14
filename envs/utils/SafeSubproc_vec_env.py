import multiprocessing as mp
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env

warnings.simplefilter(action="ignore", category=FutureWarning)

def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info: Optional[Dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            print(f"EOFError caught in worker: {mp.current_process().name}")
            remote.send(("error", "EOFError"))
            break
        except Exception as e:
            print(f"Exception in worker: {e}")
            remote.send(("error", str(e)))
            break


class SafeSubprocVecEnv(VecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        self.env_fns = env_fns
        self.start_method = start_method
        self.failed_subprocesses = set()
        self.remotes = []
        self.work_remotes = []
        self.processes = []
        self._start_subprocesses()

    def _start_subprocesses(self):
        n_envs = len(self.env_fns)
        available_methods = mp.get_all_start_methods()
        
        if self.start_method is None:
            if "spawn" in available_methods:
                self.start_method = "spawn"
            elif "forkserver" in available_methods:
                self.start_method = "forkserver"
            else:
                raise RuntimeError("Neither 'spawn' nor 'forkserver' start methods are available.")
        
        ctx = mp.get_context(self.start_method)

        remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.remotes = list(remotes)
        self.work_remotes = list(work_remotes)
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, self.env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()

        super().__init__(len(self.env_fns), observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        for idx, (remote, action) in enumerate(zip(self.remotes, actions)):
            if idx not in self.failed_subprocesses:
                remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = []
        for idx, remote in enumerate(self.remotes):
            if idx in self.failed_subprocesses:
                results.append((self.observation_space.sample(), 0.0, True, {}, {}))
            else:
                try:
                    result = remote.recv()
                    if isinstance(result, tuple) and len(result) > 0:
                        if isinstance(result[0], str) and result[0] == "error":
                            print(f"Subprocess {idx} encountered an error: {result[1]}")
                            self._handle_subprocess_failure(idx)
                            results.append((self.observation_space.sample(), 0.0, True, {}, {}))
                        else:
                            results.append(result)
                    else:
                        results.append(result)
                except EOFError:
                    print(f"EOFError caught in subprocess {idx}")
                    self._handle_subprocess_failure(idx)
                    results.append((self.observation_space.sample(), 0.0, True, {}, {}))
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)  # type: ignore[assignment]
        all_dones = np.all(dones)
        if all_dones:
            self._restart_failed_subprocesses()
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos  # type: ignore[return-value]

    def _handle_subprocess_failure(self, idx: int):
        print(f"Handling failure of subprocess {idx}.")
        self.failed_subprocesses.add(idx)

    def _restart_failed_subprocesses(self):
        if self.failed_subprocesses:
            print(f"Restarting failed subprocesses: {self.failed_subprocesses}")

            # Close only the failed subprocesses
            for idx in self.failed_subprocesses:
                try:
                    self.remotes[idx].send(("close", None))
                except BrokenPipeError:
                    pass

            # Restart the failed subprocesses
            ctx = mp.get_context(self.start_method)
            new_remotes, new_work_remotes = zip(*[ctx.Pipe() for _ in self.failed_subprocesses])
            new_remotes = list(new_remotes)
            new_work_remotes = list(new_work_remotes)
            new_processes = []
            
            for idx, work_remote, remote, env_fn in zip(self.failed_subprocesses, new_work_remotes, new_remotes, [self.env_fns[i] for i in self.failed_subprocesses]):
                args = (work_remote, remote, CloudpickleWrapper(env_fn))
                process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
                process.start()
                new_processes.append(process)
                work_remote.close()

            # Update the remotes and processes with the new ones
            for idx, new_remote, new_process in zip(self.failed_subprocesses, new_remotes, new_processes):
                self.remotes[idx] = new_remote
                self.processes[idx] = new_process
            
            self.failed_subprocesses.clear()

            # Optionally reset the environments to ensure consistency
            for idx in range(len(self.remotes)):
                if idx not in self.failed_subprocesses:
                    self.remotes[idx].send(("reset", (self._seeds[idx], self._options[idx])))
            results = []
            for idx, remote in enumerate(self.remotes):
                if idx not in self.failed_subprocesses:
                    try:
                        result = remote.recv()
                        if isinstance(result, tuple) and len(result) > 0:
                            if isinstance(result[0], str) and result[0] == "error":
                                print(f"Subprocess {idx} encountered an error during reset: {result[1]}")
                                self._handle_subprocess_failure(idx)
                                results.append((self.observation_space.sample(), {}))
                            else:
                                results.append(result)
                        else:
                            results.append(result)
                    except EOFError:
                        print(f"EOFError caught in subprocess {idx} during reset")
                        self._handle_subprocess_failure(idx)
                        results.append((self.observation_space.sample(), {}))
            obs, self.reset_infos = zip(*results)  # type: ignore[assignment]
            self._reset_seeds()
            self._reset_options()
            return _flatten_obs(obs, self.observation_space)

    def reset(self) -> VecEnvObs:
        for idx, remote in enumerate(self.remotes):
            if idx not in self.failed_subprocesses:
                remote.send(("reset", (self._seeds[idx], self._options[idx])))
        results = []
        for idx, remote in enumerate(self.remotes):
            if idx in self.failed_subprocesses:
                results.append((self.observation_space.sample(), {}))
            else:
                try:
                    result = remote.recv()
                    if isinstance(result, tuple) and len(result) > 0:
                        if isinstance(result[0], str) and result[0] == "error":
                            print(f"Subprocess {idx} encountered an error during reset: {result[1]}")
                            self._handle_subprocess_failure(idx)
                            results.append((self.observation_space.sample(), {}))
                        else:
                            results.append(result)
                    else:
                        results.append(result)
                except EOFError:
                    print(f"EOFError caught in subprocess {idx} during reset")
                    self._handle_subprocess_failure(idx)
                    results.append((self.observation_space.sample(), {}))
        obs, self.reset_infos = zip(*results)  # type: ignore[assignment]
        self._reset_seeds()
        self._reset_options()
        return _flatten_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                try:
                    remote.recv()
                except EOFError:
                    pass
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except BrokenPipeError:
                pass
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.remotes]
        for pipe in self.remotes:
            pipe.send(("render", None))
        outputs = [pipe.recv() for pipe in self.remotes]
        return outputs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices if i not in self.failed_subprocesses]


def _flatten_obs(obs: Union[List[VecEnvObs], Tuple[VecEnvObs]], space: spaces.Space) -> VecEnvObs:
    assert isinstance(obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, spaces.Tuple):
        assert isinstance(obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs]) for i in range(obs_len))  # type: ignore[index]
    else:
        return np.stack(obs)  # type: ignore[arg-type]

