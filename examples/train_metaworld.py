import os
import random
import time

import logging

logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()
logger.addFilter(CheckTypesFilter())

import gym
from gym.wrappers import RescaleAction
from gym.wrappers.pixel_observation import PixelObservationWrapper

from jaxrl import wrappers

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from jaxrl.agents import AWACLearner, DDPGLearner, REDQLearner, SACLearner, SACV1Learner
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate

import metaworld

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "HalfCheetah-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("updates_per_step", 1, "Gradient updates per step.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("track", False, "Track experiments with Weights and Biases.")
flags.DEFINE_string("wandb_project_name", "jaxrl", "The wandb's project name.")
flags.DEFINE_string("wandb_entity", None, "the entity (team) of wandb's project")
config_flags.DEFINE_config_file(
    "config",
    "configs/sac_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    kwargs = dict(FLAGS.config)
    algo = kwargs.pop("algo")
    run_name = f"{FLAGS.env_name}__{algo}__{FLAGS.seed}__{int(time.time())}"
    if FLAGS.track:
        import wandb

        wandb.init(
            project=FLAGS.wandb_project_name,
            entity=FLAGS.wandb_entity,
            sync_tensorboard=True,
            config=FLAGS,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.config.update({"algo": algo})

    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, run_name))

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, "video", "train")
        video_eval_folder = os.path.join(FLAGS.save_dir, "video", "eval")
    else:
        video_train_folder = None
        video_eval_folder = None

    ml1 = metaworld.ML1(FLAGS.env_name)

    env = ml1.train_classes[FLAGS.env_name]()
    task = random.choice(ml1.train_tasks)
    env.set_task(task)

    if env.max_path_length is not None:
        env = gym.wrappers.TimeLimit(env, env.max_path_length)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    env = wrappers.EpisodeMonitor(env)
    env = RescaleAction(env, -1.0, 1.0)

    if video_train_folder is not None:
        env = gym.wrappers.RecordVideo(env, video_train_folder)

    env = wrappers.SinglePrecision(env)

    env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)
    env.observation_space.seed(FLAGS.seed)

    eval_env = ml1.test_classes[FLAGS.env_name]()
    eval_task = random.choice(ml1.test_tasks)
    eval_env.set_task(eval_task)

    if eval_env.max_path_length is not None:
        eval_env = gym.wrappers.TimeLimit(eval_env, eval_env.max_path_length)

    if isinstance(eval_env.observation_space, gym.spaces.Dict):
        eval_env = gym.wrappers.FlattenObservation(eval_env)

    eval_env = wrappers.EpisodeMonitor(eval_env)
    eval_env = RescaleAction(eval_env, -1.0, 1.0)

    if video_eval_folder is not None:
        eval_env = gym.wrappers.RecordVideo(eval_env, video_eval_folder)

    eval_env = wrappers.SinglePrecision(eval_env)

    eval_env.seed(FLAGS.seed)
    eval_env.action_space.seed(FLAGS.seed)
    eval_env.observation_space.seed(FLAGS.seed)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    replay_buffer_size = kwargs.pop("replay_buffer_size")
    if algo == "sac":
        agent = SACLearner(
            FLAGS.seed,
            env.observation_space.sample()[np.newaxis],
            env.action_space.sample()[np.newaxis],
            **kwargs,
        )
    elif algo == "redq":
        agent = REDQLearner(
            FLAGS.seed,
            env.observation_space.sample()[np.newaxis],
            env.action_space.sample()[np.newaxis],
            policy_update_delay=FLAGS.updates_per_step,
            **kwargs,
        )
    elif algo == "sac_v1":
        agent = SACV1Learner(
            FLAGS.seed,
            env.observation_space.sample()[np.newaxis],
            env.action_space.sample()[np.newaxis],
            **kwargs,
        )
    elif algo == "awac":
        agent = AWACLearner(
            FLAGS.seed,
            env.observation_space.sample()[np.newaxis],
            env.action_space.sample()[np.newaxis],
            **kwargs,
        )
    elif algo == "ddpg":
        agent = DDPGLearner(
            FLAGS.seed,
            env.observation_space.sample()[np.newaxis],
            env.action_space.sample()[np.newaxis],
            **kwargs,
        )
    else:
        raise NotImplementedError()

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, replay_buffer_size or FLAGS.max_steps
    )

    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            observation, action, reward, mask, float(done), next_observation
        )
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info["episode"].items():
                summary_writer.add_scalar(
                    f"training/{k}", v, info["total"]["timesteps"]
                )

            if "is_success" in info:
                summary_writer.add_scalar(
                    f"training/success", info["is_success"], info["total"]["timesteps"]
                )

        if i >= FLAGS.start_training:
            for _ in range(FLAGS.updates_per_step):
                batch = replay_buffer.sample(FLAGS.batch_size)
                update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f"training/{k}", v, i)
                summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(
                    f"evaluation/average_{k}s", v, info["total"]["timesteps"]
                )
            summary_writer.flush()

            eval_returns.append((info["total"]["timesteps"], eval_stats["return"]))
            np.savetxt(
                os.path.join(FLAGS.save_dir, f"{FLAGS.seed}.txt"),
                eval_returns,
                fmt=["%d", "%.1f"],
            )


if __name__ == "__main__":
    app.run(main)