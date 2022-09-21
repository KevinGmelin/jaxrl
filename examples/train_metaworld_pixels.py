import os
import random
import time

import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()
logger.addFilter(CheckTypesFilter())

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from jaxrl.agents import DrQLearner
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate_metaworld
from jaxrl.utils import make_metaworld_env
import metaworld

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "cheetah-run", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 512, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(5e5), "Number of environment steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of environment steps to start training."
)
flags.DEFINE_integer(
    "action_repeat", None, "Action repeat, if None, uses 2 or PlaNet default values."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("track", False, "Track experiments with Weights and Biases.")
flags.DEFINE_string("wandb_project_name", "jaxrl", "The wandb's project name.")
flags.DEFINE_string("wandb_entity", None, "the entity (team) of wandb's project")
config_flags.DEFINE_config_file(
    "config",
    "configs/drq_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

PLANET_ACTION_REPEAT = {
    "cartpole-swingup": 8,
    "reacher-easy": 4,
    "cheetah-run": 4,
    "finger-spin": 2,
    "ball_in_cup-catch": 4,
    "walker-walk": 2,
}


def main(_):
    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, "tb", str(FLAGS.seed)))

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, "video", "train")
        video_eval_folder = os.path.join(FLAGS.save_dir, "video", "eval")
    else:
        video_train_folder = None
        video_eval_folder = None

    if FLAGS.action_repeat is not None:
        action_repeat = FLAGS.action_repeat
    else:
        action_repeat = PLANET_ACTION_REPEAT.get(FLAGS.env_name, 2)

    kwargs = dict(FLAGS.config)
    gray_scale = kwargs.pop("gray_scale")
    image_size = kwargs.pop("image_size")

    algo = "DrQ"
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

    def make_pixel_env(mt1, seed, video_folder):
        return make_metaworld_env(
            mt1,
            FLAGS.env_name,
            seed,
            video_folder,
            action_repeat=action_repeat,
            image_size=image_size,
            frame_stack=3,
            from_pixels=True,
            gray_scale=gray_scale,
        )

    mt1 = metaworld.MT1(FLAGS.env_name)
    env = make_pixel_env(mt1, FLAGS.seed, video_train_folder)
    eval_env = make_pixel_env(mt1, FLAGS.seed + 42, video_eval_folder)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    kwargs.pop("algo")
    replay_buffer_size = kwargs.pop("replay_buffer_size")
    agent = DrQLearner(
        FLAGS.seed,
        env.observation_space.sample()[np.newaxis],
        env.action_space.sample()[np.newaxis],
        **kwargs,
    )

    replay_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        replay_buffer_size or FLAGS.max_steps // action_repeat,
    )

    eval_returns = []
    task = random.choice(mt1.train_tasks)
    env.set_task(task)
    observation, done = env.reset(), False
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps // action_repeat + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
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
            task = random.choice(mt1.train_tasks)
            env.set_task(task)
            observation, done = env.reset(), False
            for k, v in info["episode"].items():
                summary_writer.add_scalar(
                    f"training/{k}", v, info["total"]["timesteps"]
                )
            summary_writer.flush()

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f"training/{k}", v, i)
                summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate_metaworld(agent, eval_env, mt1, FLAGS.eval_episodes)

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
