"""Stage 5: data augmentation = extra, slightly altered copies of real recordings.

Each copy simulates things that really vary between sessions:
  * a small random 3D rotation (the strap sits a bit differently each time),
    applied to accel AND gyro, since both are measured in the same sensor frame
  * ±10 % amplitude on body acceleration and gyro (walking a bit harder/softer);
    gravity is left at 1 g, otherwise the copy would describe impossible physics
  * a little extra sensor noise

Copies are only ever added to TRAINING folds. Test folds always use the real
recordings, so the score still says how well the model does on real data.
"""

import zlib

import numpy as np

from wearable import config, features, imu


def random_rotation(rng, max_deg=config.AUGMENT_ROTATION_DEG):
    """Rotation matrix about a random axis by a random angle in [-max_deg, max_deg]."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(-max_deg, max_deg))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k     # Rodrigues' formula


def augment_recording(df, rng, max_rot_deg=config.AUGMENT_ROTATION_DEG, scale=config.AUGMENT_SCALE,
                      noise_g=config.AUGMENT_NOISE_G, noise_dps=config.AUGMENT_NOISE_DPS):
    """One altered copy of a recording (same columns, same timestamps)."""
    acc = df[imu.ACCEL_COLS].to_numpy(dtype=float)
    gyr = df[imu.GYRO_COLS].to_numpy(dtype=float)
    grav = features.gravity_lowpass(acc) if len(acc) > 10 else acc

    s = rng.uniform(1 - scale, 1 + scale)
    acc = grav + s * (acc - grav)              # scale the motion, not gravity
    gyr = s * gyr
    r = random_rotation(rng, max_rot_deg)
    acc = acc @ r.T + rng.normal(0, noise_g, acc.shape)
    gyr = gyr @ r.T + rng.normal(0, noise_dps, gyr.shape)

    out = df.copy()
    out[imu.ACCEL_COLS] = np.clip(acc, -config.ACCEL_RANGE_G, config.ACCEL_RANGE_G)
    out[imu.GYRO_COLS] = np.clip(gyr, -config.GYRO_RANGE_DPS, config.GYRO_RANGE_DPS)
    return out


def copy_rng(session, copy, seed=config.RANDOM_SEED):
    """Random generator for copy number `copy` of one session.

    Seeded from the session NAME, so a given copy is identical on every run
    (which is what lets training cache its features).
    """
    return np.random.default_rng([seed, zlib.crc32(session.encode()), copy])


def augmented_copies(recordings, copies=config.AUGMENT_COPIES, seed=config.RANDOM_SEED):
    """(df, activity, session) list -> altered copies, each keeping its ORIGINAL session id.

    Keeping the session id is what makes grouped CV safe: a copy of a test
    session is dropped together with it (see training.grouped_cv).
    """
    return [(augment_recording(df, copy_rng(session, k, seed)), activity, session)
            for df, activity, session in recordings for k in range(copies)]
