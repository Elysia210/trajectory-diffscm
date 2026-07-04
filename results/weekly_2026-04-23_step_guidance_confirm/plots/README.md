# Report GIF Selection

This folder contains two kinds of assets for the five selected Seed-123
examples:

- `15` single-mode GIFs:
  - `unguided`
  - `collision_s2p0`
  - `no_collision_s1p0`
- `5` trio-comparison GIFs:
  - `*_compare_main_trio.gif`

## Selection logic

The five samples were chosen to cover a useful reporting spectrum:

- `sample12`: strongest bidirectional separation, especially strong
  no-collision steering.
- `sample31`: strongest collision push on an already risky scene.
- `sample26`: large guidance effect from a moderate baseline.
- `sample29`: medium-response, more typical case.
- `sample00`: low-response / guidance-insensitive case.

## Selected samples

### sample12

- scene: `scene-0052_ego_0_ctrl_[1]_0`
- probabilities:
  - unguided `0.661`
  - collision_s2.0 `0.643`
  - no_collision_s1.0 `0.134`
- trio:
  - `sample12_compare_main_trio.gif`
- singles:
  - `sample12_unguided.gif`
  - `sample12_collision_s2p0.gif`
  - `sample12_no_collision_s1p0.gif`

### sample31

- scene: `scene-0563_ego_0_ctrl_[5]_0`
- probabilities:
  - unguided `0.639`
  - collision_s2.0 `0.743`
  - no_collision_s1.0 `0.245`
- trio:
  - `sample31_compare_main_trio.gif`
- singles:
  - `sample31_unguided.gif`
  - `sample31_collision_s2p0.gif`
  - `sample31_no_collision_s1p0.gif`

### sample26

- scene: `scene-1007_ego_0_ctrl_[7]_0`
- probabilities:
  - unguided `0.301`
  - collision_s2.0 `0.597`
  - no_collision_s1.0 `0.118`
- trio:
  - `sample26_compare_main_trio.gif`
- singles:
  - `sample26_unguided.gif`
  - `sample26_collision_s2p0.gif`
  - `sample26_no_collision_s1p0.gif`

### sample29

- scene: `scene-0304_ego_0_ctrl_[6]_0`
- probabilities:
  - unguided `0.186`
  - collision_s2.0 `0.253`
  - no_collision_s1.0 `0.164`
- trio:
  - `sample29_compare_main_trio.gif`
- singles:
  - `sample29_unguided.gif`
  - `sample29_collision_s2p0.gif`
  - `sample29_no_collision_s1p0.gif`

### sample00

- scene: `scene-0707_ego_0_ctrl_[3]_0`
- probabilities:
  - unguided `0.169`
  - collision_s2.0 `0.175`
  - no_collision_s1.0 `0.168`
- trio:
  - `sample00_compare_main_trio.gif`
- singles:
  - `sample00_unguided.gif`
  - `sample00_collision_s2p0.gif`
  - `sample00_no_collision_s1p0.gif`

## Recommended use in slides

- Lead with `sample31` or `sample26` if the goal is to show collision guidance
  clearly increasing risk.
- Use `sample12` to show that no-collision steering can also produce a very
  strong separation.
- Keep `sample29` as a less dramatic but still representative example.
- Keep `sample00` as the "little change" / hard-case example.
