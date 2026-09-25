# sendy

Applies the `diffusion_displacement` package's model + inference pipeline
(`DisplacementDiffusionSDE`, VP-SDE, image + start-point conditioning, 10
waypoints x 3 = 30-dim displacement action) to the uterine system, driven by
a manual state machine modeled on the ALISS `TaskPublisher` pattern: one node
publishes a task name on `/current_task`, and every other node reacts only
to the task names it owns.

## Pipeline

```
                         /current_task (String)
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 uterine_diffusion_node   uterine_move_node        uterine_move_node_r
 (retract_start, retract,  (screw)                 (nothing — reacts only
  resect_start, resect)                             to trajectory topics)
        │                       │
        ├─ /retract_start_action ──┐
        ├─ /retract_action ────────┼──► uterine_move_node   (left arm, /xyz_l)
        ├─ /resect_start_action ───┐
        └─ /resect_action ─────────┴──► uterine_move_node_r (right arm, /xyz_r)
```

Type tasks into `uterine_task_publisher`'s prompt (tab-autocomplete, same as
the ALISS one) to drive the sequence, e.g.:

```
retract_start
screw
retract
retract
retract
```

Each typed task is independent — repeat any of them as many times as needed.

## Tasks

| task            | arm   | handled by              | model-driven? |
|------------------|-------|--------------------------|----------------|
| `retract_start`  | left  | uterine_diffusion_node   | yes |
| `screw`          | left  | uterine_move_node        | no — fixed screw/insert/screw sequence at the CURRENT tool pose |
| `retract`        | left  | uterine_diffusion_node   | yes |
| `resect_start`   | right | uterine_diffusion_node   | yes |
| `resect`         | right | uterine_diffusion_node   | yes |

## Nodes

- **`uterine_task_publisher`** — interactive CLI (readline autocomplete),
  publishes the typed task name on `/current_task`. Run this in its own
  terminal.

- **`uterine_diffusion_node`** — loads one `DisplacementDiffusionSDE`
  checkpoint per diffusion-driven task (`retract_start`/`retract` off
  `/fwkin_l`, `resect_start`/`resect` off `/fwkin_r`, camera shared on
  `/hy_camera/image`). On a matching `/current_task` message, runs inference
  (`N_SAMPLES` candidates → optional matplotlib picker → integrate
  displacements into `N_WAYPOINTS` waypoints) and publishes the full
  trajectory to that task's action topic. Ignores `screw` and anything else
  it doesn't recognize. Any task with a missing weights file is skipped at
  startup (logged, not fatal) so partially-trained setups still work.

- **`uterine_move_node`** (left arm) — reacts to `/current_task == "screw"`
  by running the screw/insert/screw sequence directly at whatever the
  current `/fwkin_l` pose is (no trajectory needed). Separately, walks
  every waypoint of whatever arrives on `/retract_start_action` or
  `/retract_action` via `/xyz_l` — pure point-to-point motion, no screw
  embedded anymore.

- **`uterine_move_node_r`** (right arm / cautery) — walks every waypoint of
  whatever arrives on `/resect_start_action` or `/resect_action` via
  `/xyz_r`. No screw, no extra activation trigger (per confirmation that
  cautery is plain move-to-point).

## Changed vs. the earlier single-task version

- The diffusion node's own keyboard menu is gone — dispatch is entirely via
  `/current_task` now.
- `screw` is its own task, no longer automatically run at waypoint 0 of a
  retract trajectory. `retract_start`/`retract` are now pure point-to-point.
- The screw sequence's insertion offset changed from move_node.py's original
  `0.004` m (4mm) to `SCREW_INSERT_DISTANCE_M = 0.002` m (2mm), per "screw
  then 2mm then screw" — **double check this value**, since it's a real
  change from the original hardcoded number, not just a refactor.
- The old blocking `input()` → return-home sequence was removed from the
  trajectory walk (it doesn't fit a callback-driven state machine — blocking
  stdin inside a subscription callback would freeze `/current_task`
  processing). `return_home_sequence()` is still defined in
  `uterine_move_node.py`, just unwired; add a `"home_left"` entry to
  `task_publisher_node.py`'s task list and a case in `task_callback` when
  you want it back.

## Before running

1. Drop trained checkpoints under
   `sendy/models/diffusion_displacement/weights/`:
   `uterine_retract_start_sde.pt`, `uterine_retract_sde.pt`,
   `uterine_resect_start_sde.pt`, `uterine_resect_sde.pt`. Missing ones are
   skipped with a warning rather than crashing the node.
2. Confirm `/fwkin_r` and `/xyz_r` are live on the right arm (assumed to
   mirror `/fwkin_l`/`/xyz_l` naming).
3. `colcon build --packages-select sendy`

## Not included (flagged as future work)

- BoostSlip retry (slip-detection-driven re-sampling) from
  `retract_fib_diffusion_node.py` — no `/done_retract` signal or trained
  slip classifier exists for either arm here yet.
- `home_left`/`home_right`/other CAO-style housekeeping tasks — only the 5
  tasks above are wired up; easy to extend the same way.
