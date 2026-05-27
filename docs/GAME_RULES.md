# Orbit Wars Game Rules for the Coding Agent

This file explains the Orbit Wars competition mechanics in a form useful for developing the bot. Read this before changing strategy code.

## 1. Competition type

Orbit Wars is a Kaggle game-agent competition. We do not submit predictions for a dataset. We submit a Python bot.

The submitted package must expose:

```python
def agent(obs):
    return moves
```

The bot receives the current game observation every turn and returns actions for that turn.

## 2. Objective

The game lasts up to 500 turns.

A player wins by having the highest final ship count:

```text
final_score = ships on owned planets + ships in owned fleets
```

The practical goal is to grow economy while avoiding waste:

```text
capture valuable planets -> increase production -> produce more ships -> defend and attack efficiently
```

## 3. Board

- Board size: 100 x 100 continuous coordinates.
- Origin: top-left corner.
- Center: `(50, 50)`.
- Sun: circle centered at `(50, 50)` with radius `10`.
- Fleets crossing the sun are destroyed.
- Coordinates use screen convention: increasing `x` moves right, increasing `y` moves down.

Angle convention:

```text
0       = right
pi / 2  = down
pi      = left
-pi / 2 = up
```

Use:

```python
math.atan2(target.y - source.y, target.x - source.x)
```

to aim from a source to a target.

## 4. Planets

Each planet is represented as:

```python
[id, owner, x, y, radius, ships, production]
```

Meaning:

```text
id          unique planet id
owner       -1 for neutral, otherwise player id 0-3
x, y        current position
radius      physical collision radius
ships       current garrison
production  ships produced per turn when owned
```

Important strategic facts:

- High production planets are valuable.
- A planet with many ships is expensive to capture.
- A far planet has higher travel cost and slower strategic payoff.
- Owned planets generate `production` ships every turn.

## 5. Planet movement

There are two types of normal planets:

### Static planets

Static planets do not move.

### Orbiting planets

Inner planets orbit around the sun. To hit them correctly, do not simply aim at the current position. Estimate where the planet will be when the fleet arrives.

A practical approach:

```text
1. Estimate distance from source to current target position.
2. Estimate travel time = distance / fleet_speed.
3. Predict future target position after travel_time.
4. Recompute distance, travel time, and angle.
5. Repeat several times.
```

The helper `estimate_intercept` in `src/geometry.py` does this approximately.

## 6. Fleets

Each fleet is represented as:

```python
[id, owner, x, y, angle, from_planet_id, ships]
```

Meaning:

```text
id              unique fleet id
owner           player id
x, y            current position
angle           travel direction in radians
from_planet_id  source planet id
ships           ships in fleet
```

A fleet flies in a straight line at a speed determined by the number of ships.

## 7. Fleet speed

Fleet speed increases with fleet size. Small fleets are slow. Large fleets are faster, up to maximum speed.

Competition formula:

```text
speed = 1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ^ 1.5
```

with default `maxSpeed = 6.0`.

Strategic consequence:

```text
A larger fleet is not only stronger; it also arrives sooner.
```

Do not assume `target.ships + 1` is always enough. If travel takes time and the target is owned, the target produces additional ships before arrival.

## 8. Actions

The agent returns a list of moves:

```python
[[from_planet_id, direction_angle, num_ships], ...]
```

Example:

```python
[[3, 1.57, 20]]
```

This means:

```text
From planet 3, launch 20 ships in direction 1.57 radians.
```

Constraints:

- Can launch only from planets we own.
- Cannot launch more ships than the planet currently has.
- Multiple launches from the same planet are allowed, but risky and easy to overcommit.
- Returning `[]` means no action.

## 9. Turn order

The official turn order is important for timing. Conceptually:

```text
1. Remove expired comets.
2. Spawn new comet groups.
3. Process player fleet launches.
4. Apply production on owned planets.
5. Move fleets and check collisions.
6. Rotate planets and move comets.
7. Resolve queued combats.
```

Strategic consequences:

- Owned planets produce after launches.
- A target may become stronger before our fleet arrives.
- Moving planets/comets can sweep into fleets.

## 10. Combat

Basic combat is subtraction.

If our fleet attacks a neutral planet:

```text
planet has 20 ships
we arrive with 25 ships
we capture it with 5 ships remaining
```

If our fleet attacks an enemy planet:

```text
enemy planet has 50 ships
we arrive with 80 ships
we capture it with 30 ships remaining
```

If several attacking owners arrive together:

```text
1. Arriving fleets are grouped by owner.
2. The largest attacking group fights the second largest.
3. The difference survives.
4. The survivor fights the planet garrison.
```

Strategic consequences:

- Timing matters.
- Overkill wastes ships.
- Underpowered attacks are usually bad unless they intentionally weaken a target for later.

## 11. Comets

Comets are temporary planets.

Facts:

- They appear at specific turns.
- They move through the board.
- They can be captured.
- They produce ships while owned.
- They eventually leave the board and disappear with any garrisoned ships.

Strategic rules:

- Capture easy comets if they will remain long enough.
- Do not overinvest in comets that will soon leave.
- Later versions should evacuate ships from valuable comets before they disappear.

## 12. Observation fields

Important fields in `obs`:

```text
planets             list of planet rows
fleets              list of fleet rows
player              our player id
angular_velocity    rotation speed for orbiting planets
initial_planets     initial planet positions; useful for exact orbit prediction later
comets              active comet group data and paths
comet_planet_ids    ids of planets that are comets
remainingOverageTime remaining time budget
step / turn         current turn if available
```

The template parses the most important fields in `src/game.py`.

## 13. Good strategic priorities

The bot should improve in this order:

```text
1. Return valid moves.
2. Avoid sending fleets through the sun.
3. Capture profitable neutral planets.
4. Keep defensive reserves.
5. Estimate future garrisons using travel time.
6. Aim at predicted positions for orbiting planets.
7. Detect and defend against incoming enemy fleets.
8. Attack weak enemy planets.
9. Use comets intelligently.
10. Tune parameters over many seeds.
```

## 14. What not to do early

Do not start with deep RL before the heuristic bot is strong.

Avoid:

```text
- heavy dependencies
- slow per-turn search without benchmarking
- random behavior that makes debugging hard
- printing from the official submission agent
- rewriting the whole bot at once
```

## 15. Current baseline behavior

The current template implements a simple expansion bot:

```text
1. Parse observation.
2. Find our planets.
3. Find planets not owned by us.
4. For each strong source planet:
   - keep a reserve
   - choose nearby candidate targets
   - estimate interception angle and travel time
   - reject paths that cross the sun
   - estimate required ships
   - score target by production, distance, cost, enemy/comet bonus
5. Launch if the best score is acceptable.
```

This is only a starting point. The biggest missing piece is defense against incoming enemy fleets.
