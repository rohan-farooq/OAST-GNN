#!/usr/bin/env python3
r"""Collect CARLA reporter-target V2X events for the A6 trust experiment.

Designed for CARLA 0.9.16 and Python 3.10-3.12.

The experiment uses one bidirectional CARLA ``sensor.other.v2x_custom`` sensor
per participant vehicle and one at the fixed RSU. Payload kind ``k`` separates
the two logical message types:

1. ``<channel>-beacon``: every vehicle broadcasts its claimed kinematic state.
   Each receiving vehicle becomes a reporter and the sender is the target.
   Beacon rounds use rotating one-tick TDMA slots so simultaneous ``send()``
   calls cannot suppress a persistent subset of vehicle transmitters.
2. Reports: reporters submit compact risk reports to a fixed RSU.

A vehicle never calls ``send()`` twice in the same simulation tick. Giving each
vehicle a single bidirectional custom-V2X sensor avoids conflicting sensor
registrations on the same parent actor.

Target misbehavior and Byzantine reporting are deliberately separate:

* Malicious targets falsify the position in their beacons.
* Byzantine reporters manipulate the locally derived risk score.

Only reports that traverse both simulated wireless hops and reach the RSU are
written to the original CSV. Additional .beacons.jsonl logs all drained beacon
callbacks before filtering; .self.jsonl records ideal ego states; the separate
.attacks.evaluation.jsonl records explicit attack interventions. CARLA supplies receiver-specific delivery and received power.
The local physical observation used by this starter is CARLA ground truth plus
distance/power-dependent noise. Replace ``make_local_observation`` with a
camera/LiDAR perception output when a full perception stack is available.

This is a synthetic closed-loop CARLA evaluation. It must not be described as
real-world or VeReMi data.

Example (run CarlaUE4.exe first):

    python carla_a6_v2x_collector.py ^
      --map Town05 --scenario urban --num-vehicles 20 --duration 60 ^
      --target-malicious-fraction 0.20 --target-attack position_offset ^
      --collusion-fraction 0.30 --reporter-attack coordinated ^
      --output .\outputs\carla_a6_collusion_30.csv
"""

from __future__ import annotations

import argparse
import traceback
from history_logger import HistoryLogger
from observation_logger import ObservationLogger
from target_attacks import ATTACKS, TargetAttackEngine, select_coalition
import atexit
import faulthandler
import csv
import hashlib
import json
import math
import queue
import random
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

def checkpoint(message: str) -> None:
    """Write diagnostics immediately; redirect stderr to retain native fault output."""
    print(f"[A6-DIAG {time.perf_counter():.6f} thread={threading.get_ident()}] {message}",
          file=sys.stderr, flush=True)


def diagnostic_cleanup(listening_sensors, sensors, spawned_vehicles,
                       traffic_manager, world, original_settings) -> None:
    checkpoint("CLEANUP BEGIN: autopilot-detach candidate; not a confirmed fix")
    errors = []
    for index, sensor in enumerate(listening_sensors):
        label = f"listener[{index}]"
        try:
            checkpoint(f"{label}: BEFORE is_alive")
            alive = sensor.is_alive
            checkpoint(f"{label}: AFTER is_alive={alive}")
            if alive:
                checkpoint(f"{label}: BEFORE stop")
                sensor.stop()
                checkpoint(f"{label}: AFTER stop")
        except RuntimeError as exc:
            checkpoint(f"{label}: RuntimeError {exc!r}")
            errors.append(f"{label}: {exc}")

    # Only shutdown changes below: recording has finished and listeners are stopped.
    # The original collector destroyed participant vehicles while still on autopilot.
    # Explicitly detach them on the same TM port before destroying their actors.
    for index, vehicle in enumerate(spawned_vehicles):
        label = f"autopilot-detach[{index}]"
        try:
            checkpoint(f"{label}: BEFORE is_alive")
            if vehicle.is_alive:
                checkpoint(f"{label}: BEFORE set_autopilot(False)")
                vehicle.set_autopilot(False, traffic_manager.get_port())
                checkpoint(f"{label}: AFTER set_autopilot(False)")
        except RuntimeError as exc:
            checkpoint(f"{label}: RuntimeError {exc!r}")
            errors.append(f"{label}: {exc}")

    def shutdown_tick(label):
        # No sleeps used as a claimed synchronization guarantee. A bounded tick
        # merely exercises the server/TM with the changed actor registry.
        try:
            checkpoint(f"{label}: BEFORE get_settings")
            if spawned_vehicles and world.get_settings().synchronous_mode:
                checkpoint(f"{label}: BEFORE tick (10 second timeout)")
                frame = world.tick(10.0)
                checkpoint(f"{label}: AFTER tick frame={frame}")
            else:
                checkpoint(f"{label}: tick skipped (asynchronous or no vehicles)")
        except RuntimeError as exc:
            checkpoint(f"{label}: RuntimeError {exc!r}")
            errors.append(f"{label}: {exc}")

    shutdown_tick("AFTER DETACH")
    for index, actor in enumerate(list(reversed(sensors)) + list(reversed(spawned_vehicles))):
        label = f"destroy[{index}]"
        try:
            checkpoint(f"{label}: BEFORE is_alive")
            alive = actor.is_alive
            checkpoint(f"{label}: AFTER is_alive={alive}")
            if alive:
                checkpoint(f"{label}: BEFORE destroy")
                result = actor.destroy()
                checkpoint(f"{label}: AFTER destroy result={result!r}")
                if result is False:
                    errors.append(f"{label}: destroy returned False")
        except RuntimeError as exc:
            checkpoint(f"{label}: RuntimeError {exc!r}")
            errors.append(f"{label}: {exc}")
    shutdown_tick("AFTER DESTROY")
    try:
        checkpoint("TM: BEFORE set_synchronous_mode(False)")
        traffic_manager.set_synchronous_mode(False)
        checkpoint("TM: AFTER set_synchronous_mode(False)")
    except RuntimeError as exc:
        checkpoint(f"TM: RuntimeError {exc!r}")
        errors.append(f"TM: {exc}")
    try:
        checkpoint("WORLD: BEFORE restore settings")
        world.apply_settings(original_settings)
        checkpoint("WORLD: AFTER restore settings")
    except RuntimeError as exc:
        checkpoint(f"WORLD: RuntimeError {exc!r}")
        errors.append(f"WORLD: {exc}")
    checkpoint(f"CLEANUP END; errors={len(errors)}")
    if errors:
        raise RuntimeError("Shutdown candidate had cleanup errors: " + "; ".join(errors))


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    atexit.register(checkpoint, "PYTHON ATEXIT marker (native finalization may follow)")
    checkpoint(f"START Python={sys.version} executable={sys.executable}")
    checkpoint("BEFORE import carla")

try:
    import carla
except ImportError as exc:  # pragma: no cover - requires a CARLA installation
    raise SystemExit(
        "CARLA Python API is not installed in this interpreter. Install the "
        "matching wheel from PythonAPI\\carla\\dist first."
    ) from exc

if __name__ == "__main__":
    checkpoint("AFTER import carla")


ARTIFACT_STATUS = "CARLA_CLOSED_LOOP_SYNTHETIC_V2X"
MAX_CUSTOM_MESSAGE_CHARS = 100


CSV_FIELDS: Tuple[str, ...] = (
    "artifact_status",
    "map",
    "scenario",
    "seed",
    "frame",
    "beacon_sequence",
    "beacon_slot_index",
    "report_sequence",
    "beacon_tx_time_s",
    "beacon_rx_time_s",
    "report_created_time_s",
    "report_tx_time_s",
    "report_rx_time_s",
    "observation_age_s",
    "reporter_id",
    "target_id",
    "reporter_is_byzantine",
    "target_is_malicious",
    "target_attack",
    "reporter_attack",
    "collusion_fraction",
    "beacon_rx_power_dbm",
    "report_rx_power_dbm",
    "reporter_target_distance_m",
    "true_target_x_m",
    "true_target_y_m",
    "claimed_target_x_m",
    "claimed_target_y_m",
    "observed_target_x_m",
    "observed_target_y_m",
    "true_target_speed_mps",
    "claimed_target_speed_mps",
    "true_target_heading_deg",
    "claimed_target_heading_deg",
    "measurement_noise_std_m",
    "position_anchor_residual_m",
    "signed_range_residual_m",
    "physical_anchor_probability",
    "local_risk_score",
    "submitted_risk_score",
    "stated_uncertainty",
    "observation_quality",
    "signed_score_anchor_residual",
    "absolute_score_anchor_residual",
    "audit_prev_hash",
    "audit_hash",
)


@dataclass(frozen=True)
class ReceivedMessage:
    receiver_id: int
    frame: int
    timestamp: float
    power_dbm: float
    payload: Mapping[str, Any]


@dataclass
class Counters:
    beacon_rounds_scheduled: int = 0
    beacon_transmissions: int = 0
    beacon_receptions: int = 0
    malformed_beacons: int = 0
    self_beacons_ignored: int = 0
    reports_created: int = 0
    reports_transmitted: int = 0
    report_receptions_at_rsu: int = 0
    malformed_reports: int = 0
    unmatched_reports: int = 0


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def magnitude(vector: Any) -> float:
    return math.sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)


def planar_distance(a: Any, b: Any) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def rotating_sender_order(vehicle_ids: Sequence[int], round_sequence: int) -> List[int]:
    """Return a fair, deterministic TDMA order for one beacon round."""
    ordered = list(vehicle_ids)
    if not ordered:
        return []
    start = (round_sequence - 1) % len(ordered)
    return ordered[start:] + ordered[:start]


def compact_json(payload: Mapping[str, Any]) -> str:
    message = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    if len(message) > MAX_CUSTOM_MESSAGE_CHARS:
        raise ValueError(
            f"Custom V2X payload has {len(message)} characters; CARLA's limit "
            f"is {MAX_CUSTOM_MESSAGE_CHARS}: {message}"
        )
    return message


def parse_custom_message(item: Any) -> Optional[Mapping[str, Any]]:
    """Copy a CARLA CustomV2XData item into a normal Python dictionary."""
    try:
        envelope = item.get()
        raw_message = envelope.get("Message", "")
        # CARLA 0.9.16 wraps a custom string as
        # {"Message": {"Message": "<user string>"}}. Newer builds may return
        # the user string directly, so accept both representations.
        for _ in range(3):
            if not isinstance(raw_message, dict):
                break
            if "Message" in raw_message:
                raw_message = raw_message["Message"]
            elif "message" in raw_message:
                raw_message = raw_message["message"]
            else:
                return raw_message
        if isinstance(raw_message, dict):
            return raw_message
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")
        payload = json.loads(str(raw_message))
        return payload if isinstance(payload, dict) else None
    except (AttributeError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def make_sensor_callback(
    receiver_id: int,
    destination: "queue.Queue[ReceivedMessage]",
    expected_kind: str,
) -> Any:
    """Create a callback that copies data before CARLA releases the event."""

    def callback(event: Any) -> None:
        for item in event:
            payload = parse_custom_message(item)
            if payload is None or (payload.get("k") != expected_kind and not (receiver_id == -1 and payload.get("k") == "b")):
                continue
            destination.put(
                ReceivedMessage(
                    receiver_id=receiver_id,
                    frame=int(event.frame),
                    timestamp=float(event.timestamp),
                    power_dbm=float(item.power),
                    payload=payload,
                )
            )

    return callback


def set_if_supported(blueprint: Any, name: str, value: Any) -> None:
    if blueprint.has_attribute(name):
        blueprint.set_attribute(name, str(value))


def configure_v2x_blueprint(
    library: Any,
    channel_id: str,
    scenario: str,
    seed: int,
    args: argparse.Namespace,
) -> Any:
    try:
        blueprint = library.find("sensor.other.v2x_custom")
    except RuntimeError as exc:
        raise RuntimeError(
            "This CARLA server does not expose sensor.other.v2x_custom. "
            "Verify that the server and Python wheel are both version 0.9.16."
        ) from exc

    set_if_supported(blueprint, "channel_id", channel_id)
    set_if_supported(blueprint, "scenario", scenario)
    set_if_supported(blueprint, "noise_seed", seed)
    set_if_supported(blueprint, "transmit_power", args.transmit_power)
    set_if_supported(blueprint, "receiver_sensitivity", args.receiver_sensitivity)
    set_if_supported(blueprint, "frequency_ghz", args.frequency_ghz)
    set_if_supported(blueprint, "filter_distance", args.filter_distance)
    set_if_supported(blueprint, "path_loss_model", args.path_loss_model)
    set_if_supported(blueprint, "use_etsi_fading", str(args.use_etsi_fading).lower())
    return blueprint


def choose_vehicle_blueprints(library: Any) -> List[Any]:
    candidates: List[Any] = []
    for blueprint in library.filter("vehicle.*"):
        if blueprint.has_attribute("number_of_wheels"):
            try:
                if int(blueprint.get_attribute("number_of_wheels")) != 4:
                    continue
            except (TypeError, ValueError):
                continue
        candidates.append(blueprint)
    if not candidates:
        raise RuntimeError("No four-wheel vehicle blueprints are available.")
    return candidates


def spawn_vehicles(
    world: Any,
    traffic_manager: Any,
    number: int,
    rng: random.Random,
    spawn_radius: float,
) -> List[Any]:
    library = world.get_blueprint_library()
    blueprints = choose_vehicle_blueprints(library)
    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("The selected map has no vehicle spawn points.")

    # Random spawn points on large maps can be kilometres apart, producing no
    # V2X receptions in small smoke tests. Pick a reproducible anchor, then use
    # its nearest spawn points so the initial fleet is within radio range.
    rng.shuffle(spawn_points)
    anchor_location = spawn_points[0].location
    ordered_points = sorted(
        spawn_points,
        key=lambda point: planar_distance(point.location, anchor_location),
    )
    clustered_points = [
        point
        for point in ordered_points
        if planar_distance(point.location, anchor_location) <= spawn_radius
    ]
    if len(clustered_points) >= number:
        spawn_points = clustered_points
    else:
        spawn_points = ordered_points
        print(
            f"Warning: only {len(clustered_points)} spawn points lie within "
            f"{spawn_radius:.0f} m; using the nearest available points."
        )

    vehicles: List[Any] = []
    for index, transform in enumerate(spawn_points):
        if len(vehicles) >= number:
            break
        blueprint = rng.choice(blueprints)
        set_if_supported(blueprint, "role_name", f"a6_vehicle_{index}")
        if blueprint.has_attribute("color"):
            values = list(blueprint.get_attribute("color").recommended_values)
            if values:
                blueprint.set_attribute("color", rng.choice(values))
        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            vehicles.append(vehicle)

    if len(vehicles) < 3:
        raise RuntimeError(
            f"Only {len(vehicles)} vehicles could be spawned; at least 3 are required."
        )
    if len(vehicles) < number:
        print(f"Warning: requested {number} vehicles but spawned {len(vehicles)}.")
    return vehicles


def spawn_v2x_sensors(
    world: Any,
    vehicles: Sequence[Any],
    rsu_host: Any,
    args: argparse.Namespace,
    beacon_queue: "queue.Queue[ReceivedMessage]",
    rsu_queue: "queue.Queue[ReceivedMessage]",
) -> Tuple[Dict[int, Any], Any]:
    library = world.get_blueprint_library()
    vehicle_sensors: Dict[int, Any] = {}
    relative_transform = carla.Transform(carla.Location(z=1.8))

    for index, vehicle in enumerate(vehicles):
        vehicle_bp = configure_v2x_blueprint(
            library,
            args.channel,
            args.scenario,
            args.seed + index,
            args,
        )
        vehicle_sensor = world.spawn_actor(
            vehicle_bp,
            relative_transform,
            attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )
        vehicle_sensor.listen(
            make_sensor_callback(vehicle.id, beacon_queue, expected_kind="b")
        )
        vehicle_sensors[vehicle.id] = vehicle_sensor

    # CARLA 0.9.16 custom V2X sensors are unreliable without a parent actor.
    # The caller provides a dedicated, physics-disabled parked actor as the
    # fixed RSU host; it is not part of the reporter/target population.
    rsu_transform = carla.Transform(
        carla.Location(
            x=args.rsu_offset_x,
            y=args.rsu_offset_y,
            z=args.rsu_height,
        )
    )
    rsu_bp = configure_v2x_blueprint(
        library,
        args.channel,
        args.scenario,
        args.seed + 99_999,
        args,
    )
    rsu_sensor = world.spawn_actor(
        rsu_bp,
        rsu_transform,
        attach_to=rsu_host,
        attachment_type=carla.AttachmentType.Rigid,
    )
    rsu_sensor.listen(make_sensor_callback(-1, rsu_queue, expected_kind="r"))
    return vehicle_sensors, rsu_sensor


def select_fraction(
    ids: Sequence[int],
    fraction: float,
    rng: random.Random,
) -> Set[int]:
    count = int(round(len(ids) * fraction))
    if fraction > 0.0 and count == 0:
        count = 1
    return set(rng.sample(list(ids), min(count, len(ids))))


def target_claim(
    vehicle: Any,
    malicious: bool,
    attack: str,
    offset_m: float,
    rng: random.Random,
    stable_index: int = 0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    true_state = {
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "speed": magnitude(velocity),
        "heading": float(transform.rotation.yaw),
    }
    claimed = dict(true_state)

    if malicious and attack != "none":
        if attack == "position_offset":
            # Stable target-specific direction prevents all attackers sharing
            # an unrealistic identical Cartesian offset.
            angle = math.radians((stable_index * 137.508) % 360.0)
            claimed["x"] += offset_m * math.cos(angle)
            claimed["y"] += offset_m * math.sin(angle)
        elif attack == "random_offset":
            angle = rng.uniform(0.0, 2.0 * math.pi)
            radius = rng.uniform(0.5 * offset_m, 1.5 * offset_m)
            claimed["x"] += radius * math.cos(angle)
            claimed["y"] += radius * math.sin(angle)
        else:
            raise ValueError(f"Unsupported target attack: {attack}")

    return true_state, claimed


def make_local_observation(
    true_x: float,
    true_y: float,
    distance_m: float,
    rx_power_dbm: float,
    args: argparse.Namespace,
    rng: random.Random,
) -> Tuple[float, float, float, float, float]:
    """Return observed x/y, noise std, uncertainty and observation quality."""
    weak_signal_penalty = max(0.0, -rx_power_dbm - 75.0) * args.power_noise_scale
    noise_std = (
        args.base_measurement_noise
        + distance_m * args.distance_noise_scale
        + weak_signal_penalty
    )
    observed_x = true_x + rng.gauss(0.0, noise_std)
    observed_y = true_y + rng.gauss(0.0, noise_std)

    uncertainty = clamp(noise_std / (noise_std + args.uncertainty_reference_m), 0.01, 0.99)
    signal_denominator = max(1e-6, -60.0 - args.receiver_sensitivity)
    signal_quality = clamp((rx_power_dbm - args.receiver_sensitivity) / signal_denominator)
    quality = clamp(0.5 * (1.0 - uncertainty) + 0.5 * signal_quality)
    return observed_x, observed_y, noise_std, uncertainty, quality


def manipulate_report(
    local_score: float,
    target_is_malicious: bool,
    reporter_is_byzantine: bool,
    attack: str,
    sim_time: float,
    on_period_s: float,
    rng: random.Random,
    history: MutableMapping[Tuple[int, int], float],
    reporter_id: int,
    target_id: int,
) -> float:
    if not reporter_is_byzantine or attack == "none":
        submitted = local_score
    elif attack == "false_accusation":
        submitted = 0.99 if not target_is_malicious else local_score
    elif attack == "false_reassurance":
        submitted = 0.01 if target_is_malicious else local_score
    elif attack == "coordinated":
        submitted = 0.01 if target_is_malicious else 0.99
    elif attack == "random_score":
        submitted = rng.random()
    elif attack == "on_off":
        active = int(sim_time / max(on_period_s, 0.1)) % 2 == 1
        submitted = (0.01 if target_is_malicious else 0.99) if active else local_score
    elif attack == "replay":
        submitted = history.get((reporter_id, target_id), local_score)
    else:
        raise ValueError(f"Unsupported reporter attack: {attack}")

    submitted = clamp(float(submitted))
    if attack != "replay" or (reporter_id, target_id) not in history:
        history[(reporter_id, target_id)] = local_score
    return submitted


def drain_nowait(source: "queue.Queue[ReceivedMessage]") -> Iterable[ReceivedMessage]:
    while True:
        try:
            yield source.get_nowait()
        except queue.Empty:
            return


def add_audit_hash(rows: List[Dict[str, Any]]) -> None:
    """Add a deterministic hash chain; this is not a digital signature."""
    previous = "0" * 64
    for row in rows:
        row["audit_prev_hash"] = previous
        canonical = json.dumps(
            {key: row.get(key, "") for key in CSV_FIELDS if not key.startswith("audit_")},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        current = hashlib.sha256((previous + canonical).encode("utf-8")).hexdigest()
        row["audit_hash"] = current
        previous = current


def write_outputs(
    rows: List[Dict[str, Any]],
    counters: Counters,
    output_path: Path,
    args: argparse.Namespace,
    map_name: str,
    malicious_targets: Set[int],
    byzantine_reporters: Set[int],
    participant_vehicle_ids: Sequence[int],
    elapsed_wall_s: float,
    client_version: str,
    server_version: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    add_audit_hash(rows)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "artifact_status": ARTIFACT_STATUS,
        "description": (
            "CARLA closed-loop synthetic V2X reporter-target events. "
            "Not real-world and not VeReMi data."
        ),
        "carla_client_version": client_version,
        "carla_server_version": server_version,
        "map": map_name,
        "scenario": args.scenario,
        "seed": args.seed,
        "fixed_delta_seconds": args.fixed_delta,
        "duration_seconds": args.duration,
        "warmup_seconds": args.warmup,
        "beacon_rate_hz_per_target": args.beacon_rate,
        "initial_spawn_radius_m": args.spawn_radius,
        "participant_vehicle_ids": list(participant_vehicle_ids),
        "num_participant_vehicles": len(participant_vehicle_ids),
        "num_rows": len(rows),
        "malicious_target_ids": sorted(malicious_targets),
        "byzantine_reporter_ids": sorted(byzantine_reporters),
        "target_attack": args.target_attack,
        "reporter_attack": args.reporter_attack,
        "target_malicious_fraction_requested": args.target_malicious_fraction,
        "collusion_fraction_requested": args.collusion_fraction,
        "beacon_schedule": {
            "mode": "rotating_tdma_one_sender_per_tick",
            "senders_per_tick": 1,
            "slot_index_base": 1,
            "slot_duration_seconds": args.fixed_delta,
            "round_period_seconds": 1.0 / args.beacon_rate,
            "first_sender_rotates_each_round": True,
            "v2x_sensors_per_participant": 1,
            "maximum_sends_per_participant_per_tick": 1,
        },
        "wireless": {
            "path_loss_model": args.path_loss_model,
            "use_etsi_fading": args.use_etsi_fading,
            "transmit_power_dbm": args.transmit_power,
            "receiver_sensitivity_dbm": args.receiver_sensitivity,
            "frequency_ghz": args.frequency_ghz,
            "filter_distance_m": args.filter_distance,
            "rsu_height_m": args.rsu_height,
            "rsu_offset_x_m": args.rsu_offset_x,
            "rsu_offset_y_m": args.rsu_offset_y,
        },
        "attack_model": {
            "target_position_offset_m": args.target_position_offset,
            "on_period_seconds": args.on_period,
        },
        "measurement_model": {
            "base_measurement_noise_m": args.base_measurement_noise,
            "distance_noise_scale": args.distance_noise_scale,
            "power_noise_scale": args.power_noise_scale,
            "risk_threshold_m": args.risk_threshold,
            "risk_scale_m": args.risk_scale,
        },
        "counters": vars(counters),
        "elapsed_wall_seconds": elapsed_wall_s,
        "audit": (
            "CSV rows form a SHA-256 hash chain. This detects later row/order "
            "changes but is not a cryptographic identity signature."
        ),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def retry_rpc(
    action: Callable[[], Any],
    label: str,
    attempts: int,
    delay_seconds: float,
) -> Any:
    """Retry a short CARLA/Traffic Manager RPC during map initialization."""
    last_error: Optional[RuntimeError] = None
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except RuntimeError as exc:
            last_error = exc
            if attempt >= attempts:
                break
            print(
                f"{label} was not ready (attempt {attempt}/{attempts}); "
                f"retrying in {delay_seconds:.1f}s..."
            )
            time.sleep(delay_seconds)
    raise RuntimeError(
        f"{label} failed after {attempts} attempts: {last_error}. "
        "Close any old generate_traffic.py process or retry with a different "
        "--tm-port such as 8050."
    )


def validate_args(args: argparse.Namespace) -> None:
    for name in ("collusion_fraction", "target_malicious_fraction"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if args.num_vehicles < 3:
        raise SystemExit("--num-vehicles must be at least 3.")
    if args.spawn_radius <= 0.0:
        raise SystemExit("--spawn-radius must be positive.")
    if args.duration <= 0.0 or args.fixed_delta <= 0.0 or args.beacon_rate <= 0.0:
        raise SystemExit("Duration, fixed delta and beacon rate must be positive.")
    round_slot_time = args.num_vehicles * args.fixed_delta
    round_period = 1.0 / args.beacon_rate
    if round_slot_time > round_period + 1e-9:
        raise SystemExit(
            "The rotating beacon schedule is oversubscribed: "
            f"{args.num_vehicles} vehicles need {round_slot_time:.3f}s per round, "
            f"but --beacon-rate {args.beacon_rate:g} provides only "
            f"{round_period:.3f}s. Reduce --beacon-rate or --fixed-delta."
        )
    if args.rpc_timeout <= 0.0 or args.map_settle_time < 0.0:
        raise SystemExit("RPC timeout must be positive and map settle time cannot be negative.")
    if args.tm_retries < 1 or args.tm_retry_delay < 0.0:
        raise SystemExit("TM retries must be at least 1 and retry delay cannot be negative.")
    if args.target_malicious_fraction > 0.0 and args.target_attack == "none":
        print("Warning: malicious targets are labelled but --target-attack is none.")
    if args.collusion_fraction > 0.0 and args.reporter_attack == "none":
        print("Warning: Byzantine reporters are selected but --reporter-attack is none.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect receiver-specific A6 reporter-target V2X events in CARLA 0.9.16."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--rsu-range", type=float, default=150.0)
    parser.add_argument("--rsu-noise", type=float, default=2.0)
    parser.add_argument("--rsu-dropout", type=float, default=0.10)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--rpc-timeout", type=float, default=60.0)
    parser.add_argument(
        "--map-settle-time",
        type=float,
        default=5.0,
        help="Wall-clock seconds to let a newly loaded map initialize.",
    )
    parser.add_argument("--tm-retries", type=int, default=8)
    parser.add_argument("--tm-retry-delay", type=float, default=1.0)
    parser.add_argument("--map", default=None, help="Optional CARLA map, e.g. Town05.")
    parser.add_argument("--num-vehicles", type=int, default=20)
    parser.add_argument(
        "--spawn-radius",
        type=float,
        default=300.0,
        help="Initial fleet radius in metres around a reproducible anchor.",
    )
    parser.add_argument("--duration", type=float, default=60.0, help="Measured simulation seconds.")
    parser.add_argument("--warmup", type=float, default=5.0, help="Traffic warm-up seconds.")
    parser.add_argument("--fixed-delta", type=float, default=0.05)
    parser.add_argument(
        "--beacon-rate",
        type=float,
        default=1.0,
        help=(
            "Beacon rounds per second; each target transmits once per round in "
            "a rotating one-tick TDMA slot."
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("outputs/carla_a6_events.csv"))
    parser.add_argument("--no-rendering", action="store_true")

    parser.add_argument("--channel", default="a6")
    parser.add_argument("--scenario", choices=("urban", "rural", "highway"), default="urban")
    parser.add_argument("--path-loss-model", choices=("geometric", "winner"), default="geometric")
    parser.add_argument("--use-etsi-fading", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transmit-power", type=float, default=21.5)
    parser.add_argument("--receiver-sensitivity", type=float, default=-99.0)
    parser.add_argument("--frequency-ghz", type=float, default=5.9)
    parser.add_argument("--filter-distance", type=float, default=500.0)
    parser.add_argument("--rsu-height", type=float, default=5.0)
    parser.add_argument("--rsu-offset-x", type=float, default=0.0)
    parser.add_argument("--rsu-offset-y", type=float, default=0.0)

    parser.add_argument("--target-malicious-fraction", type=float, default=0.20)
    parser.add_argument(
        "--target-attack",
        choices=("none", "position_offset", "random_offset") + ATTACKS,
        default="position_offset",
    )
    parser.add_argument("--target-position-offset", type=float, default=20.0)
    parser.add_argument("--tre-offset-bound", type=float, default=50.0)
    parser.add_argument("--stop-seconds", type=float, default=5.0)
    parser.add_argument("--a4-min-start-speed", type=float, default=2.0,
                        help="Minimum actual speed in m/s of each selected A4 target before collection.")
    parser.add_argument("--a4-start-wait", type=float, default=60.0,
                        help="Maximum extra simulated warmup seconds for selected A4 targets to move.")
    parser.add_argument("--coalition-radius", type=float, default=500.0)
    parser.add_argument("--collusion-fraction", type=float, default=0.30)
    parser.add_argument(
        "--reporter-attack",
        choices=(
            "none",
            "false_accusation",
            "false_reassurance",
            "coordinated",
            "random_score",
            "on_off",
            "replay",
        ),
        default="coordinated",
    )
    parser.add_argument("--on-period", type=float, default=5.0)

    parser.add_argument("--base-measurement-noise", type=float, default=0.50)
    parser.add_argument("--distance-noise-scale", type=float, default=0.005)
    parser.add_argument("--power-noise-scale", type=float, default=0.020)
    parser.add_argument("--uncertainty-reference-m", type=float, default=5.0)
    parser.add_argument("--risk-threshold", type=float, default=4.0)
    parser.add_argument("--risk-scale", type=float, default=1.5)
    return parser


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    if any(not math.isfinite(v) or v <= 0 for v in
           (args.tre_offset_bound, args.stop_seconds, args.coalition_radius)):
        raise ValueError("New target-attack parameters must be finite and positive")
    if any(not math.isfinite(v) or v <= 0 for v in
           (args.a4_min_start_speed, args.a4_start_wait)):
        raise ValueError("A4 start speed and wait must be finite and positive")
    if args.target_attack in ATTACKS and args.target_malicious_fraction <= 0:
        raise ValueError("A1-A5 require a positive malicious-target fraction")
    if args.target_attack == "none" and args.target_malicious_fraction != 0:
        raise ValueError("Use zero malicious-target fraction with target attack none")
    if list(args.output.parent.glob(args.output.name + '*')):
        raise ValueError("Output or sidecars already exist; use a fresh prefix")
    if args.rsu_range <= 0 or args.rsu_noise < 0 or not 0 <= args.rsu_dropout <= 1:
        raise ValueError("Invalid RSU observation parameters")
    if args.output.exists():
        raise ValueError("Output already exists; choose a new output filename")
    rng = random.Random(args.seed)
    measurement_rng = random.Random(args.seed + 1_000_003)
    attack_rng = random.Random(args.seed + 2_000_003)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.rpc_timeout)
    client_version = str(client.get_client_version())
    server_version = str(client.get_server_version())
    if client_version != server_version:
        print(
            f"Warning: CARLA client ({client_version}) and server "
            f"({server_version}) versions differ."
        )
    if args.map:
        print(f"Loading {args.map}; the first load may download navigation data...")
        world = client.load_world(args.map)
        # load_world can return before Traffic Manager's navigation structures
        # are ready, particularly on the first use of a town.
        if not world.get_settings().synchronous_mode:
            print("Waiting for the loaded map to produce a world tick...")
            world.wait_for_tick(args.rpc_timeout)
        if args.map_settle_time > 0.0:
            print(f"Allowing {args.map_settle_time:.1f}s for map/navigation initialization...")
            time.sleep(args.map_settle_time)
    else:
        world = client.get_world()
    traffic_manager = client.get_trafficmanager(args.tm_port)

    original_settings = world.get_settings()
    spawned_vehicles: List[Any] = []
    vehicles: List[Any] = []
    sensors: List[Any] = []
    listening_sensors: List[Any] = []
    beacon_queue: "queue.Queue[ReceivedMessage]" = queue.Queue()
    rsu_queue: "queue.Queue[ReceivedMessage]" = queue.Queue()
    counters = Counters()
    start_wall = time.perf_counter()
    history_log = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta
        settings.no_rendering_mode = args.no_rendering
        world.apply_settings(settings)

        retry_rpc(
            lambda: traffic_manager.set_synchronous_mode(True),
            label=f"Traffic Manager on port {args.tm_port}",
            attempts=args.tm_retries,
            delay_seconds=args.tm_retry_delay,
        )
        traffic_manager.set_random_device_seed(args.seed)
        traffic_manager.global_percentage_speed_difference(0.0)

        # One additional parked actor is used as a reliable parent for the
        # fixed RSU sensor. It is excluded from all labels and reports.
        spawned_vehicles = spawn_vehicles(
            world,
            traffic_manager,
            args.num_vehicles + 1,
            rng,
            args.spawn_radius,
        )
        rsu_host = spawned_vehicles[0]
        rsu_host.set_autopilot(False, traffic_manager.get_port())
        rsu_host.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        rsu_host.set_simulate_physics(False)
        vehicles = spawned_vehicles[1:]
        vehicle_by_id = {vehicle.id: vehicle for vehicle in vehicles}
        vehicle_ids = sorted(vehicle_by_id)
        malicious_targets = select_fraction(
            vehicle_ids, args.target_malicious_fraction, random.Random(args.seed + 11)
        )
        byzantine_reporters = select_fraction(
            vehicle_ids, args.collusion_fraction, random.Random(args.seed + 22)
        )

        vehicle_sensors, rsu_sensor = spawn_v2x_sensors(
            world, vehicles, rsu_host, args, beacon_queue, rsu_queue
        )
        sensors = list(vehicle_sensors.values()) + [rsu_sensor]
        listening_sensors = list(sensors)

        print(f"Map: {world.get_map().name}")
        print(f"Vehicles: {len(vehicles)}")
        print(f"Fixed RSU host actor: {rsu_host.id} (excluded from data)")
        print(f"Initial fleet radius: {args.spawn_radius:.0f} m")
        print(f"Byzantine reporters: {sorted(byzantine_reporters)}")
        print("Warming up traffic...")

        for _ in range(max(0, int(round(args.warmup / args.fixed_delta)))):
            world.tick()

        a4_wait_seconds = 0.0
        a4_gate_speeds = {}
        if args.target_attack == "a4_eventual_stop":
            gate_start = float(world.get_snapshot().timestamp.elapsed_seconds)
            print("A4: waiting for selected targets to move before collection...", flush=True)
            while True:
                gate_snapshot = world.get_snapshot()
                a4_wait_seconds = float(gate_snapshot.timestamp.elapsed_seconds) - gate_start
                a4_gate_speeds = {}
                for actor_id in sorted(malicious_targets):
                    actor = gate_snapshot.find(actor_id)
                    if actor is None:
                        raise RuntimeError("A4 target disappeared before start")
                    a4_gate_speeds[str(actor_id)] = magnitude(actor.get_velocity())
                if all(v >= args.a4_min_start_speed for v in a4_gate_speeds.values()):
                    break
                if a4_wait_seconds >= args.a4_start_wait:
                    raise RuntimeError(f"A4 moving-start condition not met; speeds={a4_gate_speeds}. No valid run collected.")
                world.tick()
            print(f"A4 moving-start gate passed after {a4_wait_seconds:.2f}s: {a4_gate_speeds}", flush=True)
        snapshot = world.get_snapshot()
        coalition_info = None
        if args.target_attack == "a5_collusive_offset":
            positions = {}
            for actor_id in vehicle_ids:
                actor = snapshot.find(actor_id)
                if actor is None:
                    raise ValueError("Missing actor before coalition selection")
                location = actor.get_transform().location
                positions[actor_id] = (float(location.x), float(location.y))
            malicious_targets, coalition_info = select_coalition(
                positions, args.target_malicious_fraction, args.seed, args.coalition_radius)
        print(f"Malicious targets: {sorted(malicious_targets)}")
        attack_engine = TargetAttackEngine(args.target_attack, args.seed, vehicle_ids,
            malicious_targets, args.tre_offset_bound, args.stop_seconds) if args.target_attack in ATTACKS else None
        target_attack_trace = []
        observation_log = ObservationLogger(args.output, args.seed)
        history_log = HistoryLogger(args.output, args.seed)
        history_log.self_states(snapshot, vehicles)
        start_sim = float(snapshot.timestamp.elapsed_seconds)
        end_sim = start_sim + args.duration
        next_beacon_time = start_sim
        actor_losses = {}
        def record_loss(actor_id, role, now):
            key = str(actor_id) + ":" + role
            if key not in actor_losses:
                actor_losses[key] = {"actor_id":actor_id,"role":role,"time_s":now}
                print("[ACTOR LOSS]", actor_losses[key], flush=True)

        def send_if_alive(actor_id, payload, now):
            try:
                vehicle_sensors[actor_id].send(payload)
                return True
            except RuntimeError as exc:
                if "destroyed actor" not in str(exc):
                    raise
                record_loss(actor_id, "vehicle_or_sensor", now)
                return False

        beacon_sequence = 0
        report_sequence = 0

        # Truth is captured at transmission time and joined by (target, sequence).
        beacon_truth: Dict[Tuple[int, int], Dict[str, Any]] = {}
        rsu_samples = {}
        rsu_evidence = {}
        received_reports = []
        beacon_events = []
        beacon_outbox: Deque[Tuple[int, int, int]] = deque()
        pending_reports: Dict[int, Dict[str, Any]] = {}
        report_outboxes: Dict[int, Deque[Tuple[int, str]]] = {
            reporter_id: deque() for reporter_id in vehicle_ids
        }
        reporter_history: Dict[Tuple[int, int], float] = {}
        rows: List[Dict[str, Any]] = []

        def schedule_beacon_round() -> None:
            """Queue every target once, rotating the first TDMA slot by round."""
            nonlocal beacon_sequence
            beacon_sequence += 1
            counters.beacon_rounds_scheduled += 1
            sender_order = rotating_sender_order(vehicle_ids, beacon_sequence)
            for slot_index, target_id in enumerate(sender_order, start=1):
                beacon_outbox.append((beacon_sequence, slot_index, target_id))

        def transmit_next_beacon(current_sim_time: float) -> Optional[int]:
            """Transmit at most one target beacon in a simulation tick."""
            if not beacon_outbox:
                return
            sequence, slot_index, target_id = beacon_outbox.popleft()
            vehicle = world.get_snapshot().find(target_id)
            if vehicle is None:
                record_loss(target_id, "target", current_sim_time)
                return None
            true_state, claimed = target_claim(
                vehicle,
                malicious=target_id in malicious_targets,
                attack="none" if attack_engine is not None else args.target_attack,
                offset_m=args.target_position_offset,
                rng=attack_rng,
                stable_index=vehicle_ids.index(target_id),
            )
            if attack_engine is not None:
                if (args.target_attack == "a4_eventual_stop" and target_id in malicious_targets
                        and target_id not in attack_engine.initial
                        and true_state["speed"] < args.a4_min_start_speed):
                    raise RuntimeError(f"A4 target {target_id} slowed below the minimum before its first beacon; run invalid.")
                claimed = attack_engine.claim(target_id, sequence, current_sim_time, true_state)
            # Simulator-only sensor adapter. Its random stream is independent
            # of reporter observations. This is NOT camera/LiDAR perception.
            rsu_location = rsu_sensor.get_location()
            distance = math.hypot(true_state['x']-rsu_location.x, true_state['y']-rsu_location.y)
            sensor_rng = random.Random(f"rsu:{args.seed}:{sequence}:{vehicle_ids.index(target_id)}")
            visible = distance <= args.rsu_range and sensor_rng.random() >= args.rsu_dropout
            rsu_samples[(target_id, sequence)] = {
                'observed_x': true_state['x'] + sensor_rng.gauss(0,args.rsu_noise) if visible else None,
                'observed_y': true_state['y'] + sensor_rng.gauss(0,args.rsu_noise) if visible else None,
                'sample_time': current_sim_time, 'visible': visible,
            }
            beacon_events.append({'beacon_sequence': sequence, 'target_id': target_id,
                'target_is_malicious': int(target_id in malicious_targets),
                'beacon_tx_time_s': current_sim_time})
            beacon_truth[(target_id, sequence)] = {
                **true_state,
                "timestamp": current_sim_time,
                "slot_index": slot_index,
            }
            payload = compact_json(
                {
                    "k": "b",
                    "i": target_id,
                    "n": sequence,
                    "x": round(claimed["x"], 2),
                    "y": round(claimed["y"], 2),
                    "v": round(claimed["speed"], 2),
                    "h": round(claimed["heading"], 1),
                }
            )
            if not send_if_alive(target_id, payload, current_sim_time):
                beacon_events.pop()
                beacon_truth.pop((target_id, sequence), None)
                rsu_samples.pop((target_id, sequence), None)
                return None
            if attack_engine is not None:
                wire = json.loads(payload)
                target_attack_trace.append(dict(target_id=target_id, beacon_sequence=sequence,
                    time_s=current_sim_time, true_state=true_state, claimed_state=claimed,
                    wire_state={"x":wire["x"],"y":wire["y"],"speed":wire["v"],"heading":wire["h"]},
                    malicious=target_id in malicious_targets))
            counters.beacon_transmissions += 1
            return target_id

        def process_rsu_receptions() -> None:
            for received in drain_nowait(rsu_queue):
                if received.payload.get('k') == 'b':
                    key = (int(received.payload['i']), int(received.payload['n']))
                    sample = rsu_samples.get(key)
                    if sample and sample['visible']:
                        residual = math.hypot(float(received.payload['x'])-sample['observed_x'],
                                              float(received.payload['y'])-sample['observed_y'])
                        rsu_evidence.setdefault(key, {'target_id': key[0], 'beacon_sequence': key[1],
                            'sample_time_s': sample['sample_time'], 'available_time_s': received.timestamp,
                            'rsu_score': sigmoid((residual-args.risk_threshold)/max(args.risk_scale,1e-6))})
                    continue
                counters.report_receptions_at_rsu += 1
                try:
                    report_id = int(received.payload["q"])
                except (KeyError, TypeError, ValueError):
                    counters.malformed_reports += 1
                    continue
                row = pending_reports.pop(report_id, None)
                if row is None:
                    counters.unmatched_reports += 1
                    continue
                row["frame"] = received.frame
                row["report_rx_time_s"] = round(received.timestamp, 6)
                row["report_rx_power_dbm"] = round(received.power_dbm, 6)
                # Operational inputs come from the wire, not hidden local risk.
                received_reports.append({'report_id': report_id,
                    'reporter_id': int(received.payload['r']), 'target_id': int(received.payload['t']),
                    'beacon_sequence': int(received.payload['n']),
                    'submitted_score': float(received.payload['p']),
                    'received_time_s': received.timestamp})
                rows.append(row)

        def transmit_queued_reports(
            current_sim_time: float,
            beacon_sender_id: Optional[int],
        ) -> None:
            """Transmit at most one message per vehicle and simulation tick."""
            for reporter_id, outbox in report_outboxes.items():
                if not outbox or reporter_id == beacon_sender_id:
                    continue
                report_id, report_payload = outbox.popleft()
                if not send_if_alive(reporter_id, report_payload, current_sim_time):
                    continue
                counters.reports_transmitted += 1
                pending_reports[report_id]["report_tx_time_s"] = round(
                    current_sim_time, 6
                )

        def process_beacon_receptions(current_sim_time: float) -> None:
            nonlocal report_sequence
            history_log.self_states(world.get_snapshot(), vehicles)
            for received in drain_nowait(beacon_queue):
                history_log.beacon(received, current_sim_time)
                counters.beacon_receptions += 1
                try:
                    target_id = int(received.payload["i"])
                    sequence = int(received.payload["n"])
                    claimed_x = float(received.payload["x"])
                    claimed_y = float(received.payload["y"])
                    claimed_speed = float(received.payload["v"])
                    claimed_heading = float(received.payload["h"])
                except (KeyError, TypeError, ValueError):
                    counters.malformed_beacons += 1
                    continue

                reporter_id = received.receiver_id
                if reporter_id == target_id:
                    counters.self_beacons_ignored += 1
                    continue
                truth = beacon_truth.get((target_id, sequence))
                reporter = vehicle_by_id.get(reporter_id)
                if truth is None or reporter is None or reporter_id not in vehicle_sensors:
                    counters.malformed_beacons += 1
                    continue

                observer_snapshot = world.get_snapshot().find(reporter_id)
                if observer_snapshot is None:
                    record_loss(reporter_id, "reporter", current_sim_time)
                    continue
                reporter_location = observer_snapshot.get_transform().location
                true_location = carla.Location(x=truth["x"], y=truth["y"], z=reporter_location.z)
                distance_m = planar_distance(reporter_location, true_location)
                observed_x, observed_y, noise_std, uncertainty, quality = make_local_observation(
                    truth["x"],
                    truth["y"],
                    distance_m,
                    received.power_dbm,
                    args,
                    measurement_rng,
                )
                position_residual = math.hypot(claimed_x - observed_x, claimed_y - observed_y)
                local_score = sigmoid(
                    (position_residual - args.risk_threshold) / max(args.risk_scale, 1e-6)
                )
                target_is_malicious = target_id in malicious_targets
                reporter_is_byzantine = reporter_id in byzantine_reporters
                submitted_score = manipulate_report(
                    local_score=local_score,
                    target_is_malicious=target_is_malicious,
                    reporter_is_byzantine=reporter_is_byzantine,
                    attack=args.reporter_attack,
                    sim_time=current_sim_time - start_sim,
                    on_period_s=args.on_period,
                    rng=attack_rng,
                    history=reporter_history,
                    reporter_id=reporter_id,
                    target_id=target_id,
                )

                true_range = math.hypot(truth["x"] - reporter_location.x, truth["y"] - reporter_location.y)
                claimed_range = math.hypot(claimed_x - reporter_location.x, claimed_y - reporter_location.y)
                signed_range_residual = claimed_range - true_range
                report_sequence += 1
                observation_log.record(
                    report_sequence, reporter_id, target_id, sequence,
                    current_sim_time, float(truth["timestamp"]),
                    claimed_x, claimed_y, observed_x, observed_y, noise_std)
                history_log.attack(report_sequence, reporter_id, target_id, sequence,
                    current_sim_time, current_sim_time-start_sim, args.reporter_attack,
                    reporter_is_byzantine, target_is_malicious, args.on_period,
                    local_score, submitted_score)
                report_payload = compact_json(
                    {
                        "k": "r",
                        "q": report_sequence,
                        "n": sequence,
                        "r": reporter_id,
                        "t": target_id,
                        "p": round(submitted_score, 4),
                        "u": round(uncertainty, 4),
                    }
                )
                counters.reports_created += 1

                score_residual = submitted_score - local_score
                pending_reports[report_sequence] = {
                    "artifact_status": ARTIFACT_STATUS,
                    "map": world.get_map().name,
                    "scenario": args.scenario,
                    "seed": args.seed,
                    "frame": "",
                    "beacon_sequence": sequence,
                    "beacon_slot_index": int(truth["slot_index"]),
                    "report_sequence": report_sequence,
                    "beacon_tx_time_s": round(float(truth["timestamp"]), 6),
                    "beacon_rx_time_s": round(received.timestamp, 6),
                    "report_created_time_s": round(current_sim_time, 6),
                    "report_tx_time_s": "",
                    "report_rx_time_s": "",
                    "observation_age_s": round(
                        max(0.0, received.timestamp - float(truth["timestamp"])), 6
                    ),
                    "reporter_id": reporter_id,
                    "target_id": target_id,
                    "reporter_is_byzantine": int(reporter_is_byzantine),
                    "target_is_malicious": int(target_is_malicious),
                    "target_attack": args.target_attack if target_is_malicious else "none",
                    "reporter_attack": args.reporter_attack if reporter_is_byzantine else "none",
                    "collusion_fraction": args.collusion_fraction,
                    "beacon_rx_power_dbm": round(received.power_dbm, 6),
                    "report_rx_power_dbm": "",
                    "reporter_target_distance_m": round(distance_m, 6),
                    "true_target_x_m": round(float(truth["x"]), 6),
                    "true_target_y_m": round(float(truth["y"]), 6),
                    "claimed_target_x_m": round(claimed_x, 6),
                    "claimed_target_y_m": round(claimed_y, 6),
                    "observed_target_x_m": round(observed_x, 6),
                    "observed_target_y_m": round(observed_y, 6),
                    "true_target_speed_mps": round(float(truth["speed"]), 6),
                    "claimed_target_speed_mps": round(claimed_speed, 6),
                    "true_target_heading_deg": round(float(truth["heading"]), 6),
                    "claimed_target_heading_deg": round(claimed_heading, 6),
                    "measurement_noise_std_m": round(noise_std, 6),
                    "position_anchor_residual_m": round(position_residual, 6),
                    "signed_range_residual_m": round(signed_range_residual, 6),
                    "physical_anchor_probability": round(local_score, 8),
                    "local_risk_score": round(local_score, 8),
                    "submitted_risk_score": round(submitted_score, 8),
                    "stated_uncertainty": round(uncertainty, 8),
                    "observation_quality": round(quality, 8),
                    "signed_score_anchor_residual": round(score_residual, 8),
                    "absolute_score_anchor_residual": round(abs(score_residual), 8),
                    "audit_prev_hash": "",
                    "audit_hash": "",
                }
                report_outboxes[reporter_id].append(
                    (report_sequence, report_payload)
                )

        print("Collecting V2X events. Press Ctrl+C for an early, clean save.")
        interrupted = False
        try:
            while True:
                frame = world.tick()
                snapshot = world.get_snapshot()
                sim_time = float(snapshot.timestamp.elapsed_seconds)
                process_rsu_receptions()
                process_beacon_receptions(sim_time)

                if sim_time >= next_beacon_time and sim_time < end_sim:
                    schedule_beacon_round()
                    next_beacon_time += 1.0 / args.beacon_rate
                beacon_sender_id = transmit_next_beacon(sim_time)
                transmit_queued_reports(sim_time, beacon_sender_id)

                if sim_time >= end_sim:
                    break
        except KeyboardInterrupt:
            interrupted = True
            print("Interrupted; flushing messages already in flight...")

        # Drain any slots from the final beacon round. A received beacon creates
        # at most one new report per reporter, so the per-reporter outboxes can
        # keep pace while the remaining target slots are transmitted.
        for _ in range(len(vehicles) + 3):
            world.tick()
            snapshot = world.get_snapshot()
            sim_time = float(snapshot.timestamp.elapsed_seconds)
            process_rsu_receptions()
            process_beacon_receptions(sim_time)
            beacon_sender_id = transmit_next_beacon(sim_time)
            transmit_queued_reports(sim_time, beacon_sender_id)
        for _ in range(3):
            world.tick()
            process_rsu_receptions()

        # Allow asynchronous callbacks to arrive; do not claim queue emptiness
        # proves radio delivery. Unreceived reports remain explicit in diagnostics.
        for _ in range(20):
            world.tick()
            time.sleep(0.01)
            process_rsu_receptions()
            now = float(world.get_snapshot().timestamp.elapsed_seconds)
            process_beacon_receptions(now)
            transmit_queued_reports(now, None)

        elapsed_wall = time.perf_counter() - start_wall
        write_outputs(
            rows=rows,
            counters=counters,
            output_path=args.output.resolve(),
            args=args,
            map_name=world.get_map().name,
            malicious_targets=malicious_targets,
            byzantine_reporters=byzantine_reporters,
            participant_vehicle_ids=vehicle_ids,
            elapsed_wall_s=elapsed_wall,
            client_version=client_version,
            server_version=server_version,
        )
        if attack_engine is not None:
            target_trace_path = Path(str(args.output)+".target_attacks.evaluation.jsonl")
            with target_trace_path.open("x", encoding="utf-8") as f:
                for entry in target_attack_trace:
                    f.write(json.dumps(entry, allow_nan=False)+"\n")
            import hashlib
            specification = attack_engine.describe()
            if args.target_attack == "a4_eventual_stop":
                specification["moving_start"] = dict(min_speed_mps=args.a4_min_start_speed,
                    max_extra_wait_s=args.a4_start_wait, extra_wait_s=a4_wait_seconds,
                    gate_speeds_mps=a4_gate_speeds,
                    policy="Wait for all originally selected targets; verify again at each initial beacon; no reselection")
            specification.update(coalition=coalition_info, successful_transmissions=len(target_attack_trace),
                trace_file=target_trace_path.name, trace_sha256=hashlib.sha256(target_trace_path.read_bytes()).hexdigest(),
                interrupted=interrupted or bool(actor_losses))
            Path(str(args.output)+".target_attack_manifest.json").write_text(
                json.dumps(specification, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Saved {len(rows)} RSU-received reporter-target events to:")
        operational = {'schema': 'a6-independent-rsu-v3', 'seed': args.seed,
            'sensor_model': 'synthetic range-limited noisy XY; no geometric occlusion; perfect target association',
            'rsu_range_m': args.rsu_range, 'rsu_noise_m': args.rsu_noise,
            'rsu_dropout': args.rsu_dropout,
            'evidence': list(rsu_evidence.values()), 'reports': received_reports}
        Path(str(args.output)+'.operational.json').write_text(json.dumps(operational,indent=2),encoding='utf-8')
        evaluation = {'schema': 'a6-evaluation-v3', 'interrupted': interrupted or bool(actor_losses),
            'actor_losses':list(actor_losses.values()),'collection_interrupted':interrupted,
            'scheduled_rounds': counters.beacon_rounds_scheduled, 'target_ids': vehicle_ids,
            'malicious_target_ids': sorted(malicious_targets), 'transmitted_beacons': beacon_events,
            'pending_report_ids': list(pending_reports),
            'beacon_callback_queue_remaining': beacon_queue.qsize(),
            'rsu_callback_queue_remaining': rsu_queue.qsize(),
            'report_outbox_remaining': sum(map(len,report_outboxes.values()))}
        Path(str(args.output)+'.evaluation.json').write_text(json.dumps(evaluation,indent=2),encoding='utf-8')
        print(args.output.resolve())
        print(f"Metadata: {args.output.resolve().with_suffix(args.output.suffix + '.metadata.json')}")
        print(
            "Reports created / transmitted / received at RSU: "
            f"{counters.reports_created} / {counters.reports_transmitted} / {len(rows)}"
        )
        observed_reporters = {int(row["reporter_id"]) for row in rows}
        observed_targets = {int(row["target_id"]) for row in rows}
        print(
            "Observed reporters / targets at RSU: "
            f"{len(observed_reporters)}/{len(vehicle_ids)} / "
            f"{len(observed_targets)}/{len(vehicle_ids)}"
        )
        if not rows:
            print(
                "No V2X reports were received. For a connectivity diagnostic, "
                "retry with --spawn-radius 150 --filter-distance 1000."
            )
        history_log.close()
        Path(str(args.output)+".history_manifest.json").write_text(json.dumps({
            "schema":"a6-full-history-v1", "interrupted":interrupted or bool(actor_losses),
            "actor_losses":list(actor_losses.values()), "fixed_fleet_valid":not bool(actor_losses),
            "seed":args.seed, "counts":history_log.counts,
            "files":{k:p.name for k,p in history_log.paths.items()},
            "start_sim_time_s":start_sim, "beacon_rate_requested_hz":args.beacon_rate,
            "self_state_model":"ideal simulated ego localization",
            "beacon_queue_remaining":beacon_queue.qsize(),
            "note":"Logs every beacon drained by the collector, before report filtering/transmission. Check remaining queues; not a claim of complete callback delivery."
        },indent=2),encoding="utf-8")
        observation_log.finish(args.output, interrupted or bool(actor_losses))
        print("Local observations:", observation_log.count, flush=True)
        print("Full observer history:", history_log.counts, flush=True)
        checkpoint("OUTPUTS SAVED; run preparing to return through finally")
        if actor_losses:
            print("Saved diagnostic artifacts, but fixed-fleet validation FAILED: participant loss. Exit code 4; do not train on this run.", flush=True)
        return 130 if interrupted else 4 if actor_losses else 0

    finally:
        if history_log is not None:
            history_log.close()
        print("Cleaning up CARLA actors...", flush=True)
        faulthandler.dump_traceback_later(30, repeat=True)
        diagnostic_cleanup(listening_sensors, sensors, spawned_vehicles,
                           traffic_manager, world, original_settings)
        checkpoint("RUN finally END; native local references may be released next")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        checkpoint("MAIN: BEFORE run")
        result = run(args)
        checkpoint(f"MAIN: AFTER run result={result}")
        return result
    except RuntimeError as exc:
        traceback.print_exc()
        print(f"CARLA error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    exit_code = main()
    faulthandler.cancel_dump_traceback_later()
    checkpoint(f"BEFORE SystemExit({exit_code})")
    raise SystemExit(exit_code)
