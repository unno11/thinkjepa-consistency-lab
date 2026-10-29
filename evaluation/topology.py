"""Frozen 52-joint hand topology used by the evaluation protocol.

This module intentionally has no PyTorch or training-code dependency.  Its joint
order matches ``cache_train.hand_skeleton_consts.QUERY_TFS`` at integration time.
"""

FINGER_JOINTS = ("Metacarpal", "Knuckle", "IntermediateBase", "IntermediateTip", "Tip")
REGULAR_FINGERS = ("LittleFinger", "RingFinger", "MiddleFinger", "IndexFinger")
THUMB_JOINTS = ("Knuckle", "IntermediateBase", "IntermediateTip", "Tip")


def _side_joints(side):
    result = []
    for finger in REGULAR_FINGERS:
        result.extend(f"{side}{finger}{joint}" for joint in FINGER_JOINTS)
    result.extend(f"{side}Thumb{joint}" for joint in THUMB_JOINTS)
    return result


JOINT_NAMES = tuple(_side_joints("right") + ["rightHand", "rightForearm"] +
                    _side_joints("left") + ["leftHand", "leftForearm"])
JOINT_INDEX = {name: index for index, name in enumerate(JOINT_NAMES)}


def _build_topology():
    finger_internal = []
    palm_to_finger = []
    forearm_to_hand = []
    angles = []
    for side in ("right", "left"):
        hand, forearm = f"{side}Hand", f"{side}Forearm"
        forearm_to_hand.append((forearm, hand))
        for finger in REGULAR_FINGERS:
            chain = [f"{side}{finger}{joint}" for joint in FINGER_JOINTS]
            palm_to_finger.append((hand, chain[0]))
            finger_internal.extend(zip(chain[:-1], chain[1:]))
            angles.extend((chain[i - 1], chain[i], chain[i + 1]) for i in range(1, len(chain) - 1))
            angles.append((hand, chain[0], chain[1]))
        thumb = [f"{side}Thumb{joint}" for joint in THUMB_JOINTS]
        palm_to_finger.append((hand, thumb[0]))
        finger_internal.extend(zip(thumb[:-1], thumb[1:]))
        angles.extend((thumb[i - 1], thumb[i], thumb[i + 1]) for i in range(1, len(thumb) - 1))
        angles.append((hand, thumb[0], thumb[1]))
        roots = [f"{side}{finger}Metacarpal" for finger in REGULAR_FINGERS] + [thumb[0]]
        angles.extend((forearm, hand, root) for root in roots)
    return finger_internal, palm_to_finger, forearm_to_hand, angles


def _indices(items):
    return tuple(tuple(JOINT_INDEX[name] for name in item) for item in items)


_FINGER, _PALM, _FOREARM, _ANGLES = _build_topology()
FINGER_BONES = _indices(_FINGER)
PALM_BONES = _indices(_PALM)
FOREARM_BONES = _indices(_FOREARM)
ALL_BONES = FINGER_BONES + PALM_BONES + FOREARM_BONES
# Parent-to-child traversal for the two kinematic trees.  ``ALL_BONES`` keeps
# the historical metric order; projection needs roots before descendants.
KINEMATIC_TREE_EDGES = FOREARM_BONES + PALM_BONES + FINGER_BONES
ANGLE_TRIPLETS = _indices(_ANGLES)
RIGHT_HAND_INDEX = JOINT_INDEX["rightHand"]
LEFT_HAND_INDEX = JOINT_INDEX["leftHand"]
FINGER_JOINT_INDICES = tuple(
    index for index, name in enumerate(JOINT_NAMES)
    if name not in {"rightHand", "rightForearm", "leftHand", "leftForearm"}
)

assert len(JOINT_NAMES) == 52
assert (len(FINGER_BONES), len(PALM_BONES), len(FOREARM_BONES), len(ALL_BONES)) == (38, 10, 2, 50)
assert set(KINEMATIC_TREE_EDGES) == set(ALL_BONES)
assert len(ANGLE_TRIPLETS) == 48
assert len(FINGER_JOINT_INDICES) == 48
