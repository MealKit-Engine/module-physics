#pragma once

#include "mealkit/core/Hash.h"
#include "mealkit/core/Math.h"

#include <string>
#include <vector>

// Jolt Physics wrapper. Exposes only a handle-based (BodyId) API so Jolt headers don't spread into game code.
//
// Determinism: Jolt is built with CROSS_PLATFORM_DETERMINISTIC, so the same calls in the same order
// give the same results regardless of CPU or thread count. So body create/delete/force application order is kept identical on all peers.
namespace mealkit::physics {

using BodyId = u32;
inline constexpr BodyId kInvalidBody = 0xFFFFFFFFu;

enum class Motion : u8 { Static, Dynamic, Kinematic };

struct BodyDesc {
    Motion motion = Motion::Dynamic;
    Vec3 position;
    Quat rotation;
    f32 friction = 0.5f;
    f32 restitution = 0.1f;
    f32 linearDamping = 0.05f;
    f32 angularDamping = 0.05f;
    f32 mass = 0.0f;           // 0 = computed from shape and density (1000 kg/m³)
    bool lockRotation = false; // for characters: prevents rotation so they don't fall over
    bool allowSleep = true;
    u32 group = 0;             // bodies with the same nonzero group don't collide with each other (ragdoll parts)
};

// Kinematic character (Jolt CharacterVirtual): moved by sweeping its capsule, walks stairs and slopes and sticks to the floor.
// It isn't a rigid body, so a kinematic capsule body of the same size follows it (other bodies, rays and contacts see that body).
// Its state goes into hashState, writeState and snapshots like bodies. Created/moved in a fixed order like bodies.
using CharacterId = u32;
inline constexpr CharacterId kInvalidCharacter = 0xFFFFFFFFu;

struct CharacterDesc {
    Vec3 position;            // capsule center
    f32 halfHeight = 0.45f;   // half of the cylinder part
    f32 radius = 0.45f;
    f32 mass = 80.0f;         // how hard it pushes dynamic bodies
    f32 maxStrength = 500.0f; // max push force (N)
    f32 maxSlopeCos = 0.64f;  // cos of the steepest walkable slope (no trig here: callers use detCos)
};

struct CharacterGround {
    bool supported = false; // standing on walkable ground
    Vec3 normal;
    Vec3 velocity; // velocity of what it stands on (moving platforms)
};

// Motorized joint between two bodies (Jolt SwingTwistConstraint): a ball joint with a cone and twist limit,
// whose motor springs the child toward a target rotation relative to the parent (active ragdolls, sim/Ragdoll.h).
// Joints (targets, torques, Jolt solver state) are in hashState and snapshots. Created/destroyed in a fixed order like bodies.
using JointId = u32;
inline constexpr JointId kInvalidJoint = 0xFFFFFFFFu;

struct JointDesc {
    Vec3 pivot;                     // world position at creation
    Vec3 twistAxis{0.0f, -1.0f, 0.0f}; // world, along the child bone (from parent toward child)
    Vec3 planeAxis{1.0f, 0.0f, 0.0f};  // world, perpendicular to twistAxis
    f32 normalCone = 0.8f;          // half cone angles and twist range (radians)
    f32 planeCone = 0.8f;
    f32 twistMin = -0.5f;
    f32 twistMax = 0.5f;
    f32 motorFrequency = 6.0f;      // motor spring (Hz) and damping ratio
    f32 motorDamping = 1.0f;
    f32 maxTorque = 400.0f;         // N·m, 0 = limp
};

struct RayHit {
    BodyId body = kInvalidBody;
    f32 fraction = 1.0f; // 0..1, fraction of the ray length
};

// Newly touching body pairs (for sound/effects). Not state but material for view notifications: not in the hash/snapshots.
struct ContactEvent {
    BodyId a = kInvalidBody; // a < b
    BodyId b = kInvalidBody;
    Vec3 point;      // contact point (world)
    f32 speed = 0.0f; // approach speed along the normal at the moment of contact (m/s)
};

// Snapshot of the whole physics state (body list, shapes, velocities, contact cache). Used only within the same process (rewind, shadow simulation).
class PhysicsSnapshot {
public:
    PhysicsSnapshot();
    ~PhysicsSnapshot();
    PhysicsSnapshot(PhysicsSnapshot&& other) noexcept;
    PhysicsSnapshot& operator=(PhysicsSnapshot&& other) noexcept;
    PhysicsSnapshot(const PhysicsSnapshot&) = delete;
    PhysicsSnapshot& operator=(const PhysicsSnapshot&) = delete;

    bool empty() const;
    size_t byteSize() const; // rough memory usage
    // Bytes to send to another process (including body shapes, between identical builds): late join/reconnect (game/WorldSnapshot.h)
    void write(std::vector<u8>& out) const;
    bool read(const u8* data, size_t size);

private:
    friend class PhysicsWorld;
    struct Impl;
    Impl* m = nullptr;
};

class PhysicsWorld {
public:
    PhysicsWorld() = default;
    ~PhysicsWorld();
    PhysicsWorld(const PhysicsWorld&) = delete;
    PhysicsWorld& operator=(const PhysicsWorld&) = delete;

    // threads = 0 matches the core count. The thread count doesn't affect determinism.
    bool init(u32 maxBodies = 16384, u32 threads = 0);
    void shutdown();
    bool initialized() const { return m != nullptr; }

    void setGravity(Vec3 gravity);
    BodyId createBox(Vec3 halfExtents, const BodyDesc& desc);
    BodyId createSphere(f32 radius, const BodyDesc& desc);
    BodyId createCapsule(f32 halfHeight, f32 radius, const BodyDesc& desc); // halfHeight = half of the cylinder part
    // Convex hull. For point sets that can't form one, a bounding box is used instead.
    BodyId createConvexHull(const Vec3* points, u32 count, const BodyDesc& desc);
    void destroy(BodyId body);

    CharacterId createCharacter(const CharacterDesc& desc);
    void destroyCharacter(CharacterId character);
    // Moves the character with this velocity over dt: slides along walls, climbs steps up to stepUp, sticks to floors within stickDown.
    // Call before step() in a fixed order. Its body follows during step().
    void moveCharacter(CharacterId character, Vec3 velocity, Vec3 gravity, f32 dt, f32 stepUp, f32 stickDown);
    Vec3 characterPosition(CharacterId character) const;
    Vec3 characterVelocity(CharacterId character) const;
    void setCharacterVelocity(CharacterId character, Vec3 velocity);
    CharacterGround characterGround(CharacterId character) const;
    BodyId characterBody(CharacterId character) const; // the kinematic body that follows it

    // Joints: parent and child must exist. Destroy joints before their bodies.
    JointId createJoint(BodyId parent, BodyId child, const JointDesc& desc);
    void destroyJoint(JointId joint);
    // Motor target: child rotation relative to the parent (child = parent * target, identity = same rotation as the parent)
    void setJointTarget(JointId joint, Quat target);
    void setJointTorque(JointId joint, f32 maxTorque); // 0 = limp (limits still hold)
    u32 jointCount() const;
    // Calling once after creating all static bodies speeds up the broad phase.
    void optimize();

    void step(f32 dt, int collisionSteps = 1);

    // Contact events: collects pairs that newly touched at minSpeed (m/s) or more every step(). 0 turns it off (default, no cost).
    // Physics worker threads collect them and at the end of step() they're sorted by strength (then body index) keeping maxPerStep → the same regardless of thread count.
    void setContactEvents(f32 minSpeed, u32 maxPerStep);
    const std::vector<ContactEvent>& contacts() const; // from the last step()

    Vec3 position(BodyId body) const;
    Quat rotation(BodyId body) const;
    Vec3 linearVelocity(BodyId body) const;
    Vec3 angularVelocity(BodyId body) const;
    void setLinearVelocity(BodyId body, Vec3 velocity);
    void setAngularVelocity(BodyId body, Vec3 velocity);
    void addLinearVelocity(BodyId body, Vec3 delta); // velocity change independent of mass
    void setTransform(BodyId body, Vec3 position, Quat rotation);
    bool isActive(BodyId body) const;

    // Ray: from origin along direction (length = max distance). The ignore body and bodies of ignoreGroup (if nonzero) are ignored.
    bool castRay(Vec3 origin, Vec3 direction, RayHit& hit, BodyId ignore = kInvalidBody, u32 ignoreGroup = 0) const;

    // Adds all body states to the hash in creation order (for desync detection).
    void hashState(Hasher& hasher) const;
    // Values put into the hash, one line per body (desync dump, game/DesyncDump.h): "body <seq> <id> p x y z q x y z w v x y z w x y z".
    // Floats as %.9g (reads back the float exactly)
    void writeState(std::string& out) const;
    // For tests: changes the velocity of the first created moving body very slightly (--desync-test, causes a desync on purpose)
    void nudgeForTest();

    // Snapshot save/restore. Restoring also reverts bodies created or deleted in between (with identical body ids).
    // Body ids are assigned in an engine-defined order (deleted numbers are reused in a fixed order), so bodies created after a restore get the same ids as originally.
    void save(PhysicsSnapshot& out) const;
    bool restore(const PhysicsSnapshot& snapshot);
    // Deletes all bodies and resets id allocation (before restoring another process's snapshot: the same id may have a body with a different shape)
    void clearBodies();
    u32 bodyCount() const;
    u32 activeBodyCount() const;
    u32 threadCount() const;

private:
    struct Impl;
    Impl* m = nullptr;
};

} // namespace mealkit::physics
