#include "mealkit/physics/PhysicsWorld.h"

#include "mealkit/core/Log.h"
#include "mealkit/core/Profile.h"
#include "mealkit/jobs/Jobs.h"
#include "mealkit/jobs/JoltAllocator.h"

#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/ContactListener.h>
#include <Jolt/Physics/Collision/GroupFilter.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/HeightFieldShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Constraints/SwingTwistConstraint.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/StateRecorder.h>
#include <Jolt/RegisterTypes.h>

#include <algorithm>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

namespace mealkit::physics {
namespace {

namespace Layers {
constexpr JPH::ObjectLayer NonMoving = 0;
constexpr JPH::ObjectLayer Moving = 1;
} // namespace Layers

namespace BroadPhaseLayers {
constexpr JPH::BroadPhaseLayer NonMoving(0);
constexpr JPH::BroadPhaseLayer Moving(1);
constexpr JPH::uint Count = 2;
} // namespace BroadPhaseLayers

class BroadPhaseLayerMap final : public JPH::BroadPhaseLayerInterface {
public:
    JPH::uint GetNumBroadPhaseLayers() const override { return BroadPhaseLayers::Count; }
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer layer) const override {
        return layer == Layers::NonMoving ? BroadPhaseLayers::NonMoving : BroadPhaseLayers::Moving;
    }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer layer) const override {
        return layer == BroadPhaseLayers::NonMoving ? "NonMoving" : "Moving";
    }
#endif
};

class ObjectVsBroadPhaseFilter final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer layer, JPH::BroadPhaseLayer bpLayer) const override {
        return layer == Layers::Moving || bpLayer == BroadPhaseLayers::Moving;
    }
};

class ObjectPairFilter final : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer a, JPH::ObjectLayer b) const override {
        return a == Layers::Moving || b == Layers::Moving;
    }
};

void traceToLog(const char* fmt, ...) {
    char text[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(text, sizeof(text), fmt, args);
    va_end(args);
    MEALKIT_INFO("[Jolt] %s", text);
}

// Writes and reads Jolt state to/from a byte array
class ByteStateRecorder final : public JPH::StateRecorder {
public:
    explicit ByteStateRecorder(std::vector<u8>& data) : m_data(data) {}
    void WriteBytes(const void* data, size_t size) override {
        const u8* p = static_cast<const u8*>(data);
        m_data.insert(m_data.end(), p, p + size);
    }
    void ReadBytes(void* out, size_t size) override {
        if (m_pos + size > m_data.size()) {
            m_failed = true;
            std::memset(out, 0, size);
            return;
        }
        std::memcpy(out, m_data.data() + m_pos, size);
        m_pos += size;
    }
    bool IsEOF() const override { return m_pos >= m_data.size(); }
    bool IsFailed() const override { return m_failed; }

private:
    std::vector<u8>& m_data;
    size_t m_pos = 0;
    bool m_failed = false;
};

int g_joltUsers = 0;

void acquireJolt() {
    if (g_joltUsers++ > 0) return;
    jobs::installJoltAllocator(); // shared with the job pool (jobs/JoltAllocator.h)
    JPH::Trace = traceToLog;
    JPH::Factory::sInstance = new JPH::Factory();
    JPH::RegisterTypes();
}

void releaseJolt() {
    if (--g_joltUsers > 0) return;
    JPH::UnregisterTypes();
    delete JPH::Factory::sInstance;
    JPH::Factory::sInstance = nullptr;
}

JPH::Vec3 toJ(Vec3 v) { return JPH::Vec3(v.x, v.y, v.z); }
JPH::RVec3 toJR(Vec3 v) { return JPH::RVec3(v.x, v.y, v.z); }
JPH::Quat toJ(Quat q) { return JPH::Quat(q.x, q.y, q.z, q.w).Normalized(); }
Vec3 fromJ(JPH::Vec3Arg v) { return {v.GetX(), v.GetY(), v.GetZ()}; }
Quat fromJ(JPH::QuatArg q) { return {q.GetX(), q.GetY(), q.GetZ(), q.GetW()}; }
#ifdef JPH_DOUBLE_PRECISION
Vec3 fromJ(JPH::RVec3Arg v) {
    return {static_cast<f32>(v.GetX()), static_cast<f32>(v.GetY()), static_cast<f32>(v.GetZ())};
}
#endif

// Record of created bodies: when restoring a snapshot, vanished bodies are recreated with the same id
struct BodyRecord {
    JPH::BodyID id;
    JPH::RefConst<JPH::Shape> shape;
    BodyDesc desc;
};

// Body id allocation state (goes into snapshots)
struct BodyIdAllocator {
    std::vector<u32> free; // numbers to reuse (from the back)
    std::vector<u8> sequence;
    u32 next = 0;
};

struct CharacterRecord {
    CharacterId id = kInvalidCharacter;
    CharacterDesc desc;
    BodyId body = kInvalidBody;
    JPH::Ref<JPH::CharacterVirtual> character;
};

// A character in a snapshot: Jolt CharacterVirtual::SaveState bytes (position, velocity, contacts, ground)
struct CharacterSave {
    CharacterId id = kInvalidCharacter;
    CharacterDesc desc;
    BodyId body = kInvalidBody;
    std::vector<u8> state;
};

// A joint as created (body-space frames, so it can be recreated in any pose) + its current motor values.
// POD: copied as is into snapshots
struct JointState {
    JointId id = kInvalidJoint;
    BodyId parent = kInvalidBody;
    BodyId child = kInvalidBody;
    JointDesc desc;
    Vec3 position1, twist1, plane1; // in the parent's body space
    Vec3 position2, twist2, plane2; // in the child's body space
    Quat target;
    f32 torque = 0.0f;
};

struct JointRecord {
    JointState state;
    JPH::Ref<JPH::SwingTwistConstraint> constraint;
};

// BodyDesc::group: bodies sharing a nonzero group never collide (no sub-group pairs: a ragdoll ignores itself entirely)
class SameGroupFilter final : public JPH::GroupFilter {
public:
    bool CanCollide(const JPH::CollisionGroup& a, const JPH::CollisionGroup& b) const override {
        return a.GetGroupID() == JPH::CollisionGroup::cInvalidGroup || a.GetGroupID() != b.GetGroupID();
    }
};

const JPH::GroupFilter* sameGroupFilter() {
    static JPH::Ref<SameGroupFilter> filter = new SameGroupFilter();
    return filter;
}

// Ray filter: one body and one group
class RayBodyFilter final : public JPH::BodyFilter {
public:
    RayBodyFilter(JPH::BodyID ignore, u32 group) : m_ignore(ignore), m_group(group) {}
    bool ShouldCollide(const JPH::BodyID& id) const override { return id != m_ignore; }
    bool ShouldCollideLocked(const JPH::Body& body) const override {
        return m_group == 0 || body.GetCollisionGroup().GetGroupID() != m_group;
    }

private:
    JPH::BodyID m_ignore;
    u32 m_group;
};

JPH::BodyCreationSettings makeSettings(const JPH::Shape* shape, const BodyDesc& d) {
    const JPH::EMotionType motion = d.motion == Motion::Static    ? JPH::EMotionType::Static
                                    : d.motion == Motion::Dynamic ? JPH::EMotionType::Dynamic
                                                                  : JPH::EMotionType::Kinematic;
    JPH::BodyCreationSettings s(shape, toJR(d.position), toJ(d.rotation), motion,
                                d.motion == Motion::Static ? Layers::NonMoving : Layers::Moving);
    s.mFriction = d.friction;
    s.mRestitution = d.restitution;
    s.mLinearDamping = d.linearDamping;
    s.mAngularDamping = d.angularDamping;
    s.mAllowSleeping = d.allowSleep;
    if (d.group != 0) s.mCollisionGroup = JPH::CollisionGroup(sameGroupFilter(), d.group, 0);
    if (d.motion == Motion::Dynamic && d.plane2d) // rotation locked too: only the two plane translations
        s.mAllowedDOFs = d.lockRotation ? JPH::EAllowedDOFs::TranslationX | JPH::EAllowedDOFs::TranslationY : JPH::EAllowedDOFs::Plane2D;
    else if (d.lockRotation)
        s.mAllowedDOFs =
            JPH::EAllowedDOFs::TranslationX | JPH::EAllowedDOFs::TranslationY | JPH::EAllowedDOFs::TranslationZ;
    if (d.mass > 0.0f && d.motion == Motion::Dynamic) {
        s.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
        s.mMassPropertiesOverride.mMass = d.mass;
    }
    return s;
}

} // namespace

struct PhysicsSnapshot::Impl {
    std::vector<BodyRecord> records;
    BodyIdAllocator ids;
    std::vector<u8> jolt;
    std::vector<CharacterSave> characters;
    u32 nextCharacter = 0;
    std::vector<JointState> joints; // in Jolt's constraint order
    u32 nextJoint = 0;
};

PhysicsSnapshot::PhysicsSnapshot() = default;
PhysicsSnapshot::~PhysicsSnapshot() { delete m; }
PhysicsSnapshot::PhysicsSnapshot(PhysicsSnapshot&& other) noexcept : m(other.m) { other.m = nullptr; }
PhysicsSnapshot& PhysicsSnapshot::operator=(PhysicsSnapshot&& other) noexcept {
    std::swap(m, other.m);
    return *this;
}
bool PhysicsSnapshot::empty() const { return !m; }

namespace {

template <class T>
void writeValue(JPH::StreamOut& out, const T& v) {
    out.WriteBytes(&v, sizeof(T));
}
template <class T>
void writeVector(JPH::StreamOut& out, const std::vector<T>& v) {
    writeValue(out, static_cast<u32>(v.size()));
    if (!v.empty()) out.WriteBytes(v.data(), v.size() * sizeof(T));
}
template <class T>
bool readVector(JPH::StreamIn& in, std::vector<T>& v, u32 limit) {
    u32 n = 0;
    in.ReadBytes(&n, sizeof(n));
    if (in.IsFailed() || n > limit) return false;
    v.resize(n);
    if (n) in.ReadBytes(v.data(), n * sizeof(T));
    return !in.IsFailed();
}

} // namespace

// Per body (id, BodyDesc, shape). Shapes are in Jolt binary format and identical shapes are written once (SaveWithChildren id table)
void PhysicsSnapshot::write(std::vector<u8>& out) const {
    out.clear();
    if (!m) return;
    ByteStateRecorder stream(out);
    JPH::Shape::ShapeToIDMap shapeIds;
    JPH::Shape::MaterialToIDMap materialIds;
    writeValue(stream, static_cast<u32>(m->records.size()));
    for (const BodyRecord& r : m->records) {
        writeValue(stream, r.id.GetIndexAndSequenceNumber());
        writeValue(stream, r.desc);
        r.shape->SaveWithChildren(stream, shapeIds, materialIds);
    }
    writeValue(stream, m->ids.next);
    writeVector(stream, m->ids.free);
    writeVector(stream, m->ids.sequence);
    writeVector(stream, m->jolt);
    writeValue(stream, m->nextCharacter);
    writeValue(stream, static_cast<u32>(m->characters.size()));
    for (const CharacterSave& c : m->characters) {
        writeValue(stream, c.id);
        writeValue(stream, c.desc);
        writeValue(stream, c.body);
        writeVector(stream, c.state);
    }
    writeValue(stream, m->nextJoint);
    writeVector(stream, m->joints);
}

bool PhysicsSnapshot::read(const u8* data, size_t size) {
    std::vector<u8> bytes(data, data + size);
    ByteStateRecorder stream(bytes);
    Impl loaded;
    JPH::Shape::IDToShapeMap shapes;
    JPH::Shape::IDToMaterialMap materials;
    constexpr u32 kLimit = 1u << 24;
    u32 count = 0;
    stream.ReadBytes(&count, sizeof(count));
    if (stream.IsFailed() || count > kLimit) return false;
    loaded.records.reserve(count);
    for (u32 i = 0; i < count; ++i) {
        u32 id = 0;
        BodyRecord r;
        stream.ReadBytes(&id, sizeof(id));
        stream.ReadBytes(&r.desc, sizeof(r.desc));
        if (stream.IsFailed()) return false;
        JPH::Shape::ShapeResult shape = JPH::Shape::sRestoreWithChildren(stream, shapes, materials);
        if (shape.HasError() || stream.IsFailed()) {
            MEALKIT_ERROR("Physics snapshot: body shape %u could not be read (%s)", i,
                          shape.HasError() ? shape.GetError().c_str() : "truncated");
            return false;
        }
        r.id = JPH::BodyID(id);
        r.shape = shape.Get();
        loaded.records.push_back(std::move(r));
    }
    stream.ReadBytes(&loaded.ids.next, sizeof(loaded.ids.next));
    if (!readVector(stream, loaded.ids.free, kLimit) || !readVector(stream, loaded.ids.sequence, kLimit) ||
        !readVector(stream, loaded.jolt, 1u << 30))
        return false;
    u32 characters = 0;
    stream.ReadBytes(&loaded.nextCharacter, sizeof(loaded.nextCharacter));
    stream.ReadBytes(&characters, sizeof(characters));
    if (stream.IsFailed() || characters > kLimit) return false;
    loaded.characters.resize(characters);
    for (CharacterSave& c : loaded.characters) {
        stream.ReadBytes(&c.id, sizeof(c.id));
        stream.ReadBytes(&c.desc, sizeof(c.desc));
        stream.ReadBytes(&c.body, sizeof(c.body));
        if (!readVector(stream, c.state, kLimit)) return false;
    }
    stream.ReadBytes(&loaded.nextJoint, sizeof(loaded.nextJoint));
    if (stream.IsFailed() || !readVector(stream, loaded.joints, kLimit)) return false;
    if (!m) m = new Impl();
    *m = std::move(loaded);
    return true;
}
size_t PhysicsSnapshot::byteSize() const {
    return m ? m->jolt.size() + m->records.size() * sizeof(BodyRecord) + m->ids.sequence.size() + m->ids.free.size() * 4
             : 0;
}

namespace {

void addBody(JPH::BodyInterface& bi, const BodyRecord& r) {
    bi.CreateBodyWithID(r.id, makeSettings(r.shape, r.desc));
    bi.AddBody(r.id, r.desc.motion == Motion::Static ? JPH::EActivation::DontActivate : JPH::EActivation::Activate);
}

} // namespace

// Collects contact events (called from Jolt worker threads)
class ContactCollector final : public JPH::ContactListener {
public:
    f32 minSpeed = 0.0f;
    std::mutex mutex;
    std::vector<ContactEvent> events;

    void OnContactAdded(const JPH::Body& body1, const JPH::Body& body2, const JPH::ContactManifold& manifold,
                        JPH::ContactSettings&) override {
        if (minSpeed <= 0.0f || manifold.mRelativeContactPointsOn1.empty()) return;
        const JPH::RVec3 point = manifold.GetWorldSpaceContactPointOn1(0);
        const JPH::Vec3 relative = body1.GetPointVelocity(point) - body2.GetPointVelocity(point);
        const f32 speed = relative.Dot(manifold.mWorldSpaceNormal); // the normal is 1 → 2
        if (speed < minSpeed) return;
        BodyId a = body1.GetID().GetIndexAndSequenceNumber(), b = body2.GetID().GetIndexAndSequenceNumber();
        if (b < a) std::swap(a, b);
        ContactEvent e{a, b, fromJ(point), speed};
        std::lock_guard<std::mutex> lock(mutex);
        events.push_back(e);
    }
};

struct PhysicsWorld::Impl {
    ContactCollector contacts;
    u32 maxContacts = 0;
    BroadPhaseLayerMap broadPhaseLayers;
    ObjectVsBroadPhaseFilter objectVsBroadPhase;
    ObjectPairFilter objectPairs;
    JPH::PhysicsSystem system;
    JPH::TempAllocatorImpl tempAllocator{16 * 1024 * 1024};
    JPH::JobSystem* jobs = nullptr;           // the engine's shared pool (jobs::), or ownPool
    JPH::JobSystemThreadPool* ownPool = nullptr; // only when init asked for an explicit thread count (thread count tests)
    std::vector<BodyRecord> bodies; // creation order (hash, snapshots)
    BodyIdAllocator ids;
    std::vector<CharacterRecord> characters; // creation order (declared after system: released first)
    u32 nextCharacter = 0;
    bool plane2d = false; // setPlane2D
    std::vector<JointRecord> joints; // mirrors Jolt's constraint array order (swap-remove like Jolt)
    u32 nextJoint = 0;
    u32 threads = 0;

    JPH::Ref<JPH::CharacterVirtual> makeCharacter(const CharacterDesc& d) {
        JPH::CharacterVirtualSettings s;
        s.mShape = new JPH::CapsuleShape(std::max(d.halfHeight, 0.01f), std::max(d.radius, 0.01f));
        s.mMaxSlopeAngle = JPH::ACos(std::clamp(d.maxSlopeCos, 0.0f, 1.0f)); // Jolt's own ACos/Cos (deterministic)
        s.mSupportingVolume = JPH::Plane(JPH::Vec3::sAxisY(), d.halfHeight); // only the lower hemisphere supports
        s.mMass = d.mass;
        s.mMaxStrength = d.maxStrength;
        return new JPH::CharacterVirtual(&s, toJR(d.position), JPH::Quat::sIdentity(), &system);
    }

    CharacterRecord* character(CharacterId id) {
        for (CharacterRecord& c : characters)
            if (c.id == id) return &c;
        return nullptr;
    }
    const CharacterRecord* character(CharacterId id) const {
        for (const CharacterRecord& c : characters)
            if (c.id == id) return &c;
        return nullptr;
    }

    JPH::Ref<JPH::SwingTwistConstraint> makeJoint(const JointState& j) {
        JPH::SwingTwistConstraintSettings s;
        s.mSpace = JPH::EConstraintSpace::LocalToBodyCOM;
        s.mPosition1 = toJR(j.position1);
        s.mTwistAxis1 = toJ(j.twist1);
        s.mPlaneAxis1 = toJ(j.plane1);
        s.mPosition2 = toJR(j.position2);
        s.mTwistAxis2 = toJ(j.twist2);
        s.mPlaneAxis2 = toJ(j.plane2);
        s.mNormalHalfConeAngle = j.desc.normalCone;
        s.mPlaneHalfConeAngle = j.desc.planeCone;
        s.mTwistMinAngle = j.desc.twistMin;
        s.mTwistMaxAngle = j.desc.twistMax;
        for (JPH::MotorSettings* motor : {&s.mSwingMotorSettings, &s.mTwistMotorSettings}) {
            motor->mSpringSettings =
                JPH::SpringSettings(JPH::ESpringMode::FrequencyAndDamping, j.desc.motorFrequency, j.desc.motorDamping);
            motor->SetTorqueLimit(j.torque);
        }
        JPH::Constraint* c = bi().CreateConstraint(&s, JPH::BodyID(j.parent), JPH::BodyID(j.child));
        if (!c) return nullptr;
        JPH::Ref<JPH::SwingTwistConstraint> joint = static_cast<JPH::SwingTwistConstraint*>(c);
        joint->SetSwingMotorState(JPH::EMotorState::Position);
        joint->SetTwistMotorState(JPH::EMotorState::Position);
        joint->SetTargetOrientationBS(toJ(j.target));
        system.AddConstraint(joint);
        return joint;
    }

    JointRecord* joint(JointId id) {
        for (JointRecord& j : joints)
            if (j.state.id == id) return &j;
        return nullptr;
    }

    void removeAllJoints() {
        for (JointRecord& j : joints) system.RemoveConstraint(j.constraint);
        joints.clear();
    }

    JPH::BodyInterface& bi() { return system.GetBodyInterfaceNoLock(); }
    const JPH::BodyInterface& bi() const { return system.GetBodyInterfaceNoLock(); }

    BodyId add(const JPH::Shape* shape, const BodyDesc& d) {
        u32 index = 0;
        if (!ids.free.empty()) {
            index = ids.free.back();
            ids.free.pop_back();
        } else if (ids.next < ids.sequence.size()) {
            index = ids.next++;
        } else {
            MEALKIT_ERROR("Physics: body limit reached");
            return kInvalidBody;
        }
        BodyRecord record{JPH::BodyID(index, ids.sequence[index]), shape, d};
        record.desc.plane2d = record.desc.plane2d || plane2d;
        addBody(bi(), record);
        bodies.push_back(record);
        return record.id.GetIndexAndSequenceNumber();
    }

    void remove(const JPH::BodyID& id) {
        bi().RemoveBody(id);
        bi().DestroyBody(id);
    }
};

PhysicsWorld::~PhysicsWorld() { shutdown(); }

bool PhysicsWorld::init(u32 maxBodies, u32 threads) {
    shutdown();
    acquireJolt();
    m = new Impl();
    if (threads == 0) {
        jobs::acquire();
        m->jobs = static_cast<JPH::JobSystem*>(jobs::joltJobSystem());
        m->threads = jobs::workerCount();
    } else {
        m->ownPool = new JPH::JobSystemThreadPool(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, static_cast<int>(threads));
        m->jobs = m->ownPool;
        m->threads = threads;
    }
    m->system.Init(maxBodies, 0, 65536, 20480, m->broadPhaseLayers, m->objectVsBroadPhase, m->objectPairs);
    m->bodies.reserve(maxBodies);
    m->ids.sequence.assign(maxBodies, 0);
    m->ids.free.reserve(maxBodies);
    return true;
}

void PhysicsWorld::shutdown() {
    if (!m) return;
    m->removeAllJoints();
    m->characters.clear();
    for (const BodyRecord& r : m->bodies) m->remove(r.id);
    const bool shared = m->ownPool == nullptr;
    delete m->ownPool;
    delete m;
    m = nullptr;
    if (shared) jobs::release();
    releaseJolt();
}

void PhysicsWorld::setGravity(Vec3 gravity) { m->system.SetGravity(toJ(gravity)); }
void PhysicsWorld::setPlane2D(bool on) { m->plane2d = on; }
bool PhysicsWorld::plane2D() const { return m && m->plane2d; }

BodyId PhysicsWorld::createBox(Vec3 half, const BodyDesc& desc) {
    half = {std::max(half.x, 0.01f), std::max(half.y, 0.01f), std::max(half.z, 0.01f)};
    const f32 convexRadius = std::min(JPH::cDefaultConvexRadius, std::min({half.x, half.y, half.z}) * 0.5f);
    JPH::Ref<JPH::Shape> shape = new JPH::BoxShape(toJ(half), convexRadius);
    return m->add(shape, desc);
}

BodyId PhysicsWorld::createSphere(f32 radius, const BodyDesc& desc) {
    JPH::Ref<JPH::Shape> shape = new JPH::SphereShape(std::max(radius, 0.01f));
    return m->add(shape, desc);
}

BodyId PhysicsWorld::createCapsule(f32 halfHeight, f32 radius, const BodyDesc& desc) {
    JPH::Ref<JPH::Shape> shape = new JPH::CapsuleShape(std::max(halfHeight, 0.01f), std::max(radius, 0.01f));
    return m->add(shape, desc);
}

BodyId PhysicsWorld::createConvexHull(const Vec3* points, u32 count, const BodyDesc& desc) {
    JPH::Array<JPH::Vec3> pts;
    pts.reserve(count);
    for (u32 i = 0; i < count; ++i) pts.push_back(toJ(points[i]));
    JPH::ConvexHullShapeSettings settings(pts);
    JPH::ShapeSettings::ShapeResult result = settings.Create();
    if (result.HasError()) {
        Vec3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
        for (u32 i = 0; i < count; ++i) {
            lo = {std::min(lo.x, points[i].x), std::min(lo.y, points[i].y), std::min(lo.z, points[i].z)};
            hi = {std::max(hi.x, points[i].x), std::max(hi.y, points[i].y), std::max(hi.z, points[i].z)};
        }
        MEALKIT_WARN("Physics: convex hull failed (%s), using bounding box", result.GetError().c_str());
        BodyDesc shifted = desc;
        shifted.position = desc.position + rotate(desc.rotation, (lo + hi) * 0.5f);
        return createBox((hi - lo) * 0.5f, shifted);
    }
    return m->add(result.Get(), desc);
}

BodyId PhysicsWorld::createHeightField(const f32* heights, u32 size, Vec3 offset, f32 spacingX, f32 spacingZ,
                                       const BodyDesc& desc) {
    if (size < 2) return kInvalidBody;
    // Jolt wants the sample count to be a multiple of its block size (2, at least 4 samples): pad with no-collision samples
    const u32 padded = std::max(4u, (size + 1) & ~1u);
    std::vector<float> samples(static_cast<size_t>(padded) * padded, JPH::HeightFieldShapeConstants::cNoCollisionValue);
    for (u32 z = 0; z < size; ++z)
        std::memcpy(&samples[static_cast<size_t>(z) * padded], heights + static_cast<size_t>(z) * size, sizeof(f32) * size);
    JPH::HeightFieldShapeSettings settings(samples.data(), toJ(offset), JPH::Vec3(spacingX, 1.0f, spacingZ), padded);
    JPH::ShapeSettings::ShapeResult result = settings.Create();
    if (result.HasError()) {
        MEALKIT_ERROR("Physics: height field failed (%s)", result.GetError().c_str());
        return kInvalidBody;
    }
    BodyDesc d = desc;
    d.motion = Motion::Static;
    return m->add(result.Get(), d);
}

void PhysicsWorld::destroy(BodyId body) {
    const JPH::BodyID id(body);
    auto it = std::find_if(m->bodies.begin(), m->bodies.end(), [&](const BodyRecord& r) { return r.id == id; });
    if (it == m->bodies.end()) return;
    m->remove(id);
    m->bodies.erase(it);
    const u32 index = id.GetIndex();
    ++m->ids.sequence[index]; // so reusing the same number gives a different id
    m->ids.free.push_back(index);
}

CharacterId PhysicsWorld::createCharacter(const CharacterDesc& desc) {
    BodyDesc b;
    b.motion = Motion::Kinematic;
    b.position = desc.position;
    b.allowSleep = false;
    CharacterRecord c;
    c.id = m->nextCharacter++;
    c.desc = desc;
    c.body = m->add(new JPH::CapsuleShape(std::max(desc.halfHeight, 0.01f), std::max(desc.radius, 0.01f)), b);
    c.character = m->makeCharacter(desc);
    m->characters.push_back(std::move(c));
    return m->characters.back().id;
}

void PhysicsWorld::destroyCharacter(CharacterId character) {
    auto it = std::find_if(m->characters.begin(), m->characters.end(),
                           [&](const CharacterRecord& c) { return c.id == character; });
    if (it == m->characters.end()) return;
    const BodyId body = it->body;
    m->characters.erase(it);
    destroy(body);
}

void PhysicsWorld::moveCharacter(CharacterId character, Vec3 velocity, Vec3 gravity, f32 dt, f32 stepUp, f32 stickDown) {
    CharacterRecord* c = m->character(character);
    if (!c) return;
    if (m->plane2d) velocity.z = 0.0f;
    c->character->SetLinearVelocity(toJ(velocity));
    JPH::CharacterVirtual::ExtendedUpdateSettings settings;
    settings.mStickToFloorStepDown = JPH::Vec3(0.0f, -stickDown, 0.0f);
    settings.mWalkStairsStepUp = JPH::Vec3(0.0f, stepUp, 0.0f);
    const JPH::IgnoreSingleBodyFilter self{JPH::BodyID(c->body)}; // not its own following body
    c->character->ExtendedUpdate(dt, toJ(gravity), settings, m->system.GetDefaultBroadPhaseLayerFilter(Layers::Moving),
                                 m->system.GetDefaultLayerFilter(Layers::Moving), self, {}, m->tempAllocator);
    if (m->plane2d) { // 2D: sliding along walls can push it off the plane
        const JPH::RVec3 p = c->character->GetPosition();
        c->character->SetPosition(JPH::RVec3(p.GetX(), p.GetY(), 0.0f));
        const JPH::Vec3 v = c->character->GetLinearVelocity();
        c->character->SetLinearVelocity(JPH::Vec3(v.GetX(), v.GetY(), 0.0f));
    }
    m->bi().MoveKinematic(JPH::BodyID(c->body), c->character->GetPosition(), JPH::Quat::sIdentity(), dt);
}

Vec3 PhysicsWorld::characterPosition(CharacterId character) const {
    const CharacterRecord* c = m->character(character);
    return c ? fromJ(c->character->GetPosition()) : Vec3{};
}

Vec3 PhysicsWorld::characterVelocity(CharacterId character) const {
    const CharacterRecord* c = m->character(character);
    return c ? fromJ(c->character->GetLinearVelocity()) : Vec3{};
}

void PhysicsWorld::setCharacterVelocity(CharacterId character, Vec3 velocity) {
    if (CharacterRecord* c = m->character(character)) c->character->SetLinearVelocity(toJ(velocity));
}

CharacterGround PhysicsWorld::characterGround(CharacterId character) const {
    CharacterGround g;
    const CharacterRecord* c = m->character(character);
    if (!c) return g;
    g.supported = c->character->GetGroundState() == JPH::CharacterBase::EGroundState::OnGround;
    g.normal = fromJ(c->character->GetGroundNormal());
    g.velocity = fromJ(c->character->GetGroundVelocity());
    return g;
}

BodyId PhysicsWorld::characterBody(CharacterId character) const {
    const CharacterRecord* c = m->character(character);
    return c ? c->body : kInvalidBody;
}

JointId PhysicsWorld::createJoint(BodyId parent, BodyId child, const JointDesc& desc) {
    const JPH::BodyID p(parent), c(child);
    const JPH::RMat44 parentInv = m->bi().GetCenterOfMassTransform(p).InversedRotationTranslation();
    const JPH::RMat44 childInv = m->bi().GetCenterOfMassTransform(c).InversedRotationTranslation();
    JointState j;
    j.id = m->nextJoint++;
    j.parent = parent;
    j.child = child;
    j.desc = desc;
    j.position1 = fromJ(JPH::Vec3(parentInv * toJR(desc.pivot)));
    j.twist1 = fromJ(parentInv.Multiply3x3(toJ(desc.twistAxis)));
    j.plane1 = fromJ(parentInv.Multiply3x3(toJ(desc.planeAxis)));
    j.position2 = fromJ(JPH::Vec3(childInv * toJR(desc.pivot)));
    j.twist2 = fromJ(childInv.Multiply3x3(toJ(desc.twistAxis)));
    j.plane2 = fromJ(childInv.Multiply3x3(toJ(desc.planeAxis)));
    j.target = fromJ(m->bi().GetRotation(p).Conjugated() * m->bi().GetRotation(c)); // holds the created pose
    j.torque = desc.maxTorque;
    JPH::Ref<JPH::SwingTwistConstraint> constraint = m->makeJoint(j);
    if (!constraint) {
        MEALKIT_ERROR("Physics: joint between bodies %08x and %08x could not be created", parent, child);
        return kInvalidJoint;
    }
    m->joints.push_back({j, constraint});
    return j.id;
}

void PhysicsWorld::destroyJoint(JointId joint) {
    auto it = std::find_if(m->joints.begin(), m->joints.end(), [&](const JointRecord& j) { return j.state.id == joint; });
    if (it == m->joints.end()) return;
    m->system.RemoveConstraint(it->constraint);
    // Same swap-remove as Jolt's constraint array, so the order stays equal (snapshots restore by index)
    if (it != m->joints.end() - 1) *it = std::move(m->joints.back());
    m->joints.pop_back();
}

void PhysicsWorld::setJointTarget(JointId joint, Quat target) {
    JointRecord* j = m->joint(joint);
    if (!j) return;
    j->state.target = target;
    j->constraint->SetTargetOrientationBS(toJ(target));
}

void PhysicsWorld::setJointTorque(JointId joint, f32 maxTorque) {
    JointRecord* j = m->joint(joint);
    if (!j || j->state.torque == maxTorque) return;
    j->state.torque = maxTorque;
    j->constraint->GetSwingMotorSettings().SetTorqueLimit(maxTorque);
    j->constraint->GetTwistMotorSettings().SetTorqueLimit(maxTorque);
}

u32 PhysicsWorld::jointCount() const { return static_cast<u32>(m->joints.size()); }

void PhysicsWorld::optimize() { m->system.OptimizeBroadPhase(); }

void PhysicsWorld::step(f32 dt, int collisionSteps) {
    MEALKIT_ZONE("Physics step");
    std::vector<ContactEvent>& events = m->contacts.events;
    events.clear();
    m->system.Update(dt, collisionSteps, &m->tempAllocator, m->jobs);
    if (events.empty()) return;
    // Collection order differs per thread: sort by strength → body index → position to get the same result
    std::sort(events.begin(), events.end(), [](const ContactEvent& x, const ContactEvent& y) {
        if (x.speed != y.speed) return x.speed > y.speed;
        if (x.a != y.a) return x.a < y.a;
        if (x.b != y.b) return x.b < y.b;
        if (x.point.x != y.point.x) return x.point.x < y.point.x;
        if (x.point.y != y.point.y) return x.point.y < y.point.y;
        return x.point.z < y.point.z;
    });
    if (events.size() > m->maxContacts) events.resize(m->maxContacts);
}

void PhysicsWorld::setContactEvents(f32 minSpeed, u32 maxPerStep) {
    m->contacts.minSpeed = minSpeed;
    m->maxContacts = maxPerStep;
    m->system.SetContactListener(minSpeed > 0.0f ? &m->contacts : nullptr);
}

const std::vector<ContactEvent>& PhysicsWorld::contacts() const { return m->contacts.events; }

Vec3 PhysicsWorld::position(BodyId body) const { return fromJ(m->bi().GetPosition(JPH::BodyID(body))); }
Quat PhysicsWorld::rotation(BodyId body) const { return fromJ(m->bi().GetRotation(JPH::BodyID(body))); }
Vec3 PhysicsWorld::linearVelocity(BodyId body) const { return fromJ(m->bi().GetLinearVelocity(JPH::BodyID(body))); }
Vec3 PhysicsWorld::angularVelocity(BodyId body) const { return fromJ(m->bi().GetAngularVelocity(JPH::BodyID(body))); }

void PhysicsWorld::setLinearVelocity(BodyId body, Vec3 v) { m->bi().SetLinearVelocity(JPH::BodyID(body), toJ(v)); }
void PhysicsWorld::setAngularVelocity(BodyId body, Vec3 v) { m->bi().SetAngularVelocity(JPH::BodyID(body), toJ(v)); }

void PhysicsWorld::addLinearVelocity(BodyId body, Vec3 delta) {
    const JPH::BodyID id(body);
    m->bi().SetLinearVelocity(id, m->bi().GetLinearVelocity(id) + toJ(delta)); // also wakes sleeping bodies
}

void PhysicsWorld::setTransform(BodyId body, Vec3 position, Quat rotation) {
    m->bi().SetPositionAndRotation(JPH::BodyID(body), toJR(position), toJ(rotation), JPH::EActivation::Activate);
}

bool PhysicsWorld::isActive(BodyId body) const { return m->bi().IsActive(JPH::BodyID(body)); }

bool PhysicsWorld::castRay(Vec3 origin, Vec3 direction, RayHit& hit, BodyId ignore, u32 ignoreGroup) const {
    const JPH::RRayCast ray{toJR(origin), toJ(direction)};
    JPH::RayCastResult result;
    const RayBodyFilter filter(JPH::BodyID(ignore == kInvalidBody ? JPH::BodyID::cInvalidBodyID : ignore), ignoreGroup);
    if (!m->system.GetNarrowPhaseQueryNoLock().CastRay(ray, result, {}, {}, filter)) return false;
    hit.body = result.mBodyID.GetIndexAndSequenceNumber();
    hit.fraction = result.mFraction;
    return true;
}

void PhysicsWorld::hashState(Hasher& hasher) const {
    const JPH::BodyInterface& bi = m->bi();
    for (const BodyRecord& r : m->bodies) {
        const JPH::BodyID id = r.id;
        hasher.add(id.GetIndexAndSequenceNumber());
        const Vec3 p = fromJ(bi.GetPosition(id));
        const Quat q = fromJ(bi.GetRotation(id));
        const Vec3 v = fromJ(bi.GetLinearVelocity(id));
        const Vec3 w = fromJ(bi.GetAngularVelocity(id));
        hasher.add(p);
        hasher.add(q);
        hasher.add(v);
        hasher.add(w);
    }
    for (const CharacterRecord& c : m->characters) {
        hasher.add(c.id);
        hasher.add(fromJ(c.character->GetPosition()));
        hasher.add(fromJ(c.character->GetLinearVelocity()));
        hasher.add(static_cast<u32>(c.character->GetGroundState()));
    }
    for (const JointRecord& j : m->joints) {
        hasher.add(j.state.id);
        hasher.add(j.state.target);
        hasher.add(j.state.torque);
    }
}

void PhysicsWorld::writeState(std::string& out) const {
    const JPH::BodyInterface& bi = m->bi();
    char line[320];
    u32 index = 0;
    for (const BodyRecord& r : m->bodies) {
        const Vec3 p = fromJ(bi.GetPosition(r.id));
        const Quat q = fromJ(bi.GetRotation(r.id));
        const Vec3 v = fromJ(bi.GetLinearVelocity(r.id));
        const Vec3 w = fromJ(bi.GetAngularVelocity(r.id));
        std::snprintf(line, sizeof(line),
                      "body %u %08x p %.9g %.9g %.9g q %.9g %.9g %.9g %.9g v %.9g %.9g %.9g w %.9g %.9g %.9g\n", index++,
                      r.id.GetIndexAndSequenceNumber(), p.x, p.y, p.z, q.x, q.y, q.z, q.w, v.x, v.y, v.z, w.x, w.y, w.z);
        out += line;
    }
    for (const CharacterRecord& c : m->characters) {
        const Vec3 p = fromJ(c.character->GetPosition());
        const Vec3 v = fromJ(c.character->GetLinearVelocity());
        std::snprintf(line, sizeof(line), "character %u p %.9g %.9g %.9g v %.9g %.9g %.9g ground %u\n", c.id, p.x, p.y, p.z,
                      v.x, v.y, v.z, static_cast<u32>(c.character->GetGroundState()));
        out += line;
    }
    for (const JointRecord& j : m->joints) {
        const Quat q = j.state.target;
        std::snprintf(line, sizeof(line), "joint %u %08x %08x target %.9g %.9g %.9g %.9g torque %.9g\n", j.state.id,
                      j.state.parent, j.state.child, q.x, q.y, q.z, q.w, j.state.torque);
        out += line;
    }
}

void PhysicsWorld::nudgeForTest() {
    for (const BodyRecord& r : m->bodies)
        if (r.desc.motion == Motion::Dynamic) {
            m->bi().SetLinearVelocity(r.id, m->bi().GetLinearVelocity(r.id) + JPH::Vec3(0.001f, 0.0f, 0.0f));
            return;
        }
}

void PhysicsWorld::save(PhysicsSnapshot& out) const {
    if (!out.m) out.m = new PhysicsSnapshot::Impl();
    out.m->records = m->bodies;
    out.m->ids = m->ids;
    out.m->jolt.clear();
    ByteStateRecorder recorder(out.m->jolt);
    m->system.SaveState(recorder, JPH::EStateRecorderState::All);
    out.m->characters.clear();
    for (const CharacterRecord& c : m->characters) {
        CharacterSave save{c.id, c.desc, c.body, {}};
        ByteStateRecorder state(save.state);
        c.character->SaveState(state);
        out.m->characters.push_back(std::move(save));
    }
    out.m->nextCharacter = m->nextCharacter;
    out.m->joints.clear();
    for (const JointRecord& j : m->joints) out.m->joints.push_back(j.state);
    out.m->nextJoint = m->nextJoint;
}

bool PhysicsWorld::restore(const PhysicsSnapshot& snapshot) {
    if (!snapshot.m) return false;
    const PhysicsSnapshot::Impl& s = *snapshot.m;
    auto contains = [](const std::vector<BodyRecord>& list, const JPH::BodyID& id) {
        return std::any_of(list.begin(), list.end(), [&](const BodyRecord& r) { return r.id == id; });
    };
    // Joints are rebuilt from the snapshot after the bodies (in its order, which is then Jolt's order too)
    m->removeAllJoints();
    // Bodies created after the snapshot are deleted, and bodies deleted after the snapshot are recreated with the same id
    for (const BodyRecord& r : m->bodies)
        if (!contains(s.records, r.id)) m->remove(r.id);
    for (const BodyRecord& r : s.records)
        if (!contains(m->bodies, r.id)) addBody(m->bi(), r);
    m->bodies = s.records;
    m->ids = s.ids;
    for (const JointState& j : s.joints) m->joints.push_back({j, m->makeJoint(j)});
    m->nextJoint = s.nextJoint;
    std::vector<u8> bytes = s.jolt; // StateRecorder has a read position
    ByteStateRecorder recorder(bytes);
    if (!m->system.RestoreState(recorder)) {
        MEALKIT_ERROR("Physics: RestoreState failed");
        return false;
    }
    // Characters: same as bodies (their following bodies were restored above)
    std::vector<CharacterRecord> characters;
    characters.reserve(s.characters.size());
    for (const CharacterSave& save : s.characters) {
        CharacterRecord c;
        if (CharacterRecord* existing = m->character(save.id)) c = std::move(*existing);
        else c.character = m->makeCharacter(save.desc);
        c.id = save.id;
        c.desc = save.desc;
        c.body = save.body;
        std::vector<u8> state = save.state;
        ByteStateRecorder in(state);
        c.character->RestoreState(in);
        characters.push_back(std::move(c));
    }
    m->characters = std::move(characters);
    m->nextCharacter = s.nextCharacter;
    return true;
}

void PhysicsWorld::clearBodies() {
    m->removeAllJoints();
    m->nextJoint = 0;
    m->characters.clear();
    m->nextCharacter = 0;
    for (const BodyRecord& r : m->bodies) m->remove(r.id);
    m->bodies.clear();
    const size_t maxBodies = m->ids.sequence.size();
    m->ids = {};
    m->ids.sequence.assign(maxBodies, 0);
}

u32 PhysicsWorld::bodyCount() const { return static_cast<u32>(m->bodies.size()); }
u32 PhysicsWorld::activeBodyCount() const { return m->system.GetNumActiveBodies(JPH::EBodyType::RigidBody); }
u32 PhysicsWorld::threadCount() const { return m->threads; }

} // namespace mealkit::physics
