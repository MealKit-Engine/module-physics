#include "mealkit/physics/PhysicsWorld.h"

#include "mealkit/core/Log.h"
#include "mealkit/core/Profile.h"

#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/ContactListener.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
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

// Jolt 상태를 바이트 배열에 쓰고 읽는다
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
    JPH::RegisterDefaultAllocator();
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

// 만든 바디의 기록: 스냅샷을 되돌릴 때 사라진 바디를 같은 id로 다시 만든다
struct BodyRecord {
    JPH::BodyID id;
    JPH::RefConst<JPH::Shape> shape;
    BodyDesc desc;
};

// 바디 id 할당 상태 (스냅샷에 들어간다)
struct BodyIdAllocator {
    std::vector<u32> free; // 다시 쓸 번호 (맨 뒤부터)
    std::vector<u8> sequence;
    u32 next = 0;
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
    if (d.lockRotation)
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
};

PhysicsSnapshot::PhysicsSnapshot() = default;
PhysicsSnapshot::~PhysicsSnapshot() { delete m; }
PhysicsSnapshot::PhysicsSnapshot(PhysicsSnapshot&& other) noexcept : m(other.m) { other.m = nullptr; }
PhysicsSnapshot& PhysicsSnapshot::operator=(PhysicsSnapshot&& other) noexcept {
    std::swap(m, other.m);
    return *this;
}
bool PhysicsSnapshot::empty() const { return !m; }
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

// 접촉 이벤트 모으기 (Jolt 작업 스레드에서 불린다)
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
        const f32 speed = relative.Dot(manifold.mWorldSpaceNormal); // 법선은 1 → 2
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
    JPH::JobSystemThreadPool* jobs = nullptr;
    std::vector<BodyRecord> bodies; // 생성 순서 (해시, 스냅샷)
    BodyIdAllocator ids;
    u32 threads = 0;

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
        const u32 cores = std::max(1u, std::thread::hardware_concurrency());
        threads = std::clamp(cores - 1, 1u, 6u); // 메인 스레드 하나는 남긴다
    }
    m->threads = threads;
    m->jobs = new JPH::JobSystemThreadPool(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, static_cast<int>(threads));
    m->system.Init(maxBodies, 0, 65536, 20480, m->broadPhaseLayers, m->objectVsBroadPhase, m->objectPairs);
    m->bodies.reserve(maxBodies);
    m->ids.sequence.assign(maxBodies, 0);
    m->ids.free.reserve(maxBodies);
    return true;
}

void PhysicsWorld::shutdown() {
    if (!m) return;
    for (const BodyRecord& r : m->bodies) m->remove(r.id);
    delete m->jobs;
    delete m;
    m = nullptr;
    releaseJolt();
}

void PhysicsWorld::setGravity(Vec3 gravity) { m->system.SetGravity(toJ(gravity)); }

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

void PhysicsWorld::destroy(BodyId body) {
    const JPH::BodyID id(body);
    auto it = std::find_if(m->bodies.begin(), m->bodies.end(), [&](const BodyRecord& r) { return r.id == id; });
    if (it == m->bodies.end()) return;
    m->remove(id);
    m->bodies.erase(it);
    const u32 index = id.GetIndex();
    ++m->ids.sequence[index]; // 같은 번호를 다시 쓰면 다른 id가 되게
    m->ids.free.push_back(index);
}

void PhysicsWorld::optimize() { m->system.OptimizeBroadPhase(); }

void PhysicsWorld::step(f32 dt, int collisionSteps) {
    MEALKIT_ZONE("Physics step");
    std::vector<ContactEvent>& events = m->contacts.events;
    events.clear();
    m->system.Update(dt, collisionSteps, &m->tempAllocator, m->jobs);
    if (events.empty()) return;
    // 모은 순서는 스레드마다 다르다: 세기 → 바디 번호 → 위치로 정렬해 같은 결과를 만든다
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
    m->bi().SetLinearVelocity(id, m->bi().GetLinearVelocity(id) + toJ(delta)); // 잠든 바디도 깨운다
}

void PhysicsWorld::setTransform(BodyId body, Vec3 position, Quat rotation) {
    m->bi().SetPositionAndRotation(JPH::BodyID(body), toJR(position), toJ(rotation), JPH::EActivation::Activate);
}

bool PhysicsWorld::isActive(BodyId body) const { return m->bi().IsActive(JPH::BodyID(body)); }

bool PhysicsWorld::castRay(Vec3 origin, Vec3 direction, RayHit& hit, BodyId ignore) const {
    const JPH::RRayCast ray{toJR(origin), toJ(direction)};
    JPH::RayCastResult result;
    const JPH::IgnoreSingleBodyFilter filter(
        JPH::BodyID(ignore == kInvalidBody ? JPH::BodyID::cInvalidBodyID : ignore));
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
}

bool PhysicsWorld::restore(const PhysicsSnapshot& snapshot) {
    if (!snapshot.m) return false;
    const PhysicsSnapshot::Impl& s = *snapshot.m;
    auto contains = [](const std::vector<BodyRecord>& list, const JPH::BodyID& id) {
        return std::any_of(list.begin(), list.end(), [&](const BodyRecord& r) { return r.id == id; });
    };
    // 스냅샷 뒤에 만든 바디는 지우고, 스냅샷 뒤에 지운 바디는 같은 id로 다시 만든다
    for (const BodyRecord& r : m->bodies)
        if (!contains(s.records, r.id)) m->remove(r.id);
    for (const BodyRecord& r : s.records)
        if (!contains(m->bodies, r.id)) addBody(m->bi(), r);
    m->bodies = s.records;
    m->ids = s.ids;
    std::vector<u8> bytes = s.jolt; // StateRecorder는 읽기 위치를 가진다
    ByteStateRecorder recorder(bytes);
    if (!m->system.RestoreState(recorder)) {
        MEALKIT_ERROR("Physics: RestoreState failed");
        return false;
    }
    return true;
}

u32 PhysicsWorld::bodyCount() const { return static_cast<u32>(m->bodies.size()); }
u32 PhysicsWorld::activeBodyCount() const { return m->system.GetNumActiveBodies(JPH::EBodyType::RigidBody); }
u32 PhysicsWorld::threadCount() const { return m->threads; }

} // namespace mealkit::physics
