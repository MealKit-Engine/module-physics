#pragma once

#include "mealkit/core/Hash.h"
#include "mealkit/core/Math.h"

// Jolt Physics 래퍼. 게임 코드에 Jolt 헤더가 퍼지지 않게 핸들(BodyId) 기반 API만 노출한다.
//
// 결정론: Jolt는 CROSS_PLATFORM_DETERMINISTIC으로 빌드되어, 같은 호출을 같은 순서로 하면
// CPU나 스레드 수와 관계없이 같은 결과를 낸다. 따라서 바디 생성/삭제/힘 적용 순서를 모든 피어에서 같게 유지한다.
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
    f32 mass = 0.0f;           // 0 = 모양과 밀도(1000 kg/m³)로 계산
    bool lockRotation = false; // 캐릭터용: 넘어지지 않게 회전을 막는다
    bool allowSleep = true;
};

struct RayHit {
    BodyId body = kInvalidBody;
    f32 fraction = 1.0f; // 0..1, 광선 길이에 대한 비율
};

// 물리 상태 전체의 스냅샷 (바디 목록·모양·속도·접촉 캐시). 같은 프로세스 안에서만 쓴다 (되감기, 그림자 시뮬레이션).
class PhysicsSnapshot {
public:
    PhysicsSnapshot();
    ~PhysicsSnapshot();
    PhysicsSnapshot(PhysicsSnapshot&& other) noexcept;
    PhysicsSnapshot& operator=(PhysicsSnapshot&& other) noexcept;
    PhysicsSnapshot(const PhysicsSnapshot&) = delete;
    PhysicsSnapshot& operator=(const PhysicsSnapshot&) = delete;

    bool empty() const;
    size_t byteSize() const; // 대략적인 메모리 사용량

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

    // threads = 0 이면 코어 수에 맞춘다. 스레드 수는 결정론에 영향을 주지 않는다.
    bool init(u32 maxBodies = 16384, u32 threads = 0);
    void shutdown();
    bool initialized() const { return m != nullptr; }

    void setGravity(Vec3 gravity);
    BodyId createBox(Vec3 halfExtents, const BodyDesc& desc);
    BodyId createSphere(f32 radius, const BodyDesc& desc);
    BodyId createCapsule(f32 halfHeight, f32 radius, const BodyDesc& desc); // halfHeight = 원기둥 부분의 절반
    // 볼록 껍질. 만들 수 없는 점 집합이면 경계 상자로 대신한다.
    BodyId createConvexHull(const Vec3* points, u32 count, const BodyDesc& desc);
    void destroy(BodyId body);
    // 정적 바디를 다 만든 뒤 한 번 호출하면 브로드페이즈가 빨라진다.
    void optimize();

    void step(f32 dt, int collisionSteps = 1);

    Vec3 position(BodyId body) const;
    Quat rotation(BodyId body) const;
    Vec3 linearVelocity(BodyId body) const;
    Vec3 angularVelocity(BodyId body) const;
    void setLinearVelocity(BodyId body, Vec3 velocity);
    void setAngularVelocity(BodyId body, Vec3 velocity);
    void addLinearVelocity(BodyId body, Vec3 delta); // 질량과 무관한 속도 변화
    void setTransform(BodyId body, Vec3 position, Quat rotation);
    bool isActive(BodyId body) const;

    // 광선: origin에서 direction(길이 = 최대 거리)만큼. ignore 바디는 무시한다.
    bool castRay(Vec3 origin, Vec3 direction, RayHit& hit, BodyId ignore = kInvalidBody) const;

    // 모든 바디 상태를 생성 순서대로 해시에 넣는다 (디싱크 감지용).
    void hashState(Hasher& hasher) const;

    // 스냅샷 저장·복원. 복원하면 그 사이에 만들거나 지운 바디도 되돌린다 (바디 id까지 같게).
    // 바디 id는 엔진이 정하는 순서로 매기므로(지운 번호를 정해진 순서로 다시 씀) 복원 뒤에 만드는 바디의 id도 원래와 같다.
    void save(PhysicsSnapshot& out) const;
    bool restore(const PhysicsSnapshot& snapshot);
    u32 bodyCount() const;
    u32 activeBodyCount() const;
    u32 threadCount() const;

private:
    struct Impl;
    Impl* m = nullptr;
};

} // namespace mealkit::physics
