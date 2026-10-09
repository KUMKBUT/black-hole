'use client';

import React, { useRef, useMemo, useState, useEffect, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';

// ──────────────────────────────────────────────────────────────
// GLSL Shaders: Schwarzschild Geodesic Raymarching (RK2 / Midpoint)
// ──────────────────────────────────────────────────────────────

const raymarchVertexShader = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

const raymarchFragmentShader = /* glsl */ `
  precision highp float;

  uniform vec3 uCameraPos;
  uniform mat4 uCameraMatrixWorld;
  uniform mat4 uProjInverse;
  uniform vec2 uResolution;
  uniform float uTime;

  // Physics & Visual Uniforms
  uniform float uRs;                // Schwarzschild radius
  uniform float uDiskInner;         // Inner accretion disk radius (ISCO)
  uniform float uDiskOuter;         // Outer accretion disk radius
  uniform float uLensingStrength;   // Light deflection multiplier
  uniform float uDopplerStrength;   // Relativistic beaming strength
  uniform float uDiskBrightness;    // Luminosity multiplier
  uniform float uSpeed;             // Plasma rotation speed
  uniform int uColorScheme;         // 0: Gargantua, 1: M87 Amber, 2: Cyan Quasar, 3: Superflare
  uniform bool uShowPlanet;         // Toggle Miller's planet

  varying vec2 vUv;

  // ── Procedural Hash Functions ──
  float hash21(vec2 p) {
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
  }

  float hash31(vec3 p) {
    p = fract(p * vec3(234.34, 435.345, 123.456));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y * p.z);
  }

  // 2D Value Noise (Continuous & C1-smooth)
  float valueNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
  }

  // Multi-Octave Fractal Brownian Motion
  float fbm(vec2 p) {
    float total = 0.0;
    float amp = 0.5;
    mat2 rot = mat2(0.80, 0.60, -0.60, 0.80);
    for (int i = 0; i < 4; i++) {
      total += amp * valueNoise(p);
      p = rot * p * 2.15 + vec2(1.7, 3.2);
      amp *= 0.5;
    }
    return total;
  }

  // ── Rich Multi-Spectral Cosmic Starfield ──
  vec3 getRichStarfield(vec3 dir) {
    vec3 col = vec3(0.0);

    // Layer 1: Dense background micro-starfield (thousands of distant pinpricks)
    vec3 p1 = dir * 340.0;
    vec3 id1 = floor(p1);
    vec3 f1 = fract(p1) - 0.5;
    float h1_1 = hash31(id1);
    float h1_2 = hash31(id1 + 17.3);
    vec3 offset1 = (vec3(h1_1, h1_2, hash31(id1 + 31.7)) - 0.5) * 0.75;
    float d1 = length(f1 - offset1);
    float microStar = pow(h1_1, 9.0) * smoothstep(0.18, 0.0, d1) * 1.5;
    vec3 col1 = mix(vec3(0.8, 0.9, 1.0), vec3(1.0, 0.88, 0.75), h1_2);
    col += col1 * microStar;

    // Layer 2: Medium prominent stars with astronomical spectral classifications
    vec3 p2 = dir * 150.0;
    vec3 id2 = floor(p2);
    vec3 f2 = fract(p2) - 0.5;
    float h2_1 = hash31(id2 + 43.1);
    float h2_2 = hash31(id2 + 89.7);
    float h2_3 = hash31(id2 + 113.3);
    vec3 offset2 = (vec3(h2_1, h2_2, h2_3) - 0.5) * 0.72;
    float d2 = length(f2 - offset2);
    float medStar = pow(h2_1, 16.0) * smoothstep(0.11, 0.0, d2) * 5.5;

    // Spectral distribution: O/B Blue Supergiant, A/F White, G Yellow, K/M Red Giant
    vec3 col2 = vec3(1.0);
    if (h2_2 > 0.72) {
      col2 = vec3(0.68, 0.85, 1.0) * 1.35; // Hot Blue Supergiant
    } else if (h2_2 > 0.42) {
      col2 = vec3(1.0, 0.98, 0.94);        // Diamond White
    } else if (h2_2 > 0.18) {
      col2 = vec3(1.0, 0.88, 0.65);        // Solar Yellow
    } else {
      col2 = vec3(1.0, 0.42, 0.22) * 1.45; // Cool Red Giant
    }
    col += col2 * medStar;

    // Layer 3: Rare bright 1st-magnitude stars with subtle optical diffraction spikes
    vec3 p3 = dir * 58.0;
    vec3 id3 = floor(p3);
    vec3 f3 = fract(p3) - 0.5;
    float h3_1 = hash31(id3 + 221.7);
    float h3_2 = hash31(id3 + 319.1);
    if (h3_1 > 0.965) {
      vec3 offset3 = (vec3(h3_1, h3_2, hash31(id3 + 171.9)) - 0.5) * 0.6;
      vec3 diff = f3 - offset3;
      float d3 = length(diff);
      float coreStar = smoothstep(0.08, 0.0, d3) * 16.0;
      // 4-point IMAX anamorphic optical spike
      float spike = (smoothstep(0.025, 0.0, abs(diff.x)) * smoothstep(0.38, 0.0, abs(diff.y)) +
                     smoothstep(0.025, 0.0, abs(diff.y)) * smoothstep(0.38, 0.0, abs(diff.x))) * 3.5;
      vec3 starTint = mix(vec3(0.72, 0.90, 1.0), vec3(1.0, 0.75, 0.45), h3_2);
      col += starTint * (coreStar + spike);
    }

    // Layer 4: Distant spiral galaxy cluster smudge
    vec3 galDir = normalize(vec3(-0.45, 0.55, -0.70));
    float galDist = length(dir - galDir);
    if (galDist < 0.28) {
      float galIntensity = smoothstep(0.28, 0.0, galDist) * 0.35;
      col += vec3(0.55, 0.65, 0.85) * galIntensity;
    }

    // Layer 5: Cosmic nebulae and deep space dust lanes
    float core = max(0.0, dot(dir, normalize(vec3(0.65, 0.15, -0.75))));
    float n1 = sin(dir.x * 2.2 + dir.y * 1.6) * cos(dir.z * 2.2);
    float n2 = sin(dir.x * 4.5 - dir.z * 3.8) * cos(dir.y * 4.5);
    float dust = clamp(0.5 + 0.3 * n1 + 0.2 * n2, 0.0, 1.0);

    vec3 deepSpace = vec3(0.0015, 0.003, 0.007);
    vec3 cosmicDust = vec3(0.022, 0.010, 0.030);
    vec3 galacticCore = vec3(0.045, 0.022, 0.010);

    vec3 nebula = mix(deepSpace, cosmicDust, dust);
    nebula += galacticCore * pow(core, 3.2) * 0.5;

    return col + nebula;
  }

  // ── Interstellar Gargantua Color Palette (Smooth, Continuous Hermite Curve) ──
  vec3 getAccretionPalette(float temp, int scheme) {
    if (scheme == 0) {
      // 🌟 Gargantua (Interstellar): Incandescent Cream -> Pure Solar Gold -> Warm Amber -> Smoldering Bronze
      vec3 cWhite = vec3(1.0, 0.98, 0.93) * 1.7;
      vec3 cGold  = vec3(1.0, 0.80, 0.26) * 1.45;
      vec3 cAmber = vec3(1.0, 0.50, 0.07) * 1.15;
      vec3 cBronze= vec3(0.68, 0.20, 0.02) * 0.90;
      vec3 cDark  = vec3(0.18, 0.03, 0.005);

      // Smooth Hermite interpolation to guarantee zero creases or sharp color jumps
      float t1 = smoothstep(0.0, 0.22, temp);
      float t2 = smoothstep(0.22, 0.52, temp);
      float t3 = smoothstep(0.52, 0.82, temp);
      float t4 = smoothstep(0.82, 1.0, temp);

      vec3 col = mix(cDark, cBronze, t1);
      col = mix(col, cAmber, t2);
      col = mix(col, cGold, t3);
      col = mix(col, cWhite, t4);
      return col;

    } else if (scheme == 1) {
      // 🕳️ M87* / Sagittarius A*: Deep amber synchrotron radiation
      vec3 cWhite = vec3(1.0, 0.92, 0.75) * 1.5;
      vec3 cAmber = vec3(1.0, 0.55, 0.05) * 1.3;
      vec3 cRust  = vec3(0.80, 0.22, 0.01) * 0.9;
      vec3 cDark  = vec3(0.20, 0.02, 0.005);

      float t1 = smoothstep(0.0, 0.40, temp);
      float t2 = smoothstep(0.40, 0.75, temp);
      float t3 = smoothstep(0.75, 1.0, temp);

      vec3 col = mix(cDark, cRust, t1);
      col = mix(col, cAmber, t2);
      col = mix(col, cWhite, t3);
      return col;

    } else if (scheme == 2) {
      // ⚡ Cyan Quasar: High-energy relativistic plasma
      vec3 cWhite  = vec3(0.95, 1.0, 1.0) * 1.7;
      vec3 cCyan   = vec3(0.20, 0.85, 1.0) * 1.4;
      vec3 cBlue   = vec3(0.12, 0.38, 0.95) * 1.1;
      vec3 cPurple = vec3(0.38, 0.08, 0.65) * 0.8;
      vec3 cDark   = vec3(0.06, 0.01, 0.12);

      float t1 = smoothstep(0.0, 0.25, temp);
      float t2 = smoothstep(0.25, 0.50, temp);
      float t3 = smoothstep(0.50, 0.80, temp);
      float t4 = smoothstep(0.80, 1.0, temp);

      vec3 col = mix(cDark, cPurple, t1);
      col = mix(col, cBlue, t2);
      col = mix(col, cCyan, t3);
      col = mix(col, cWhite, t4);
      return col;

    } else {
      // 🔥 Superflare: Pulsating shock fronts
      vec3 cFlash = vec3(1.0, 1.0, 1.0) * 2.2;
      vec3 cLemon = vec3(1.0, 0.95, 0.4) * 1.6;
      vec3 cPeach = vec3(1.0, 0.52, 0.1) * 1.2;
      vec3 cBlood = vec3(0.70, 0.06, 0.02) * 0.85;

      float t1 = smoothstep(0.0, 0.35, temp);
      float t2 = smoothstep(0.35, 0.70, temp);
      float t3 = smoothstep(0.70, 1.0, temp);

      vec3 col = mix(cBlood, cPeach, t1);
      col = mix(col, cLemon, t2);
      col = mix(col, cFlash, t3);
      return col;
    }
  }

  void main() {
    // 1. Reconstruct camera ray in 3D world space
    vec2 ndc = (gl_FragCoord.xy / uResolution.xy) * 2.0 - 1.0;
    vec4 target = uProjInverse * vec4(ndc, 1.0, 1.0);
    vec3 rdView = normalize(target.xyz / target.w);
    vec3 rd = normalize((uCameraMatrixWorld * vec4(rdView, 0.0)).xyz);
    vec3 ro = uCameraPos;

    // 2. Analytical Bounding Sphere Optimization (R = 8.8)
    // Fast-forwards ray directly to the active lensing sphere, saving steps in empty space
    float R_BOUND = 8.8;
    float roDist = length(ro);
    if (roDist > R_BOUND) {
      float b = dot(ro, rd);
      float c = roDist * roDist - R_BOUND * R_BOUND;
      float disc = b * b - c;
      if (disc < 0.0) {
        // Ray misses the entire black hole system: render background directly
        vec3 sky = getRichStarfield(rd);
        gl_FragColor = vec4(sky, 1.0);
        return;
      }
      float tEnter = -b - sqrt(disc);
      if (tEnter > 0.0) {
        ro += rd * (tEnter - 0.02);
      }
    }

    // 3. Initialize Geodesic Raytracer
    vec3 pos = ro;
    vec3 vel = rd;

    float rs = uRs;
    float diskInner = uDiskInner;
    float diskOuter = uDiskOuter;

    vec3 accumColor = vec3(0.0);
    float accumAlpha = 0.0;
    bool hitHorizon = false;

    // Miller's Planet position (orbiting on the upper-left of Gargantua outside the inner disk)
    float planetAngle = 2.45 + uTime * 0.04;
    vec3 planetPos = vec3(cos(planetAngle) * 3.5, 0.42, sin(planetAngle) * 2.8);
    float planetRadius = 0.095;

    // 4. Runge-Kutta 2 (Midpoint) Geodesic Integration Loop
    const int MAX_STEPS = 100;
    for (int i = 0; i < MAX_STEPS; i++) {
      float r = length(pos);

      // Event horizon entry
      if (r < rs) {
        hitHorizon = true;
        break;
      }

      // Escaped outside bounding sphere and traveling outward
      if (r > R_BOUND && dot(pos, vel) > 0.0) {
        break;
      }

      // ── Miller's Planet Ray Intersection ──
      if (uShowPlanet) {
        float distToPlanet = length(pos - planetPos);
        if (distToPlanet < planetRadius) {
          // Silhouette of Miller's Planet with thin atmospheric limb
          vec3 planetNormal = normalize(pos - planetPos);
          float limb = 1.0 - max(0.0, dot(planetNormal, -vel));
          limb = pow(limb, 3.5);
          vec3 planetSurface = vec3(0.025, 0.03, 0.04);
          vec3 atmosphereLimb = vec3(0.20, 0.60, 0.95); // Cyan oceanic atmosphere
          vec3 planetCol = mix(planetSurface, atmosphereLimb, limb * 0.75);

          accumColor += (1.0 - accumAlpha) * planetCol;
          accumAlpha = 1.0;
          break;
        }
      }

      // Adaptive step size: ultra-fine near horizon, larger far away
      float dt = clamp(0.042 * r, 0.018, 0.32);

      // Conserved Angular Momentum L = pos x vel
      vec3 L = cross(pos, vel);
      float h2 = dot(L, L);

      // Step 1: Accel at pos
      float aMag1 = -1.5 * rs * h2 / (pow(r, 5.0) + 0.00005) * uLensingStrength;
      vec3 a1 = pos * aMag1;

      // Midpoint state
      vec3 midPos = pos + vel * (dt * 0.5);
      vec3 midVel = normalize(vel + a1 * (dt * 0.5));
      float midR = length(midPos);

      if (midR < rs) {
        hitHorizon = true;
        break;
      }

      // Step 2: Accel at midPos
      vec3 midL = cross(midPos, midVel);
      float midH2 = dot(midL, midL);
      float aMag2 = -1.5 * rs * midH2 / (pow(midR, 5.0) + 0.00005) * uLensingStrength;
      vec3 a2 = midPos * aMag2;

      // Full step position and velocity
      vec3 nextPos = pos + midVel * dt;
      vec3 nextVel = normalize(vel + a2 * dt);

      // ── Accretion Disk Equatorial Plane Intersection ──
      // Checks whenever the ray crosses y = 0
      if (pos.y * nextPos.y < 0.0) {
        float t = -pos.y / (nextPos.y - pos.y);
        vec3 xHit = mix(pos, nextPos, clamp(t, 0.0, 1.0));
        float rHit = length(xHit.xz);

        if (rHit >= diskInner && rHit <= diskOuter) {
          // ── 1. Seamless 2D Rotational Coordinates (Eliminates atan branch-cut seam!) ──
          // Differential Keplerian angular rotation Omega(r) ~ r^(-1.25)
          float omega = (uSpeed * 0.65) / (pow(rHit, 1.25) + 0.1);
          float angle = -uTime * omega;
          float cA = cos(angle);
          float sA = sin(angle);
          // Pure 2D rotation of world position: 100% mathematically continuous 360° with ZERO SEAMS
          vec2 pRot = vec2(xHit.x * cA - xHit.z * sA, xHit.x * sA + xHit.z * cA);

          // Spiral coordinate winding (logarithmic spiral transformation)
          float twist = log(rHit) * 3.6;
          float cTwist = cos(twist);
          float sTwist = sin(twist);
          vec2 pSpiral = vec2(pRot.x * cTwist - pRot.y * sTwist, pRot.x * sTwist + pRot.y * cTwist);

          // ── 2. Multi-Scale Concentric Ringlets (The Saturn / Vinyl Record Micro-Bands) ──
          float rFreq = rHit;
          float band1 = sin(rFreq * 34.0);
          float band2 = sin(rFreq * 82.0 + 1.2);
          float band3 = sin(rFreq * 175.0 + 2.4);
          float band4 = sin(rFreq * 390.0 + 0.8);
          float band5 = sin(rFreq * 780.0 + 3.1);
          float ringlets = (band1 * 0.28 + band2 * 0.28 + band3 * 0.24 + band4 * 0.12 + band5 * 0.08) * 0.5 + 0.5;
          ringlets = pow(ringlets, 1.35);

          // ── 3. Plasma Streamers & Dark Absorption Dust Lanes ──
          float turb = fbm(pSpiral * 1.5);
          float dustLanes = smoothstep(0.32, 0.78, fbm(pSpiral * 2.8 - vec2(3.1, 1.5)));

          // ── 4. Relativistic Doppler Beaming (Smooth & Cinematic) ──
          float beta = sqrt(rs / (2.0 * max(rHit, rs)));
          beta = clamp(beta, 0.05, 0.58);

          // Unit orbital tangent vector
          vec3 ePhi = vec3(-xHit.z, 0.0, xHit.x) / max(rHit, 0.001);
          vec3 kObs = -normalize(vel);
          float betaParallel = beta * dot(ePhi, kObs) * uDopplerStrength;
          betaParallel = clamp(betaParallel, -0.65, 0.65);

          float gamma = 1.0 / sqrt(max(0.01, 1.0 - beta * beta * uDopplerStrength));
          float doppler = 1.0 / (gamma * (1.0 - betaParallel));
          float gravRedshift = sqrt(clamp(1.0 - rs / max(rHit, rs * 1.01), 0.05, 1.0));
          float shift = doppler * gravRedshift;

          // Beaming intensity
          float beaming = pow(shift, 2.2);
          beaming = clamp(beaming, 0.35, 2.8);

          // Radial temperature profile
          float temp = pow(diskInner / rHit, 0.65) * pow(clamp(1.0 - sqrt(diskInner / rHit), 0.0, 1.0), 0.25);
          temp = clamp(temp * 2.2, 0.05, 1.0) * shift;

          // Color palette with smooth Hermite interpolation
          vec3 diskColor = getAccretionPalette(temp, uColorScheme);

          // Modulate with concentric micro-ringlets and turbulent filaments
          diskColor *= (0.72 + 0.38 * ringlets + 0.22 * turb);
          // Dark absorption dust lanes
          diskColor *= (0.42 + 0.58 * dustLanes);

          // Soft boundaries at inner ISCO edge and outer rim
          float innerEdge = smoothstep(diskInner, diskInner + 0.22, rHit);
          float outerEdge = smoothstep(diskOuter, diskOuter - 1.2, rHit);
          float edgeMask = innerEdge * outerEdge;

          // Optical depth path length (grazing angles pass through more thickness)
          float pathFactor = 1.0 / max(abs(vel.y), 0.18);
          pathFactor = clamp(pathFactor, 1.0, 2.8);

          float hitAlpha = clamp(edgeMask * (0.68 + 0.3 * ringlets) * (pathFactor * 0.45), 0.0, 0.95);
          float hitLum = beaming * uDiskBrightness * edgeMask;

          accumColor += (1.0 - accumAlpha) * diskColor * hitLum;
          accumAlpha += (1.0 - accumAlpha) * hitAlpha;

          if (accumAlpha > 0.96) {
            break;
          }
        }
      }

      pos = nextPos;
      vel = nextVel;
    }

    // 5. Final Composite
    if (hitHorizon) {
      // Event Horizon: Absolute pure cosmic void
      accumColor = mix(accumColor, vec3(0.0), 1.0 - accumAlpha);
    } else {
      // Ray escaped: sample background starfield with gravitational lensing
      vec3 skyDir = normalize(vel);
      vec3 background = getRichStarfield(skyDir);

      accumColor += (1.0 - accumAlpha) * background;
    }

    gl_FragColor = vec4(accumColor, 1.0);
  }
`;

// ──────────────────────────────────────────────────────────────
// Custom Fullscreen Shader Quad Component
// ──────────────────────────────────────────────────────────────

interface BlackHoleQuadProps {
  rs: number;
  diskInner: number;
  diskOuter: number;
  lensingStrength: number;
  dopplerStrength: number;
  diskBrightness: number;
  speed: number;
  colorScheme: number;
  showPlanet: boolean;
}

function BlackHoleQuad({
  rs,
  diskInner,
  diskOuter,
  lensingStrength,
  dopplerStrength,
  diskBrightness,
  speed,
  colorScheme,
  showPlanet,
}: BlackHoleQuadProps) {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  const { camera, size } = useThree();

  const uniforms = useMemo(
    () => ({
      uCameraPos: { value: new THREE.Vector3() },
      uCameraMatrixWorld: { value: new THREE.Matrix4() },
      uProjInverse: { value: new THREE.Matrix4() },
      uResolution: { value: new THREE.Vector2(size.width, size.height) },
      uTime: { value: 0 },
      uRs: { value: rs },
      uDiskInner: { value: diskInner },
      uDiskOuter: { value: diskOuter },
      uLensingStrength: { value: lensingStrength },
      uDopplerStrength: { value: dopplerStrength },
      uDiskBrightness: { value: diskBrightness },
      uSpeed: { value: speed },
      uColorScheme: { value: colorScheme },
      uShowPlanet: { value: showPlanet },
    }),
    []
  );

  useEffect(() => {
    if (materialRef.current) {
      materialRef.current.uniforms.uResolution.value.set(size.width, size.height);
    }
  }, [size.width, size.height]);

  useEffect(() => {
    if (materialRef.current) {
      materialRef.current.uniforms.uRs.value = rs;
      materialRef.current.uniforms.uDiskInner.value = diskInner;
      materialRef.current.uniforms.uDiskOuter.value = diskOuter;
      materialRef.current.uniforms.uLensingStrength.value = lensingStrength;
      materialRef.current.uniforms.uDopplerStrength.value = dopplerStrength;
      materialRef.current.uniforms.uDiskBrightness.value = diskBrightness;
      materialRef.current.uniforms.uSpeed.value = speed;
      materialRef.current.uniforms.uColorScheme.value = colorScheme;
      materialRef.current.uniforms.uShowPlanet.value = showPlanet;
    }
  }, [rs, diskInner, diskOuter, lensingStrength, dopplerStrength, diskBrightness, speed, colorScheme, showPlanet]);

  useFrame((_, delta) => {
    if (!materialRef.current) return;
    materialRef.current.uniforms.uTime.value += delta;
    materialRef.current.uniforms.uCameraPos.value.copy(camera.position);
    materialRef.current.uniforms.uCameraMatrixWorld.value.copy(camera.matrixWorld);
    materialRef.current.uniforms.uProjInverse.value.copy(camera.projectionMatrixInverse);
  });

  return (
    <mesh ref={meshRef} frustumCulled={false}>
      <planeGeometry args={[2, 2]} />
      <shaderMaterial
        ref={materialRef}
        vertexShader={raymarchVertexShader}
        fragmentShader={raymarchFragmentShader}
        uniforms={uniforms}
        depthWrite={false}
        depthTest={false}
      />
    </mesh>
  );
}

// ──────────────────────────────────────────────────────────────
// Web Audio API Synthesized Cosmic Drone Generator
// ──────────────────────────────────────────────────────────────

class CosmicAudioEngine {
  private ctx: AudioContext | null = null;
  private oscSub: OscillatorNode | null = null;
  private oscHum: OscillatorNode | null = null;
  private gainNode: GainNode | null = null;
  private filter: BiquadFilterNode | null = null;
  private isPlaying = false;

  public toggle(): boolean {
    if (this.isPlaying) {
      this.stop();
      return false;
    } else {
      this.start();
      return true;
    }
  }

  public start() {
    try {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.ctx = new AudioCtx();

      // Deep sub-bass oscillator (38 Hz)
      this.oscSub = this.ctx.createOscillator();
      this.oscSub.type = 'sine';
      this.oscSub.frequency.setValueAtTime(38, this.ctx.currentTime);

      // Low resonant harmonic hum (76 Hz)
      this.oscHum = this.ctx.createOscillator();
      this.oscHum.type = 'triangle';
      this.oscHum.frequency.setValueAtTime(76, this.ctx.currentTime);

      // Resonant Lowpass Filter
      this.filter = this.ctx.createBiquadFilter();
      this.filter.type = 'lowpass';
      this.filter.frequency.setValueAtTime(110, this.ctx.currentTime);
      this.filter.Q.setValueAtTime(3.5, this.ctx.currentTime);

      this.gainNode = this.ctx.createGain();
      this.gainNode.gain.setValueAtTime(0.001, this.ctx.currentTime);
      this.gainNode.gain.exponentialRampToValueAtTime(0.25, this.ctx.currentTime + 2.0);

      this.oscSub.connect(this.filter);
      this.oscHum.connect(this.filter);
      this.filter.connect(this.gainNode);
      this.gainNode.connect(this.ctx.destination);

      this.oscSub.start();
      this.oscHum.start();
      this.isPlaying = true;
    } catch {
      // Audio context might fail if not permitted
    }
  }

  public stop() {
    if (this.gainNode && this.ctx) {
      this.gainNode.gain.setValueAtTime(this.gainNode.gain.value, this.ctx.currentTime);
      this.gainNode.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + 1.2);
      setTimeout(() => {
        this.oscSub?.stop();
        this.oscHum?.stop();
        this.ctx?.close();
        this.isPlaying = false;
      }, 1200);
    } else {
      this.isPlaying = false;
    }
  }
}

// ──────────────────────────────────────────────────────────────
// Camera Telemetry & Controller
// ──────────────────────────────────────────────────────────────

interface CameraTrackerProps {
  onTelemetry: (dist: number, pitch: number, fps: number) => void;
  autoRotate: boolean;
}

function CameraTracker({ onTelemetry, autoRotate }: CameraTrackerProps) {
  const { camera } = useThree();
  const lastTime = useRef(performance.now());
  const frameCount = useRef(0);
  const currentFps = useRef(60);

  useFrame(() => {
    frameCount.current++;
    const now = performance.now();
    if (now - lastTime.current >= 500) {
      currentFps.current = Math.round((frameCount.current * 1000) / (now - lastTime.current));
      frameCount.current = 0;
      lastTime.current = now;

      const dist = camera.position.length();
      const pitch = (Math.asin(camera.position.y / dist) * 180) / Math.PI;
      onTelemetry(dist, pitch, currentFps.current);
    }
  });

  return (
    <OrbitControls
      enableDamping
      dampingFactor={0.06}
      minDistance={3.5}
      maxDistance={22.0}
      enablePan={false}
      autoRotate={autoRotate}
      autoRotateSpeed={0.25}
      maxPolarAngle={Math.PI * 0.92}
      minPolarAngle={Math.PI * 0.08}
    />
  );
}

// ──────────────────────────────────────────────────────────────
// Main Exported Component
// ──────────────────────────────────────────────────────────────

export default function BlackHole() {
  const [presetIndex, setPresetIndex] = useState(0);

  // Exact Gargantua physics parameters matching the reference IMAX shot
  const [rs, setRs] = useState(1.0);
  const [diskInner, setDiskInner] = useState(1.25);
  const [diskOuter, setDiskOuter] = useState(6.8);
  const [lensingStrength, setLensingStrength] = useState(1.0);
  const [dopplerStrength, setDopplerStrength] = useState(0.35); // Balanced Interstellar movie look
  const [diskBrightness, setDiskBrightness] = useState(1.85);
  const [speed, setSpeed] = useState(0.8);
  const [bloomIntensity, setBloomIntensity] = useState(0.65);
  const [showPlanet, setShowPlanet] = useState(true); // Miller's planet
  const [autoRotate, setAutoRotate] = useState(true);

  // Telemetry HUD state
  const [cameraDist, setCameraDist] = useState(9.2);
  const [cameraPitch, setCameraPitch] = useState(14.0);
  const [fps, setFps] = useState(60);
  const [isAudioActive, setIsAudioActive] = useState(false);
  const [isHudOpen, setIsHudOpen] = useState(false); // Collapsed by default for pure cinema

  const audioEngine = useMemo(() => new CosmicAudioEngine(), []);

  const handleTelemetry = useCallback((dist: number, pitch: number, currentFps: number) => {
    setCameraDist(dist);
    setCameraPitch(pitch);
    setFps(currentFps);
  }, []);

  const toggleAudio = () => {
    const active = audioEngine.toggle();
    setIsAudioActive(active);
  };

  // Presets definition
  const presets = [
    {
      name: 'Gargantua (Interstellar)',
      description: 'Rich incandescent golden-amber disk with smooth continuous arching, Saturn-like ringlets, and Miller planet.',
      scheme: 0,
      rs: 1.0,
      inner: 1.25,
      outer: 6.8,
      lensing: 1.0,
      doppler: 0.35,
      brightness: 1.85,
      speed: 0.8,
      bloom: 0.65,
    },
    {
      name: 'M87* / Sagittarius A*',
      description: 'Deep amber submillimeter synchrotron radiation glow matching the Event Horizon Telescope data.',
      scheme: 1,
      rs: 1.1,
      inner: 1.35,
      outer: 6.2,
      lensing: 1.05,
      doppler: 0.45,
      brightness: 1.6,
      speed: 0.7,
      bloom: 0.55,
    },
    {
      name: 'Cyan Quasar / Pulsar',
      description: 'Ultra-relativistic super-heated plasma disk radiating extreme ultraviolet and electric cyan photons.',
      scheme: 2,
      rs: 0.95,
      inner: 1.2,
      outer: 7.2,
      lensing: 1.0,
      doppler: 0.4,
      brightness: 2.0,
      speed: 1.1,
      bloom: 0.75,
    },
    {
      name: 'Superflare Shockwave',
      description: 'Hyper-luminous magnetohydrodynamic flare event with blinding plasma bursts and shockwaves.',
      scheme: 3,
      rs: 1.0,
      inner: 1.25,
      outer: 7.0,
      lensing: 1.15,
      doppler: 0.6,
      brightness: 2.4,
      speed: 1.3,
      bloom: 0.9,
    },
  ];

  const applyPreset = (index: number) => {
    setPresetIndex(index);
    const p = presets[index];
    setRs(p.rs);
    setDiskInner(p.inner);
    setDiskOuter(p.outer);
    setLensingStrength(p.lensing);
    setDopplerStrength(p.doppler);
    setDiskBrightness(p.brightness);
    setSpeed(p.speed);
    setBloomIntensity(p.bloom);
  };

  return (
    <div className="relative w-full h-full overflow-hidden select-none bg-black font-sans">
      {/* ── 3D WebGL Canvas ── */}
      <Canvas
        camera={{
          position: [-1.8, 1.6, 9.0], // Cinematic 3/4 Interstellar perspective
          fov: 52,
          near: 0.1,
          far: 100,
        }}
        gl={{
          antialias: true,
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1.0,
          powerPreference: 'high-performance',
        }}
        style={{ width: '100%', height: '100%' }}
      >
        {/* Fullscreen Geodesic Raymarching Mesh */}
        <BlackHoleQuad
          rs={rs}
          diskInner={diskInner}
          diskOuter={diskOuter}
          lensingStrength={lensingStrength}
          dopplerStrength={dopplerStrength}
          diskBrightness={diskBrightness}
          speed={speed}
          colorScheme={presets[presetIndex].scheme}
          showPlanet={showPlanet}
        />

        {/* Cinematic Post-Processing Bloom: Soft highlights only, no blinding overexposure */}
        <EffectComposer multisampling={0}>
          <Bloom
            intensity={bloomIntensity}
            luminanceThreshold={0.72}
            luminanceSmoothing={0.45}
            mipmapBlur
          />
        </EffectComposer>

        {/* Orbit Controls & Telemetry Tracker */}
        <CameraTracker onTelemetry={handleTelemetry} autoRotate={autoRotate} />
      </Canvas>

      {/* ── Top Left: Title & Mission Badge ── */}
      <div className="absolute top-5 left-5 z-20 pointer-events-none flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-500 animate-pulse shadow-[0_0_10px_#f59e0b]" />
          <h1 className="text-sm font-mono tracking-widest uppercase text-white/90 font-bold">
            GARGANTUA // INTERSTELLAR
          </h1>
        </div>
        <p className="text-xs text-white/50 font-mono tracking-wide">
          General Relativity Geodesic Simulation • Miller&apos;s Planet System
        </p>
      </div>

      {/* ── Top Right: Quick Action Controls ── */}
      <div className="absolute top-5 right-5 z-20 flex items-center gap-2.5">
        <button
          onClick={() => setShowPlanet(!showPlanet)}
          className={`px-3 py-1.5 rounded-full text-xs font-mono tracking-wider transition-all duration-300 backdrop-blur-md border ${
            showPlanet
              ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/60 shadow-[0_0_12px_rgba(6,182,212,0.3)]'
              : 'bg-white/5 text-white/60 border-white/10 hover:bg-white/10 hover:text-white'
          }`}
          title="Toggle Miller's Planet"
        >
          {showPlanet ? '🪐 PLANET ON' : '🪐 PLANET OFF'}
        </button>

        <button
          onClick={toggleAudio}
          className={`px-3 py-1.5 rounded-full text-xs font-mono tracking-wider transition-all duration-300 backdrop-blur-md border ${
            isAudioActive
              ? 'bg-amber-500/20 text-amber-300 border-amber-500/60 shadow-[0_0_12px_rgba(245,158,11,0.3)]'
              : 'bg-white/5 text-white/60 border-white/10 hover:bg-white/10 hover:text-white'
          }`}
          title="Toggle Deep Space Audio Resonance"
        >
          {isAudioActive ? '🔊 AUDIO ON' : '🔇 AUDIO OFF'}
        </button>

        <button
          onClick={() => setAutoRotate(!autoRotate)}
          className={`px-3 py-1.5 rounded-full text-xs font-mono tracking-wider transition-all duration-300 backdrop-blur-md border ${
            autoRotate
              ? 'bg-amber-500/20 text-amber-300 border-amber-500/60 shadow-[0_0_12px_rgba(245,158,11,0.3)]'
              : 'bg-white/5 text-white/60 border-white/10 hover:bg-white/10 hover:text-white'
          }`}
          title="Toggle Cinematic Auto-Orbit"
        >
          {autoRotate ? '🌀 ORBIT ON' : '⏸️ ORBIT OFF'}
        </button>

        <button
          onClick={() => setIsHudOpen(!isHudOpen)}
          className="px-3.5 py-1.5 rounded-full text-xs font-mono tracking-wider bg-white/10 text-white/80 border border-white/15 hover:bg-white/20 transition-all duration-200 backdrop-blur-md"
        >
          {isHudOpen ? '✕ HIDE HUD' : '⚙️ CONTROLS'}
        </button>
      </div>

      {/* ── Bottom Floating Astrophysics HUD & Control Drawer ── */}
      {isHudOpen && (
        <div className="absolute bottom-5 left-5 right-5 z-20 max-w-4xl mx-auto flex flex-col gap-3 p-4 rounded-2xl bg-black/75 backdrop-blur-xl border border-white/10 shadow-2xl transition-all duration-300 text-white">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-3">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-[11px] font-mono text-white/40 uppercase mr-1">Presets:</span>
              {presets.map((p, idx) => (
                <button
                  key={p.name}
                  onClick={() => applyPreset(idx)}
                  className={`px-3 py-1 rounded-lg text-xs font-mono transition-all duration-200 ${
                    presetIndex === idx
                      ? 'bg-amber-500/30 text-amber-300 border border-amber-500/60 shadow-[0_0_8px_rgba(245,158,11,0.4)]'
                      : 'bg-white/5 text-white/60 hover:bg-white/10 hover:text-white border border-transparent'
                  }`}
                >
                  {p.name.split(' ')[0]}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-3 text-[11px] font-mono text-white/60">
              <div>
                <span className="text-white/40">FPS:</span>{' '}
                <span className={fps >= 50 ? 'text-emerald-400' : 'text-amber-400'}>{fps}</span>
              </div>
              <div>
                <span className="text-white/40">DIST:</span>{' '}
                <span className="text-amber-300">{cameraDist.toFixed(1)} Rs</span>
              </div>
              <div>
                <span className="text-white/40">INCL:</span>{' '}
                <span className="text-amber-300">{cameraPitch.toFixed(0)}°</span>
              </div>
              <div>
                <span className="text-white/40">MILLER:</span>{' '}
                <span className="text-cyan-300">{showPlanet ? 'ORBITING' : 'OFF'}</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-1 text-xs font-mono">
            <div className="flex flex-col gap-1 bg-white/5 p-2 rounded-lg border border-white/5">
              <div className="flex justify-between items-center text-white/70">
                <span>Doppler Beaming</span>
                <span className="text-amber-400">{dopplerStrength.toFixed(2)}x</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={dopplerStrength}
                onChange={(e) => setDopplerStrength(parseFloat(e.target.value))}
                className="w-full h-1 bg-white/20 rounded-lg appearance-none cursor-pointer accent-amber-500"
              />
              <span className="text-[10px] text-white/40">Balanced relativistic asymmetry</span>
            </div>

            <div className="flex flex-col gap-1 bg-white/5 p-2 rounded-lg border border-white/5">
              <div className="flex justify-between items-center text-white/70">
                <span>Lensing Deflection</span>
                <span className="text-amber-400">{lensingStrength.toFixed(1)}x</span>
              </div>
              <input
                type="range"
                min="0.4"
                max="1.6"
                step="0.05"
                value={lensingStrength}
                onChange={(e) => setLensingStrength(parseFloat(e.target.value))}
                className="w-full h-1 bg-white/20 rounded-lg appearance-none cursor-pointer accent-amber-500"
              />
              <span className="text-[10px] text-white/40">Spacetime curvature</span>
            </div>

            <div className="flex flex-col gap-1 bg-white/5 p-2 rounded-lg border border-white/5">
              <div className="flex justify-between items-center text-white/70">
                <span>Luminosity & Bloom</span>
                <span className="text-amber-400">{diskBrightness.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.8"
                max="3.0"
                step="0.1"
                value={diskBrightness}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  setDiskBrightness(val);
                  setBloomIntensity(val * 0.35);
                }}
                className="w-full h-1 bg-white/20 rounded-lg appearance-none cursor-pointer accent-amber-500"
              />
              <span className="text-[10px] text-white/40">Incandescent emission</span>
            </div>

            <div className="flex flex-col gap-1 bg-white/5 p-2 rounded-lg border border-white/5">
              <div className="flex justify-between items-center text-white/70">
                <span>Rotation Velocity</span>
                <span className="text-amber-400">{speed.toFixed(1)}x</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="2.5"
                step="0.1"
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
                className="w-full h-1 bg-white/20 rounded-lg appearance-none cursor-pointer accent-amber-500"
              />
              <span className="text-[10px] text-white/40">Differential Keplerian flow</span>
            </div>
          </div>

          <div className="flex items-center justify-between text-[11px] text-white/40 font-mono border-t border-white/5 pt-2">
            <span>{presets[presetIndex].description}</span>
            <span className="hidden md:inline text-white/30">
              Drag to Orbit • Scroll to Zoom • Continuous Geodesic RK2
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
